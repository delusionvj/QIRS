import os
import time
import json
import pickle
import random
import logging
import networkx as nx
import numpy as np
import pandas as pd
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Dict, Any, Tuple, Optional, List, Union
from tqdm.auto import tqdm

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import softmax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_hybrid_model")

# Fix seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

###############################################################################
#                       Enhanced Graph Data Processor                         #
###############################################################################

class EnhancedGraphDataProcessor:
    """
    Enhanced processor that handles GraphML with user nodes included
    """
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.logger = logging.getLogger("enhanced_hybrid_model.processor")
        
    def load_graphml(self, path: str) -> nx.Graph:
        """Load graph from GraphML file with robustness against errors"""
        self.logger.info(f"Loading graph from {path}")
        
        try:
            return nx.read_graphml(path)
        except Exception as e:
            self.logger.warning(f"Error loading GraphML: {str(e)}. Trying with node_type as string...")
            # Try with node_type enforcement
            return nx.read_graphml(path, node_type=str)
    
    def process_nx_graph(self, nx_graph: nx.Graph, embedding_dim=128) -> Tuple[Data, Dict]:
        """
        Convert NetworkX to PyG data with enhanced feature processing
        that properly handles user nodes directly from GraphML
        """
        self.logger.info(f"Processing graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
        
        # Start with basic mappings
        node_list = list(nx_graph.nodes())
        node_mapping = {node: i for i, node in enumerate(node_list)}
        
        # Enhanced node type management
        node_types = []
        node_type_to_idx = {}
        node_idx_to_type = {}
        
        # First pass - collect node types
        for node, data in nx_graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            if node_type not in node_type_to_idx:
                node_type_to_idx[node_type] = len(node_type_to_idx)
        
        # Second pass - assign types
        for node, data in nx_graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            node_idx = node_mapping[node]
            node_idx_to_type[node_idx] = node_type
            node_types.append(node_type_to_idx[node_type])
        
        # Track user and movie indices for recommendation
        user_indices = []
        movie_indices = []
        
        # Create node features
        node_features = {}
        community_ids = {}
        node_years = {}
        
        # Collections for dimensionality reduction of embeddings
        all_embeddings = []
        
        # Process each node
        for node, data in nx_graph.nodes(data=True):
            node_idx = node_mapping[node]
            feats = []
            
            # 1. Node type one-hot encoding
            node_type = data.get("node_type", "unknown").lower()
            type_oh = [0] * len(node_type_to_idx)
            type_oh[node_type_to_idx[node_type]] = 1
            feats.extend(type_oh)
            
            # Track user and movie indices
            if node_type == "user":
                user_indices.append(node_idx)
            elif node_type == "movie":
                movie_indices.append(node_idx)
            
            # 2. Community information if available
            community = data.get("community_id")
            if community is not None:
                try:
                    community_ids[node_idx] = int(community)
                except (ValueError, TypeError):
                    pass
            
            # 3. Temporal information
            year_val = data.get("year")
            timestamp = None
            
            # Process year with various formats
            if isinstance(year_val, str) and year_val.isdigit():
                year_val = int(year_val)
                timestamp = year_val
            elif isinstance(year_val, int):
                timestamp = year_val
            
            # Try alternative date formats if year not found
            if timestamp is None:
                for date_field in ["release_date", "created_at", "timestamp"]:
                    date_val = data.get(date_field)
                    if date_val:
                        # Try to extract year from date string
                        if isinstance(date_val, str):
                            import re
                            year_match = re.search(r'\b(19\d\d|20\d\d)\b', date_val)
                            if year_match:
                                timestamp = int(year_match.group(1))
                                break
            
            if timestamp:
                # Normalize year to 0-1 range (1900-2023)
                norm_year = (timestamp - 1900) / (2023 - 1900)
                # Recency feature (exponential decay)
                recency = np.exp((timestamp - 2023) / 10)
                
                feats.append(norm_year)
                feats.append(recency)
                node_years[node_idx] = timestamp
            else:
                # Default temporal features
                feats.append(0.5)  # Default normalized year
                feats.append(0.5)  # Default recency
            
            # 4. Additional type-specific features
            if node_type == "user":
                # User-specific features - adding placeholders
                # You can expand this with actual user features if available
                feats.append(1.0)  # User indicator
                feats.append(0.0)  # Movie indicator
                feats.append(0.5)  # Default value
            elif node_type == "movie":
                # Movie-specific features
                feats.append(0.0)  # User indicator
                feats.append(1.0)  # Movie indicator
                feats.append(0.5)  # Default value
            else:
                # Default padding
                feats.append(0.0)  # User indicator  
                feats.append(0.0)  # Movie indicator
                feats.append(0.5)  # Default value
            
            # 5. Handle embeddings
            if "embedding" in data:
                try:
                    emb_data = data["embedding"]
                    if isinstance(emb_data, str):
                        # Maybe hex string
                        raw_bytes = bytes.fromhex(emb_data)
                        embedding = np.frombuffer(raw_bytes, dtype=np.float32)
                        all_embeddings.append((node_idx, embedding))
                    elif isinstance(emb_data, list):
                        embedding = np.array(emb_data, dtype=np.float32)
                        all_embeddings.append((node_idx, embedding))
                except Exception as e:
                    self.logger.warning(f"Error processing embedding: {e}")
            
            # Add placeholder for embeddings
            feats.extend([0.0] * embedding_dim)
            
            node_features[node_idx] = feats
        
        # Apply dimensionality reduction to embeddings if we have any
        if all_embeddings:
            self.logger.info(f"Applying dimensionality reduction to {len(all_embeddings)} embeddings")
            
            # Get indices and embeddings
            embed_indices = [idx for idx, _ in all_embeddings]
            embed_arrays = [emb for _, emb in all_embeddings]
            
            # Check dimensions
            dims = [arr.shape[0] for arr in embed_arrays]
            if len(set(dims)) > 1:
                self.logger.warning(f"Embeddings have different dimensions: {set(dims)}")
                # Resize all to minimum dimension
                min_dim = min(dims)
                embed_arrays = [arr[:min_dim] if arr.shape[0] > min_dim else arr for arr in embed_arrays]
            
            # Apply dimensionality reduction
            try:
                # Try using UMAP if available (better for preserving structure)
                try:
                    from umap import UMAP
                    reducer = UMAP(n_components=embedding_dim, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embed_arrays)
                    self.logger.info("Used UMAP for dimensionality reduction")
                except ImportError:
                    # Fallback to PCA
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=embedding_dim)
                    reduced_embeddings = reducer.fit_transform(embed_arrays)
                    self.logger.info("Used PCA for dimensionality reduction")
                
                # Update node features with reduced embeddings
                for i, node_idx in enumerate(embed_indices):
                    if node_idx in node_features:
                        feats = node_features[node_idx]
                        # Replace placeholder zeros with reduced embeddings
                        embed_start = len(feats) - embedding_dim
                        feats[embed_start:] = reduced_embeddings[i].tolist()
                        node_features[node_idx] = feats
            
            except Exception as e:
                self.logger.warning(f"Dimensionality reduction failed: {e}. Using simple averaging instead.")
                # Fallback to simple averaging
                for i, node_idx in enumerate(embed_indices):
                    if node_idx in node_features:
                        feats = node_features[node_idx]
                        embed_start = len(feats) - embedding_dim
                        
                        # Use simple averaging for dimensionality reduction
                        arr = embed_arrays[i]
                        if len(arr) > embedding_dim:
                            chunks = np.array_split(arr, embedding_dim)
                            reduced = np.array([chunk.mean() for chunk in chunks])
                        else:
                            reduced = np.pad(arr, (0, max(0, embedding_dim - len(arr))))
                        
                        feats[embed_start:] = reduced.tolist()
                        node_features[node_idx] = feats
        
        # Create node features tensor
        x = torch.tensor([node_features[i] for i in range(len(node_list))], dtype=torch.float)
        
        # Process edges with enhanced features
        edge_list = []
        edge_attr_list = []
        edge_type_list = []
        
        # Map edge types to integers
        edge_type_to_idx = {}
        
        # Collect edge types first
        for u, v, edata in nx_graph.edges(data=True):
            e_type = edata.get("edge_type", "unknown").lower()
            if e_type not in edge_type_to_idx:
                edge_type_to_idx[e_type] = len(edge_type_to_idx)
        
        # Now process edges
        for u, v, edata in nx_graph.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            
            # Get edge type
            e_type = edata.get("edge_type", "unknown").lower()
            e_type_idx = edge_type_to_idx[e_type]
            
            # Add both directions (for undirected graphs)
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])
            
            edge_type_list.append(e_type_idx)
            edge_type_list.append(e_type_idx)
            
            # Basic edge features
            weight = float(edata.get("weight", 1.0))
            
            # Get interaction rating if this is a user-movie edge
            rating = 0.0
            if e_type == "rated" or e_type == "interacted_with":
                rating = float(edata.get("rating", edata.get("value", 0.0)))
            
            # Add year difference if temporal data is available
            year_diff = 0.0
            temporal_context = 0.5
            
            if u_idx in node_years and v_idx in node_years:
                year_diff = abs(node_years[u_idx] - node_years[v_idx]) / 100.0
                # Add temporal context (average year)
                avg_year = (node_years[u_idx] + node_years[v_idx]) / 2
                temporal_context = (avg_year - 1900) / (2023 - 1900)
            
            # Create edge features
            e_feats = [weight, rating, year_diff, temporal_context]
            
            # Add for both directions
            edge_attr_list.append(e_feats)
            edge_attr_list.append(e_feats)
        
        # Create edge tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
        
        # Add node type information
        data.node_type = torch.tensor(node_types, dtype=torch.long)
        
        # Add community information if available
        if community_ids:
            # Get max community ID for one-hot encoding
            max_community = max(community_ids.values()) + 1
            
            # Create community tensor (default to 0 for nodes without community)
            communities = torch.zeros(len(node_list), dtype=torch.long)
            for node_idx, comm_id in community_ids.items():
                communities[node_idx] = comm_id
            
            data.node_community = communities
        
        # Store mapping information
        data.node_mapping = node_mapping
        data.node_list = node_list
        data.node_type_map = node_idx_to_type
        data.edge_type_map = {idx: e_type for e_type, idx in edge_type_to_idx.items()}
        
        # Store user and movie indices
        data.user_indices = torch.tensor(user_indices, dtype=torch.long)
        data.movie_indices = torch.tensor(movie_indices, dtype=torch.long)
        
        self.logger.info(f"Processed graph: {len(node_list)} nodes, {edge_index.size(1)} edges, " 
                         f"{len(edge_type_to_idx)} edge types, {len(node_type_to_idx)} node types, "
                         f"{len(user_indices)} users, {len(movie_indices)} movies")
        
        return data, node_idx_to_type
    
    def extract_user_item_interactions(self, data: Data, rating_edge_types=None):
        """
        Extract user-item interactions from the graph directly
        This works when users are already nodes in the GraphML
        """
        if rating_edge_types is None:
            rating_edge_types = ["rated", "interacted_with", "watched"]
        
        # Debug output to see what edge types are available
        print("Debug - Edge type map:", data.edge_type_map)
        
        # Convert rating_edge_types to indices based on the edge_type_map
        rating_type_indices = []
        for idx, edge_type in data.edge_type_map.items():
            edge_type_str = str(edge_type).lower()
            if edge_type_str in [rt.lower() for rt in rating_edge_types]:
                rating_type_indices.append(idx)
                print(f"Found rating edge type: {edge_type} with index {idx}")
        
        if not rating_type_indices:
            self.logger.warning(f"No rating edge types found matching {rating_edge_types}")
            
            # Additional fallback: try to find any edge connecting users and movies
            print("Attempting fallback: looking for any edges between users and movies...")
            # Get user and movie indices
            user_indices = data.user_indices
            movie_indices = data.movie_indices
            
            # Create sets for faster lookup
            user_set = set(user_indices.tolist())
            movie_set = set(movie_indices.tolist())
            
            # Go through all edges to find user-movie connections
            user_movie_edges = []
            
            edge_index = data.edge_index
            edge_type = data.edge_type
            
            for i in range(edge_index.size(1)):
                source = edge_index[0, i].item()
                target = edge_index[1, i].item()
                
                if (source in user_set and target in movie_set):
                    user_movie_edges.append((source, target, edge_type[i].item()))
            
            print(f"Found {len(user_movie_edges)} user-movie edges in total")
            
            if user_movie_edges:
                # Use all edge types that connect users to movies
                edge_types_found = set([et for _, _, et in user_movie_edges])
                rating_type_indices = list(edge_types_found)
                print(f"Using {len(rating_type_indices)} edge types as rating edges: {rating_type_indices}")
            
            if not rating_type_indices:
                return data
        
        # Get user and movie indices
        user_indices = data.user_indices
        movie_indices = data.movie_indices
        
        # Create a set for fast lookup
        user_set = set(user_indices.tolist())
        movie_set = set(movie_indices.tolist())
        
        # Filter edges to only include user-movie interactions
        edge_index = data.edge_index
        edge_type = data.edge_type
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        
        user_items = []
        ratings = []
        
        # Go through all edges
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            
            # Check if this is a user->movie edge of a rating type
            if (source in user_set and target in movie_set and 
                edge_type[i].item() in rating_type_indices):
                
                user_items.append([source, target])
                
                # Extract rating from edge attributes if available
                rating = 0.0
                if edge_attr is not None:
                    # Assume rating is stored in the second position of edge_attr
                    # This matches our edge feature construction
                    rating = edge_attr[i, 1].item()
                
                ratings.append(rating)
        
        if not user_items:
            self.logger.warning("No user-movie interactions found")
            return data
        
        # Add to data
        data.user_items = torch.tensor(user_items, dtype=torch.long)
        data.ratings = torch.tensor(ratings, dtype=torch.float)
        
        self.logger.info(f"Extracted {len(ratings)} user-movie interactions from graph")
        return data
    
    def add_external_ratings(self, data: Data, ratings_df: pd.DataFrame, 
                            movie_mapping: Dict[str, str]) -> Data:
        """
        Add external ratings data to enhance the graph
        Can be used when there are additional ratings not in the graph
        """
        existing_ui_pairs = set()
        
        # If there are existing user-items, track them to avoid duplicates
        if hasattr(data, "user_items"):
            for ui in data.user_items:
                existing_ui_pairs.add((ui[0].item(), ui[1].item()))
        
        user_items = []
        rating_list = []
        
        for _, row in ratings_df.iterrows():
            user_id = row["userId"]
            movie_id = row["movieId"]
            rating = row["rating"]
            
            # Find user in graph
            user_node = None
            for node, idx in data.node_mapping.items():
                node_type = data.node_type_map.get(idx)
                if node_type == "user" and str(user_id) in str(node):
                    user_node = idx
                    break
            
            if user_node is None:
                continue
            
            # Find movie in graph
            if movie_id in movie_mapping:
                graph_movie_id = movie_mapping[movie_id]
                if graph_movie_id in data.node_mapping:
                    movie_idx = data.node_mapping[graph_movie_id]
                    
                    # Check if this pair already exists
                    if (user_node, movie_idx) not in existing_ui_pairs:
                        user_items.append([user_node, movie_idx])
                        rating_list.append(rating)
                        existing_ui_pairs.add((user_node, movie_idx))
        
        if not user_items:
            self.logger.warning("No external ratings added")
            return data
        
        # Combine with existing ratings if any
        if hasattr(data, "user_items"):
            user_items_tensor = torch.tensor(user_items, dtype=torch.long)
            ratings_tensor = torch.tensor(rating_list, dtype=torch.float)
            
            data.user_items = torch.cat([data.user_items, user_items_tensor], dim=0)
            data.ratings = torch.cat([data.ratings, ratings_tensor], dim=0)
        else:
            data.user_items = torch.tensor(user_items, dtype=torch.long)
            data.ratings = torch.tensor(rating_list, dtype=torch.float)
        
        self.logger.info(f"Added {len(rating_list)} external ratings to data")
        return data
    
    def split_data(self, data: Data, val_ratio=0.1, test_ratio=0.1, 
                  stratify_by_user=True, min_user_ratings=3):
        """Split data into train/val/test with stratification"""
        if not hasattr(data, "user_items"):
            self.logger.warning("No ratings in data, returning single dataset only")
            return data, None, None
        
        user_items = data.user_items.numpy()
        ratings = data.ratings.numpy()
        
        if stratify_by_user and len(user_items) > 0:
            # Group ratings by user for better stratification
            user_to_items = {}
            for i, (user, item) in enumerate(user_items):
                if user not in user_to_items:
                    user_to_items[user] = []
                user_to_items[user].append((i, item, ratings[i]))
            
            # Split indices
            train_idx, val_idx, test_idx = [], [], []
            
            for user, items in user_to_items.items():
                if len(items) < min_user_ratings:
                    # Too few ratings, put all in training
                    train_idx.extend([i for i, _, _ in items])
                    continue
                
                # Shuffle user's ratings
                random.shuffle(items)
                
                # Calculate split sizes
                n_test = max(1, int(len(items) * test_ratio))
                n_val = max(1, int(len(items) * val_ratio))
                n_train = len(items) - n_test - n_val
                
                # Ensure at least one item in train
                if n_train <= 0:
                    n_train = 1
                    if n_val > 1:
                        n_val -= 1
                    else:
                        n_test -= 1
                
                # Add to splits
                train_idx.extend([i for i, _, _ in items[:n_train]])
                val_idx.extend([i for i, _, _ in items[n_train:n_train+n_val]])
                test_idx.extend([i for i, _, _ in items[n_train+n_val:]])
        else:
            # Simple random split
            n = len(user_items)
            indices = list(range(n))
            random.shuffle(indices)
            
            test_size = int(test_ratio * n)
            val_size = int(val_ratio * n)
            train_size = n - test_size - val_size
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
        
        # Convert to tensors
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
        
        # Create dataset subsets
        def subset_data(base, subset_idx):
            d = Data()
            # Copy all attributes from base
            for key in base.keys():
                if key not in ['user_items', 'ratings']:
                    d[key] = base[key]
            
            # Add subset of ratings data
            d.user_items = base.user_items[subset_idx]
            d.ratings = base.ratings[subset_idx]
            return d
        
        train_data = subset_data(data, train_idx)
        val_data = subset_data(data, val_idx)
        test_data = subset_data(data, test_idx)
        
        self.logger.info(f"Split data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return train_data, val_data, test_data

###############################################################################
#                  Enhanced Attention-based ODE Function                      #
###############################################################################

class EnhancedODEFunction(nn.Module):
    """
    Enhanced ODE function with multi-head attention and gated skip connections
    """
    def __init__(self, hidden_dim, edge_index, edge_attr=None, edge_type=None, 
                dropout=0.1, attention_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_type = edge_type
        
        # Use TransformerConv for more powerful attention mechanism
        self.transformer = TransformerConv(
            hidden_dim, 
            hidden_dim // attention_heads,
            heads=attention_heads,
            dropout=dropout,
            edge_dim=edge_attr.size(1) if edge_attr is not None else None
        )
        
        # Skip connection with gating
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Normalization and activation
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, t, x):
        # Store original for residual and gating
        identity = x
        
        # Apply transformer convolution
        if self.edge_attr is not None:
            h = self.transformer(x, self.edge_index, self.edge_attr)
        else:
            h = self.transformer(x, self.edge_index)
        
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        
        # Process skip connection
        skip = self.skip_proj(identity)
        
        # Calculate adaptive gate
        gate_input = torch.cat([h, identity], dim=1)
        gate = self.gate(gate_input)
        
        # Apply gated skip connection
        h = gate * h + (1 - gate) * skip
        
        # Final normalization
        h = self.norm2(h)
        
        return h

###############################################################################
#                   Enhanced Neural ODE Block                                 #
###############################################################################

class EnhancedNeuralODEBlock(nn.Module):
    """
    Enhanced Neural ODE block with improved integration scheme
    """
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_attr=None, 
                edge_type=None, time_steps=6, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ODE function
        self.odefunc = EnhancedODEFunction(
            hidden_dim=hidden_dim,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            dropout=dropout
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Time step weighting - learnable parameters
        self.time_weights = nn.Parameter(torch.ones(time_steps))
        
        self.time_steps = time_steps
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Apply input projection
        h = self.in_proj(x)
        
        # Run ODE integration
        # Instead of using torchdiffeq package, implement manually for better control
        dt = 1.0 / (self.time_steps - 1) if self.time_steps > 1 else 1.0
        
        # Track all states in trajectory
        trajectory = [h]
        current = h
        
        # Run integration steps
        for step in range(1, self.time_steps):
            # Compute derivative
            t = step * dt
            dh = self.odefunc(t, current)
            
            # Euler update
            current = current + dh * dt
            
            # Store in trajectory
            trajectory.append(current)
        
        # Stack trajectory
        trajectory = torch.stack(trajectory, dim=0)
        
        # Apply learned weights to time steps
        time_weights = F.softmax(self.time_weights, dim=0).reshape(-1, 1, 1)
        h_weighted = torch.sum(trajectory * time_weights, dim=0)
        
        # Apply output projection
        out = self.out_proj(h_weighted)
        
        return out

###############################################################################
#                    Enhanced Quantum-Inspired Block                          #
###############################################################################

class ComplexGATLayer(nn.Module):
    """
    Quantum-inspired graph attention layer with complex-valued operations
    """
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, edge_dim=None):
        super().__init__()
        
        # Use standard GAT for attention mechanism
        self.gat = GATConv(
            in_dim, 
            out_dim // heads, 
            heads=heads, 
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Phase shift parameters (one per output dimension)
        self.phase = nn.Parameter(torch.randn(out_dim) * 0.02)
        
        # Normalization and nonlinearity
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Apply GAT layer
        if edge_attr is not None:
            h = self.gat(x, edge_index, edge_attr)
        else:
            h = self.gat(x, edge_index)
        
        # Apply complex phase shift (quantum-inspired operation)
        # Calculate real and imaginary components
        h_real = h * torch.cos(self.phase)
        h_imag = h * torch.sin(self.phase)
        
        # Convert back to real domain using complex magnitude
        h = torch.sqrt(h_real.pow(2) + h_imag.pow(2) + 1e-12)
        
        # Apply normalization and nonlinearity
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        
        return h

class EnhancedQuantumBlock(nn.Module):
    """
    Enhanced quantum-inspired block with multiple complex-valued layers
    """
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_attr=None, 
                edge_type=None, n_layers=4, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Complex-valued graph attention layers
        self.quantum_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.quantum_layers.append(ComplexGATLayer(
                hidden_dim,
                hidden_dim,
                heads=4,
                dropout=dropout,
                edge_dim=edge_attr.size(1) if edge_attr is not None else None
            ))
        
        # Global phase shift (entanglement-inspired)
        self.global_phase = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Apply input projection
        h = self.in_proj(x)
        
        # Apply quantum layers with residual connections
        for i, layer in enumerate(self.quantum_layers):
            # Store input for residual
            h_in = h
            
            # Apply quantum layer
            h = layer(h, self.edge_index, self.edge_attr)
            
            # Add residual connection after first layer
            if i > 0:
                h = h + h_in
        
        # Apply global phase operation (simulating entanglement)
        h_real = h * torch.cos(self.global_phase)
        h_imag = h * torch.sin(self.global_phase)
        h = torch.sqrt(h_real.pow(2) + h_imag.pow(2) + 1e-12)
        
        # Apply final output projection
        out = self.out_proj(h)
        
        return out

###############################################################################
#                Enhanced Hybrid GNN Recommender                              #
###############################################################################

class EnhancedHybridGNNRecommender(nn.Module):
    """
    Enhanced hybrid GNN recommender that combines Neural ODE and Quantum-inspired components
    """
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_attr=None, edge_type=None,
                time_steps=6, q_layers=4, dropout=0.1):
        super().__init__()
        
        # 1. Neural ODE Block
        self.ode_block = EnhancedNeuralODEBlock(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,  # Intermediate output
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            time_steps=time_steps,
            dropout=dropout
        )
        
        # 2. Quantum Block
        self.quantum_block = EnhancedQuantumBlock(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            n_layers=q_layers,
            dropout=dropout
        )
        
        # 3. Skip connection
        self.skip_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. Adaptive gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Process through ODE block
        h_ode = self.ode_block(x)
        
        # Process through quantum block
        h_q = self.quantum_block(h_ode)
        
        # Process through skip connection
        h_skip = self.skip_proj(x)
        
        # Compute adaptive gates
        gates = self.gate_net(x)
        
        # Combine outputs with learned gates
        out = gates[:, 0].unsqueeze(1) * h_q + gates[:, 1].unsqueeze(1) * h_skip
        
        return out
    
    def predict_rating(self, node_embeddings, user_idx, movie_idx):
        """
        Predict ratings for given user-movie pairs
        """
        # Get user and movie embeddings
        user_emb = node_embeddings[user_idx]
        movie_emb = node_embeddings[movie_idx]
        
        # Compute dot product as rating predictor
        # Normalize embeddings first for better stability
        user_emb_norm = F.normalize(user_emb, p=2, dim=1)
        movie_emb_norm = F.normalize(movie_emb, p=2, dim=1)
        
        # Compute similarity scores
        scores = torch.sum(user_emb_norm * movie_emb_norm, dim=1)
        
        # Map to [-1, 1] range using tanh
        ratings = torch.tanh(scores)
        
        return ratings

###############################################################################
#                   Enhanced Recommender Trainer                              #
###############################################################################

class EnhancedRecommenderTrainer:
    """
    Enhanced trainer for GNN-based recommenders with optimized training
    """
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5, device=None):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        
        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup and cosine decay
        self.scheduler = None  # Will be initialized during training
        
        self.logger = logging.getLogger("enhanced_hybrid_model.trainer")
    
    def train(self, data, val_data=None, epochs=70, batch_size=128, patience=10, 
             warmup_epochs=5, val_freq=1):
        """
        Train the model with enhanced training loop
        """
        if not hasattr(data, "user_items"):
            raise ValueError("No user_items in data - can't train recommender!")
        
        # Move data to device
        x = data.x.to(self.device)
        user_items = data.user_items.to(self.device)
        ratings = data.ratings.to(self.device)
        
        # Initialize scheduler
        total_steps = epochs * (len(user_items) + batch_size - 1) // batch_size
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=warmup_epochs / epochs,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Training history
        history = {
            "train_loss": [],
            "train_mse": [],
            "train_mae": [],
            "val_loss": [],
            "val_mse": [],
            "val_mae": [],
            "lr": []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        self.model.train()
        
        # Main epoch loop with progress bar
        epoch_bar = tqdm(range(1, epochs + 1), desc="Training")
        
        for epoch in epoch_bar:
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history["lr"].append(current_lr)
            
            # Shuffle indices
            idx = torch.randperm(len(user_items))
            ui_shuffled = user_items[idx]
            r_shuffled = ratings[idx]
            
            # Track metrics
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_mae = 0.0
            
            # Number of batches
            n_batches = (len(ui_shuffled) + batch_size - 1) // batch_size
            
            # Batch loop with progress bar
            batch_bar = tqdm(range(n_batches), desc=f"Epoch {epoch}", leave=False)
            
            for b in batch_bar:
                # Get batch
                start = b * batch_size
                end = min(start + batch_size, len(ui_shuffled))
                
                b_ui = ui_shuffled[start:end]
                b_r = r_shuffled[start:end]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                node_embs = self.model(x)
                u_idx = b_ui[:, 0]
                m_idx = b_ui[:, 1]
                
                # Predict ratings
                preds = self.model.predict_rating(node_embs, u_idx, m_idx)
                
                # Calculate loss
                # Combined loss: MSE + Huber loss for robustness
                mse_loss = F.mse_loss(preds, b_r)
                huber_loss = F.smooth_l1_loss(preds, b_r, beta=0.1)
                loss = 0.7 * mse_loss + 0.3 * huber_loss
                
                # Calculate MAE for tracking
                mae = F.l1_loss(preds, b_r)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update scheduler
                self.scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_mse += mse_loss.item()
                epoch_mae += mae.item()
                
                # Update batch progress bar
                batch_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{mse_loss.item():.4f}',
                    'mae': f'{mae.item():.4f}'
                })
            
            # Calculate epoch average metrics
            epoch_loss /= n_batches
            epoch_mse /= n_batches
            epoch_mae /= n_batches
            
            # Update history
            history["train_loss"].append(epoch_loss)
            history["train_mse"].append(epoch_mse)
            history["train_mae"].append(epoch_mae)
            
            # Validation
            val_metrics = {}
            if val_data is not None and epoch % val_freq == 0:
                val_loss, val_mse, val_mae = self.evaluate(val_data)
                
                history["val_loss"].append(val_loss)
                history["val_mse"].append(val_mse)
                history["val_mae"].append(val_mae)
                
                val_metrics = {
                    'val_loss': f'{val_loss:.4f}',
                    'val_mse': f'{val_mse:.4f}',
                    'val_mae': f'{val_mae:.4f}'
                }
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss
                    }
                    
                    # Save to file
                    torch.save(best_model_state, 'best_enhanced_model.pt')
                else:
                    patience_counter += 1
                    
                    # Check patience
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
            
            # Update epoch progress bar
            epoch_metrics = {
                'loss': f'{epoch_loss:.4f}',
                'mse': f'{epoch_mse:.4f}',
                'mae': f'{epoch_mae:.4f}',
                'lr': f'{current_lr:.6f}'
            }
            epoch_metrics.update(val_metrics)
            
            epoch_bar.set_postfix(epoch_metrics)
            
            # Log to file
            self.logger.info(f"Epoch {epoch}: " + " ".join([f"{k}={v}" for k, v in epoch_metrics.items()]))
        
        # Load best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {best_model_state['epoch']} with val_loss={best_model_state['val_loss']:.4f}")
        
        return history
    
    def evaluate(self, data, batch_size=256):
        """
        Evaluate the model on validation or test data
        """
        self.model.eval()
        
        # Move data to device
        x = data.x.to(self.device)
        user_items = data.user_items.to(self.device)
        ratings = data.ratings.to(self.device)
        
        # Tracking metrics
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        
        # Number of batches
        n_batches = (len(user_items) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            # Compute all node embeddings once
            node_embs = self.model(x)
            
            for b in range(n_batches):
                # Get batch
                start = b * batch_size
                end = min(start + batch_size, len(user_items))
                
                b_ui = user_items[start:end]
                b_r = ratings[start:end]
                
                # Get user and movie indices
                u_idx = b_ui[:, 0]
                m_idx = b_ui[:, 1]
                
                # Predict ratings
                preds = self.model.predict_rating(node_embs, u_idx, m_idx)
                
                # Calculate metrics
                mse = F.mse_loss(preds, b_r)
                mae = F.l1_loss(preds, b_r)
                loss = 0.7 * mse + 0.3 * F.smooth_l1_loss(preds, b_r, beta=0.1)
                
                # Update totals
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
        
        # Calculate averages
        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_mae = total_mae / n_batches
        
        return avg_loss, avg_mse, avg_mae
    
    def save(self, path, data=None, embed_path=None):
        """
        Save model weights and optionally generate embeddings
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
        
        # Save embeddings if requested
        if data is not None and embed_path is not None:
            self.model.eval()
            with torch.no_grad():
                x = data.x.to(self.device)
                node_embs = self.model(x).cpu().numpy()
                
                # Create dictionary mapping node IDs to embeddings
                emb_dict = {}
                for i, node_id in enumerate(data.node_list):
                    emb_dict[node_id] = node_embs[i]
                
                # Save to file
                with open(embed_path, 'wb') as f:
                    pickle.dump(emb_dict, f)
                
                self.logger.info(f"Embeddings saved to {embed_path}")
    
    def load(self, path):
        """
        Load model from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {path}")

###############################################################################
#                          Main Pipeline                                      #
###############################################################################

def train_enhanced_hybrid_model(
    graphml_path,
    output_dir="./output_enhanced",
    rating_edge_types=None,
    ratings_path=None,
    movies_path=None,
    hidden_dim=128,
    out_dim=64,
    time_steps=6,
    q_layers=4,
    dropout=0.1,
    epochs=50,
    batch_size=128,
    learning_rate=0.001,
    patience=10,
    device=None
):
    """
    Main pipeline to train the enhanced hybrid model
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data processor
    processor = EnhancedGraphDataProcessor(device=device)
    
    # Load and process graph
    nx_graph = processor.load_graphml(graphml_path)
    data, node_type_map = processor.process_nx_graph(nx_graph, embedding_dim=out_dim)
    
    data = processor.extract_user_item_interactions(data, ["rated"])
    print(f"After extraction: {hasattr(data, 'user_items')} user_items, {len(data.ratings) if hasattr(data, 'ratings') else 0} ratings found")
    # Add external ratings if provided
    if ratings_path and movies_path:
        # Load ratings and movies
        if ratings_path.endswith('.json'):
            with open(ratings_path, 'r') as f:
                ratings_data = json.load(f)
                
            # Convert to DataFrame
            ratings_rows = []
            for item in ratings_data:
                if isinstance(item, dict) and "userId" in item and "movieId" in item and "rating" in item:
                    ratings_rows.append(item)
                elif isinstance(item, dict) and "_id" in item and "rated" in item:
                    user_id = item["_id"]
                    for movie_id, rating_value in item["rated"].items():
                        if isinstance(rating_value, list) and len(rating_value) > 0:
                            try:
                                rating = float(rating_value[0])
                                ratings_rows.append({
                                    "userId": user_id,
                                    "movieId": movie_id,
                                    "rating": rating
                                })
                            except (ValueError, TypeError):
                                pass
            
            ratings_df = pd.DataFrame(ratings_rows)
        else:
            ratings_df = pd.read_csv(ratings_path)
        
        # Load movie mapping
        movie_mapping = {}
        if movies_path.endswith('.json'):
            with open(movies_path, 'r') as f:
                movies_data = json.load(f)
            
            # Process different JSON formats
            if isinstance(movies_data, list):
                for movie in movies_data:
                    if "movie_id" in movie and "name" in movie:
                        movie_mapping[movie["movie_id"]] = movie["name"]
            elif isinstance(movies_data, dict):
                for movie_id, movie_data in movies_data.items():
                    if isinstance(movie_data, dict) and "title" in movie_data:
                        movie_mapping[movie_id] = movie_data["title"]
        else:
            movies_df = pd.read_csv(movies_path)
            for _, row in movies_df.iterrows():
                if "movie_id" in row and "title" in row:
                    movie_mapping[row["movie_id"]] = row["title"]
        
        # Add external ratings
        data = processor.add_external_ratings(data, ratings_df, movie_mapping)
    
    # Split data if ratings exist
    train_data, val_data, test_data = data, None, None
    if hasattr(data, "user_items") and len(data.user_items) > 0:
        train_data, val_data, test_data = processor.split_data(
            data, 
            val_ratio=0.1, 
            test_ratio=0.1,
            stratify_by_user=True
        )
    else:
        logger.warning("No ratings found in data, skipping train/val/test split")
    
    # Extract graph structure for model
    in_dim = data.x.size(1)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") else None
    edge_type = data.edge_type.to(device) if hasattr(data, "edge_type") else None
    
    # Initialize model
    model = EnhancedHybridGNNRecommender(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        time_steps=time_steps,
        q_layers=q_layers,
        dropout=dropout
    )
    
    # Initialize trainer
    trainer = EnhancedRecommenderTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        device=device
    )
    
    # Train model if ratings exist
    history = None
    if hasattr(train_data, "user_items") and len(train_data.user_items) > 0:
        logger.info("Starting training...")
        
        # Record training time
        start_time = time.time()
        
        # Train model
        history = trainer.train(
            data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # Calculate training time
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate on test set
        if test_data is not None:
            test_loss, test_mse, test_mae = trainer.evaluate(test_data)
            logger.info(f"Test results: loss={test_loss:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
        
        # Save training history
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert history values to standard Python types for JSON serialization
            serializable_history = {}
            for key, values in history.items():
                serializable_history[key] = [float(val) for val in values]
            
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    else:
        logger.warning("No ratings found, skipping training")
    
    # Save model and embeddings
    model_path = os.path.join(output_dir, "enhanced_hybrid_model.pt")
    embed_path = os.path.join(output_dir, "node_embeddings.pkl")
    
    trainer.save(model_path, data, embed_path)
    
    # Create model summary
    summary = {
        "model_type": "enhanced_hybrid_gnn",
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "time_steps": time_steps,
        "q_layers": q_layers,
        "dropout": dropout,
        "num_nodes": nx_graph.number_of_nodes(),
        "num_edges": nx_graph.number_of_edges(),
        "num_node_types": len(node_type_map),
        "num_users": len(data.user_indices) if hasattr(data, "user_indices") else 0,
        "num_movies": len(data.movie_indices) if hasattr(data, "movie_indices") else 0,
        "num_ratings": len(data.ratings) if hasattr(data, "ratings") else 0,
        "model_path": model_path,
        "embeddings_path": embed_path
    }
    
    # Add test metrics if available
    if history is not None and test_data is not None:
        summary["test_loss"] = test_loss
        summary["test_mse"] = test_mse
        summary["test_mae"] = test_mae
    
    # Save summary
    summary_path = os.path.join(output_dir, "model_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Model summary saved to {summary_path}")
    
    return model, history, summary

###############################################################################
#                      Interactive Recommendation Functions                   #
###############################################################################

def get_recommendations(model, data, user_node_idx, n=10, exclude_rated=True):
    """
    Get movie recommendations for a specific user
    """
    device = next(model.parameters()).device
    
    # Move data to device
    x = data.x.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get node embeddings
        node_embs = model(x)
        
        # Get user embedding
        user_emb = node_embs[user_node_idx].unsqueeze(0)
        
        # Get movie indices
        movie_indices = data.movie_indices.to(device)
        
        # Get rated movies to exclude
        rated_movies = set()
        if exclude_rated and hasattr(data, "user_items"):
            user_items = data.user_items.cpu().numpy()
            for u, m in user_items:
                if u == user_node_idx:
                    rated_movies.add(m)
        
        # Filter movie indices to exclude rated movies
        if exclude_rated:
            movie_indices = torch.tensor([idx for idx in movie_indices if idx not in rated_movies], 
                                         device=device)
        
        # Get movie embeddings
        movie_embs = node_embs[movie_indices]
        
        # Calculate predicted ratings
        preds = model.predict_rating(node_embs, 
                                     torch.tensor([user_node_idx] * len(movie_indices), device=device),
                                     movie_indices)
        
        # Get top-n movies
        _, indices = torch.topk(preds, min(n, len(preds)))
        
        # Get recommended movie indices
        recommended_indices = movie_indices[indices].cpu().tolist()
        recommended_scores = preds[indices].cpu().tolist()
        
        # Map indices to node IDs
        recommendations = []
        for idx, score in zip(recommended_indices, recommended_scores):
            node_id = data.node_list[idx]
            recommendations.append((node_id, idx, score))
        
        return recommendations

def find_user_node(data, user_id_or_name):
    """
    Find a user node in the graph by ID or name
    """
    for node_idx, node_id in enumerate(data.node_list):
        if user_id_or_name in str(node_id):
            node_type = data.node_type_map.get(node_idx)
            if node_type == "user":
                return node_idx
    
    # Try looser matching if exact match not found
    for node_idx, node_id in enumerate(data.node_list):
        node_type = data.node_type_map.get(node_idx)
        if node_type == "user" and (
            str(user_id_or_name).lower() in str(node_id).lower() or
            any(user_id_or_name in attr for attr in data.x[node_idx].tolist() if isinstance(attr, str))
        ):
            return node_idx
    
    return None

def find_movie_node(data, movie_id_or_title):
    """
    Find a movie node in the graph by ID or title
    """
    for node_idx, node_id in enumerate(data.node_list):
        if movie_id_or_title in str(node_id):
            node_type = data.node_type_map.get(node_idx)
            if node_type == "movie":
                return node_idx
    
    # Try looser matching if exact match not found
    for node_idx, node_id in enumerate(data.node_list):
        node_type = data.node_type_map.get(node_idx)
        if node_type == "movie" and (
            str(movie_id_or_title).lower() in str(node_id).lower() or
            any(movie_id_or_title.lower() in str(attr).lower() for attr in data.x[node_idx].tolist() if isinstance(attr, str))
        ):
            return node_idx
    
    return None

###############################################################################
#                                Main Function                                #
###############################################################################

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Enhanced Hybrid GNN Recommender")
    parser.add_argument("--graphml", type=str, required=True, help="Path to GraphML file with user nodes")
    parser.add_argument("--output", type=str, default="./output_enhanced", help="Output directory")
    parser.add_argument("--ratings", type=str, default=None, help="Path to additional ratings file (optional)")
    parser.add_argument("--movies", type=str, default=None, help="Path to movie mapping file (optional)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--out-dim", type=int, default=128, help="Output dimension size")
    parser.add_argument("--time-steps", type=int, default=6, help="ODE time steps")
    parser.add_argument("--q-layers", type=int, default=4, help="Number of quantum layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Train model
    model, history, summary = train_enhanced_hybrid_model(
        graphml_path=args.graphml,
        output_dir=args.output,
        rating_edge_types=["rated", "interacted_with", "watched"],
        ratings_path=args.ratings,
        movies_path=args.movies,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        time_steps=args.time_steps,
        q_layers=args.q_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=device
    )
    
    logger.info("Training complete!")
    
    if hasattr(summary, "test_mae"):
        logger.info(f"Test MAE: {summary['test_mae']:.4f}")
    
    logger.info(f"Model and embeddings saved to {args.output}")
