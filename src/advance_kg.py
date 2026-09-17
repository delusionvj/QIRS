import os
import time
import json
import pickle
import random
import logging
import re
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, SAGEConv
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter, deque
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta
import math
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quantum_kg")

# Fix seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        # Add more conversions if needed
        return super().default(obj)
#############################################################
#                 Quantum-Inspired Layers                   #
#############################################################

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

class QuantumTransformerLayer(nn.Module):
    """
    Quantum-inspired transformer layer for graph data
    """
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, edge_dim=None):
        super().__init__()
        
        # Transformer layer
        self.transformer = TransformerConv(
            in_dim,
            out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Quantum phase parameters
        self.phase = nn.Parameter(torch.randn(out_dim) * 0.02)
        
        # Superposition parameters (for quantum-inspired mixing)
        self.superposition = nn.Parameter(torch.randn(out_dim, out_dim) * 0.01)
        
        # Normalization and nonlinearity
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Apply transformer layer
        if edge_attr is not None:
            h = self.transformer(x, edge_index, edge_attr)
        else:
            h = self.transformer(x, edge_index)
        
        # Apply quantum superposition (simplified simulation)
        # This creates quantum-inspired mixing of features
        superposition_weights = F.softmax(self.superposition, dim=1)
        h_mixed = torch.matmul(h, superposition_weights)
        
        # Apply complex phase shifts
        h_real = h_mixed * torch.cos(self.phase) 
        h_imag = h_mixed * torch.sin(self.phase)
        
        # Combine real and imaginary parts (simulating quantum measurement)
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
        
        # Complex-valued graph layers
        self.quantum_layers = nn.ModuleList()
        for i in range(n_layers):
            # Alternate between GAT and Transformer layers
            if i % 2 == 0:
                self.quantum_layers.append(ComplexGATLayer(
                    hidden_dim,
                    hidden_dim,
                    heads=4,
                    dropout=dropout,
                    edge_dim=edge_attr.size(1) if edge_attr is not None else None
                ))
            else:
                self.quantum_layers.append(QuantumTransformerLayer(
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

#############################################################
#               Enhanced Neural ODE Block                   #
#############################################################

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

#############################################################
#             Quantum-Enhanced Hybrid GNN                   #
#############################################################

class QuantumEnhancedHybridGNN(nn.Module):
    """
    Quantum-enhanced hybrid GNN that combines Neural ODE and Quantum-inspired blocks
    """
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_attr=None, 
                 edge_type=None, node_type=None, time_steps=6, q_layers=4, dropout=0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_type = edge_type
        self.node_type = node_type
        
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
        
        # 5. Node type embeddings (if provided)
        if node_type is not None:
            num_node_types = int(node_type.max().item()) + 1
            self.node_type_embeddings = nn.Embedding(num_node_types, hidden_dim)
        else:
            self.node_type_embeddings = None
        
        # 6. Conversation context integration
        self.context_integration = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x, context_embedding=None):
        # Add node type embeddings if available
        if self.node_type_embeddings is not None and self.node_type is not None:
            node_type_emb = self.node_type_embeddings(self.node_type)
            x = x + node_type_emb
        
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
        
        # Integrate conversation context if provided
        if context_embedding is not None:
            # Expand context embedding to match node count
            expanded_context = context_embedding.expand(out.size(0), -1)
            
            # Concatenate node embeddings with context
            combined = torch.cat([out, expanded_context], dim=1)
            
            # Process through context integration
            context_enhanced = self.context_integration(combined)
            
            # Add as residual
            out = out + 0.3 * context_enhanced
        
        return out
    
    def predict_rating(self, node_embeddings, user_idx, movie_idx):
        """
        Predict ratings for user-movie pairs
        
        Args:
            node_embeddings: Node embeddings
            user_idx: User indices
            movie_idx: Movie indices
            
        Returns:
            Predicted ratings
        """
        # Get user and movie embeddings
        user_emb = node_embeddings[user_idx]
        movie_emb = node_embeddings[movie_idx]
        
        # Normalize embeddings
        user_emb_norm = F.normalize(user_emb, p=2, dim=1)
        movie_emb_norm = F.normalize(movie_emb, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.sum(user_emb_norm * movie_emb_norm, dim=1)
        
        # Map to [-1, 1] range
        ratings = torch.tanh(similarity)
        
        return ratings

#############################################################
#                  Advanced Meta-Path Reasoning             #
#############################################################

class AdvancedMetaPathReasoner:
    """
    Advanced meta-path reasoning engine for knowledge graphs
    """
    
    def __init__(self, graph):
        """
        Initialize meta-path reasoner
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self.logger = logging.getLogger("quantum_kg.meta_path")
        
        # Extract node types
        self.node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            self.node_types[node] = node_type
        
        # Extract edge types
        self.edge_types = {}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "connected_to").lower()
            self.edge_types[(u, v)] = edge_type
            # Add reverse direction for undirected graph
            self.edge_types[(v, u)] = edge_type
        
        # Meta-path schemas and instances
        self.meta_path_schemas = {}
        self.path_instances = {}
        self.path_semantics = self._initialize_path_semantics()
        
        self.logger.info(f"Advanced meta-path reasoner initialized with {len(set(self.node_types.values()))} node types and {len(set(self.edge_types.values()))} edge types")
    
    def _initialize_path_semantics(self):
        """Initialize semantic mappings for common meta-paths"""
        return {
            "user->rated->movie": "User rated a movie",
            "user->rated->movie->has_genre->genre": "User rated a movie with genre",
            "movie->has_genre->genre->has_genre->movie": "Movies share the same genre",
            "user->rated->movie->acted_in->actor": "User rated a movie with actor",
            "user->rated->movie->directed->director": "User rated a movie with director",
            "movie->acted_in->actor->acted_in->movie": "Movies share an actor",
            "movie->directed->director->directed->movie": "Movies share a director",
            "user->rated->movie->acted_in->actor->acted_in->movie": "User might like a movie featuring an actor from a movie they liked",
            "user->rated->movie->has_genre->genre->has_genre->movie": "User might like a movie in the same genre as one they liked",
            "user->rated->movie->directed->director->directed->movie": "User might like a movie by the same director as one they liked"
        }
    
    def extract_meta_path_schemas(self, max_length=4, sample_nodes=100):
        """
        Extract all meta-path schemas from the graph
        
        Args:
            max_length: Maximum path length
            sample_nodes: Number of nodes to sample for exploration
            
        Returns:
            Dictionary of meta-path schemas
        """
        self.logger.info("Extracting meta-path schemas")
        
        # Sample nodes for each type
        type_to_nodes = defaultdict(list)
        for node, node_type in self.node_types.items():
            type_to_nodes[node_type].append(node)
        
        # Sample starting nodes 
        sample_nodes_list = []
        for node_type, nodes in type_to_nodes.items():
            # Sample nodes of each type
            sampled = random.sample(nodes, min(sample_nodes // len(type_to_nodes), len(nodes)))
            sample_nodes_list.extend(sampled)
        
        # BFS to find meta-paths
        meta_paths = set()
        
        for start_node in tqdm(sample_nodes_list, desc="Extracting meta-paths"):
            visited = set([start_node])
            queue = deque([(start_node, [(self.node_types[start_node],)])])
            
            while queue:
                node, path = queue.popleft()
                
                # Check neighbors
                for neighbor in self.graph.neighbors(node):
                    if len(path) >= max_length:
                        break
                        
                    # Get edge type
                    edge_type = self.edge_types.get((node, neighbor), "connected_to")
                    neighbor_type = self.node_types.get(neighbor, "unknown")
                    
                    # Create new path
                    new_path = path + [(edge_type, neighbor_type)]
                    
                    # Add meta-path to set
                    meta_path_tuple = tuple(item for segment in new_path for item in (segment if isinstance(segment, tuple) else (segment,)))
                    meta_paths.add(meta_path_tuple)
                    
                    # Add to queue if not visited and not too long
                    if neighbor not in visited and len(new_path) < max_length:
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
        
        # Convert to dictionary with schema names
        meta_path_schemas = {}
        for path in meta_paths:
            # Create path name
            path_elements = []
            for j in range(0, len(path), 2):
                if j + 1 < len(path):
                    path_elements.append(f"{path[j]}-{path[j+1]}")
                else:
                    path_elements.append(path[j])
            
            path_name = "->".join(path_elements)
            meta_path_schemas[path_name] = path
        
        self.meta_path_schemas = meta_path_schemas
        self.logger.info(f"Extracted {len(meta_path_schemas)} meta-path schemas")
        
        return meta_path_schemas
    
    def find_paths(self, start_node, end_node, max_length=4, semantic_filtering=True, max_paths=10):
        """
        Find paths between two nodes with advanced filtering
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            semantic_filtering: Whether to filter paths by semantic relevance
            max_paths: Maximum number of paths to return
            
        Returns:
            List of path dictionaries with metadata
        """
        # Check if nodes exist
        if start_node not in self.graph or end_node not in self.graph:
            return []
        
        # Cache key
        cache_key = (start_node, end_node, max_length, semantic_filtering)
        if cache_key in self.path_instances:
            return self.path_instances[cache_key][:max_paths]
        
        try:
            # Use NetworkX to find simple paths
            all_paths = list(nx.all_simple_paths(self.graph, start_node, end_node, cutoff=max_length))
            
            # Convert to meta-path instances with metadata
            path_instances = []
            
            for path in all_paths:
                if len(path) < 2:
                    continue
                
                # Extract meta-path schema
                meta_path = [self.node_types.get(path[0], "unknown")]
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_type = self.edge_types.get((u, v), "connected_to")
                    meta_path.extend([edge_type, self.node_types.get(v, "unknown")])
                
                # Get path name
                path_elements = []
                for j in range(0, len(meta_path), 2):
                    if j + 1 < len(meta_path):
                        path_elements.append(f"{meta_path[j]}-{meta_path[j+1]}")
                    else:
                        path_elements.append(meta_path[j])
                
                path_name = "->".join(path_elements)
                
                # Calculate path significance
                significance = self._calculate_path_significance(path)
                
                # Calculate semantic relevance
                semantic_relevance = 1.0
                semantics = self._get_path_semantics(path_name)
                
                # Check if this is a semantically important path
                if semantic_filtering and semantics == path_name:
                    # No semantic meaning found, decrease relevance
                    semantic_relevance = 0.5
                
                # Calculate overall score
                score = significance * semantic_relevance
                
                # Add to instances
                path_instances.append({
                    "path": path,
                    "meta_path": meta_path,
                    "path_name": path_name,
                    "significance": significance,
                    "semantics": semantics,
                    "semantic_relevance": semantic_relevance,
                    "score": score
                })
            
            # Sort by overall score
            path_instances.sort(key=lambda x: x["score"], reverse=True)
            
            # Cache result
            self.path_instances[cache_key] = path_instances
            
            return path_instances[:max_paths]
            
        except Exception as e:
            self.logger.error(f"Error finding paths: {str(e)}")
            return []
    
    def _calculate_path_significance(self, path):
        """Calculate significance score for a path"""
        if len(path) < 2:
            return 0.0
            
        # Calculate edge significance
        edge_significance = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                edge_data = self.graph.get_edge_data(u, v)
                
                # Base significance from edge weight
                weight = float(edge_data.get("weight", 1.0))
                edge_significance += weight
                
                # Add bonus for specific relationship types
                edge_type = edge_data.get("edge_type", "").lower()
                if edge_type == "rated":
                    # User ratings are significant
                    rating = edge_data.get("rating", 0)
                    if abs(rating) > 0:
                        edge_significance += 0.5  # Bonus for explicit ratings
                elif edge_type in ["acted_in", "directed"]:
                    # Main roles are significant
                    edge_significance += 0.3
                elif edge_type == "has_genre":
                    edge_significance += 0.2
        
        # Normalize by path length
        avg_edge_significance = edge_significance / (len(path) - 1)
        
        # Length penalty (shorter paths are more significant)
        length_penalty = 1.0 / (1.0 + 0.2 * (len(path) - 2))
        
        # Combine factors
        significance = 0.7 * avg_edge_significance + 0.3 * length_penalty
        
        return min(1.0, significance)
    
    def _get_path_semantics(self, path_name):
        """Get semantic description for a meta-path"""
        # Check direct mapping
        if path_name in self.path_semantics:
            return self.path_semantics[path_name]
        
        # Try to generate a description
        try:
            # Split path into elements
            elements = path_name.split("->")
            
            # Generate description based on pattern
            if len(elements) == 3 and elements[0] == "user" and elements[1] == "rated":
                return f"User rated a {elements[2]}"
            elif len(elements) == 5 and elements[0] == "user" and elements[2] == "movie":
                return f"User {elements[1]} a movie that {elements[3]} a {elements[4]}"
            elif len(elements) == 5 and elements[0] == "movie" and elements[4] == "movie":
                return f"Movies share the same {elements[2]}"
            elif len(elements) == 7 and elements[0] == "user" and elements[2] == "movie" and elements[6] == "movie":
                return f"User might like a movie with the same {elements[4]} as a movie they liked"
            else:
                # Generic fallback
                return "A path connecting " + " to ".join(elements[::2])
                
        except Exception as e:
            self.logger.warning(f"Error generating path semantics: {str(e)}")
            return path_name
    
    def generate_explanation(self, path_data, personalize=True):
        """
        Generate natural language explanation for a path
        
        Args:
            path_data: Path data dictionary
            personalize: Whether to personalize the explanation
            
        Returns:
            Explanation string
        """
        if not path_data:
            return "I don't have a good explanation for this recommendation."
            
        path = path_data["path"]
        significance = path_data["significance"]
        semantics = path_data["semantics"]
        
        if len(path) < 2:
            return "This is a direct recommendation."
            
        # Start building explanation
        if personalize:
            if significance > 0.8:
                intro = "I'm very confident that you'll enjoy"
            elif significance > 0.6:
                intro = "I think you might enjoy"
            else:
                intro = "You might be interested in"
        else:
            if significance > 0.8:
                intro = "This is a strong recommendation"
            elif significance > 0.6:
                intro = "This is a good recommendation"
            else:
                intro = "This could be interesting"
        
        # Get path entities
        path_entities = []
        for node in path:
            if node in self.graph:
                name = self.graph.nodes[node].get("name", 
                                                self.graph.nodes[node].get("title", node))
                path_entities.append(name)
        
        # Generate explanation
        explanation = f"{intro} {path_entities[-1]}"
        
        # Add semantic explanation
        if semantics and semantics != path_data["path_name"]:
            explanation += f" because {semantics.lower()}"
            
            # Add entity details for key relationships
            if "share" in semantics.lower():
                # For shared entity patterns, identify the shared entity
                shared_entity_type = path_data["meta_path"][4] if len(path_data["meta_path"]) >= 5 else None
                
                if shared_entity_type:
                    shared_entity_index = 2  # Middle node in a 5-node path
                    if shared_entity_index < len(path):
                        shared_entity = path[shared_entity_index]
                        shared_entity_name = path_entities[shared_entity_index]
                        explanation += f". Both movies are connected to {shared_entity_name}"
                        
            elif "actor" in semantics.lower() and len(path) >= 4:
                # For actor-based recommendations
                actor_index = None
                for i, node in enumerate(path):
                    if self.node_types.get(node) == "actor":
                        actor_index = i
                        break
                
                if actor_index is not None:
                    actor_name = path_entities[actor_index]
                    explanation += f". {actor_name} appears in both movies"
                    
            elif "director" in semantics.lower() and len(path) >= 4:
                # For director-based recommendations
                director_index = None
                for i, node in enumerate(path):
                    if self.node_types.get(node) == "director":
                        director_index = i
                        break
                
                if director_index is not None:
                    director_name = path_entities[director_index]
                    explanation += f". {director_name} directed both movies"
                    
            elif "genre" in semantics.lower() and len(path) >= 4:
                # For genre-based recommendations
                genre_index = None
                for i, node in enumerate(path):
                    if self.node_types.get(node) == "genre":
                        genre_index = i
                        break
                
                if genre_index is not None:
                    genre_name = path_entities[genre_index]
                    explanation += f". Both movies are in the {genre_name} genre"
        else:
            # Generic path explanation
            if len(path) > 2:
                path_str = " → ".join(path_entities)
                explanation += f" based on the connection: {path_str}"
        
        return explanation

#############################################################
#              Spectral Community Detection                 #
#############################################################

import logging
import random
import traceback
import json
import math
import numpy as np
import networkx as nx
from collections import defaultdict, Counter, deque
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class EnhancedCommunityDetector:
    """
    Advanced community detection with hierarchical and overlapping capabilities
    """
    
    def __init__(self, graph):
        """
        Initialize enhanced community detector
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self.logger = logging.getLogger("quantum_kg.enhanced_community")
        
        # Community assignments at multiple levels
        self.communities = {}  # Primary communities
        self.hierarchical_communities = {}  # Hierarchical structure
        self.overlapping_communities = {}  # Overlapping assignments
        self.community_to_nodes = defaultdict(list)  # Reverse mapping
        self.community_profiles = {}  # Community profiles
        self.dendrogram = None  # Full hierarchical structure
        self.subcommunities = {}  # Subcommunities within main communities
        
        # Quality metrics
        self.modularity = 0.0
        self.conductance = {}
        
        # Node types for later analysis
        self.node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            self.node_types[node] = node_type
        
        # Load existing communities if available
        self._load_existing_communities()
        
        self.logger.info(f"Enhanced community detector initialized with {self.graph.number_of_nodes()} nodes")
    
    def _load_existing_communities(self):
        """Load existing community assignments from graph"""
        for node, data in self.graph.nodes(data=True):
            if "community_id" in data:
                comm_id = data["community_id"]
                
                # Update mappings
                self.communities[node] = comm_id
                self.community_to_nodes[comm_id].append(node)
        
        if self.communities:
            self.logger.info(f"Loaded {len(set(self.communities.values()))} existing communities")
    
    def detect_communities(self, method="louvain", resolution=0.8, min_size=3, 
                          hierarchical=True, overlapping=True, levels=3):
        """
        Detect communities using advanced methods
        
        Args:
            method: Detection method ("louvain", "infomap", "label_prop", "spectral", "leiden")
            resolution: Resolution parameter (lower values = larger communities)
            min_size: Minimum community size
            hierarchical: Whether to perform hierarchical detection
            overlapping: Whether to detect overlapping communities
            levels: Number of hierarchy levels
            
        Returns:
            Dictionary mapping nodes to communities
        """
        self.logger.info(f"Detecting communities using {method} method")
        
        # Handle disconnected graphs by processing components separately
        if not nx.is_connected(self.graph):
            self._process_disconnected_graph(method, resolution, min_size)
        else:
            # Apply selected community detection method
            if method == "louvain":
                self._detect_louvain_communities(resolution)
            elif method == "infomap":
                self._detect_infomap_communities()
            elif method == "label_prop":
                self._detect_label_propagation_communities()
            elif method == "spectral":
                self._detect_spectral_communities()
            elif method == "leiden":
                self._detect_leiden_communities(resolution)
            else:
                self._detect_louvain_communities(resolution)
        
        # Verify we don't have trivial communities (each node in its own)
        if len(self.community_to_nodes) > self.graph.number_of_nodes() * 0.8:
            self.logger.warning("Trivial communities detected, retrying with adjusted parameters")
            # Retry with lower resolution for larger communities
            if method == "louvain":
                self._detect_louvain_communities(resolution * 0.5)
            else:
                self._detect_louvain_communities(0.4)  # Fallback to Louvain with low resolution
        
        # Another check - if still highly fragmented, force a merge
        if len(self.community_to_nodes) > self.graph.number_of_nodes() * 0.5:
            self.logger.warning("Still highly fragmented after retry, forcing community merges")
            self._force_community_merges()
                
        # Filter small communities
        if min_size > 1:
            self._merge_small_communities(min_size)
            
        # Update graph with community information
        for node, comm_id in self.communities.items():
            if node in self.graph.nodes:
                self.graph.nodes[node]["community_id"] = comm_id
                
        # Calculate quality metrics
        self._calculate_quality_metrics()
        
        # Detect hierarchical structure if requested
        if hierarchical:
            self._detect_hierarchical_structure(levels, min_size)
            
        # Detect overlapping communities if requested
        if overlapping:
            self._detect_overlapping_communities()
            
        # Generate community profiles
        self._generate_community_profiles()
        
        # Detect subcommunities
        self._detect_subcommunities(resolution * 1.2)
        
        self.logger.info(f"Detected {len(self.community_to_nodes)} communities with modularity {self.modularity:.4f}")
        
        return self.communities
    
    def _process_disconnected_graph(self, method, resolution, min_size):
        """Process disconnected graph by detecting communities in each component"""
        self.logger.info("Processing disconnected graph by components")
        
        # Get connected components
        components = list(nx.connected_components(self.graph))
        self.logger.info(f"Found {len(components)} connected components")
        
        # Process each component
        next_community_id = 0
        for i, component in enumerate(components):
            self.logger.info(f"Processing component {i+1}/{len(components)} with {len(component)} nodes")
            
            # Skip tiny components (assign directly)
            if len(component) < min_size:
                for node in component:
                    self.communities[node] = next_community_id
                    self.community_to_nodes[next_community_id].append(node)
                next_community_id += 1
                continue
            
            # Create subgraph
            subgraph = self.graph.subgraph(component).copy()
            
            # Detect communities in subgraph
            community_map = {}
            
            if method == "louvain":
                community_map = self._detect_louvain_in_subgraph(subgraph, resolution)
            elif method == "infomap":
                community_map = self._detect_infomap_in_subgraph(subgraph)
            elif method == "label_prop":
                community_map = self._detect_label_prop_in_subgraph(subgraph)
            elif method == "spectral":
                community_map = self._detect_spectral_in_subgraph(subgraph)
            elif method == "leiden":
                community_map = self._detect_leiden_in_subgraph(subgraph, resolution)
            else:
                community_map = self._detect_louvain_in_subgraph(subgraph, resolution)
            
            # Remap community IDs to be unique across components
            for node, local_comm_id in community_map.items():
                global_comm_id = local_comm_id + next_community_id
                self.communities[node] = global_comm_id
                self.community_to_nodes[global_comm_id].append(node)
            
            # Update next available community ID
            if community_map:
                next_community_id += max(community_map.values()) + 1
            else:
                # If no communities found, assign all to one community
                for node in component:
                    self.communities[node] = next_community_id
                    self.community_to_nodes[next_community_id].append(node)
                next_community_id += 1
    
    def _detect_louvain_communities(self, resolution):
        """Detect communities using Louvain method"""
        try:
            import community as community_louvain
            
            self.logger.info(f"Running Louvain algorithm with resolution {resolution}")
            
            # Use Louvain algorithm
            partition = community_louvain.best_partition(self.graph, resolution=resolution, random_state=42)
            
            # Clear existing assignments
            self.communities = {}
            self.community_to_nodes = defaultdict(list)
            
            # Add new assignments
            for node, comm_id in partition.items():
                self.communities[node] = comm_id
                self.community_to_nodes[comm_id].append(node)
            
            self.logger.info(f"Louvain method found {len(self.community_to_nodes)} communities")
            
        except ImportError:
            self.logger.warning("community-louvain package not found, falling back to networkx implementation")
            self._detect_label_propagation_communities()
        except Exception as e:
            self.logger.error(f"Error in Louvain algorithm: {str(e)}")
            traceback.print_exc()
            self._detect_label_propagation_communities()
    
    def _detect_louvain_in_subgraph(self, subgraph, resolution):
        """Run Louvain algorithm on a subgraph"""
        try:
            import community as community_louvain
            return community_louvain.best_partition(subgraph, resolution=resolution, random_state=42)
        except (ImportError, Exception) as e:
            self.logger.warning(f"Louvain on subgraph failed: {str(e)}")
            return self._detect_label_prop_in_subgraph(subgraph)
    
    def _detect_infomap_communities(self):
        """Detect communities using Infomap method"""
        try:
            import infomap
            
            self.logger.info("Running Infomap algorithm")
            
            # Create Infomap instance
            im = infomap.Infomap("--two-level")
            
            # Add nodes and edges
            for i, node in enumerate(self.graph.nodes()):
                im.add_node(i)
            
            # Add edges with weights if available
            for u, v, data in self.graph.edges(data=True):
                # Get node indices
                u_idx = list(self.graph.nodes()).index(u)
                v_idx = list(self.graph.nodes()).index(v)
                
                # Get weight if available
                weight = data.get("weight", 1.0)
                
                # Add edge
                im.add_link(u_idx, v_idx, weight)
            
            # Run Infomap
            im.run()
            
            # Clear existing assignments
            self.communities = {}
            self.community_to_nodes = defaultdict(list)
            
            # Create mapping from index to node ID
            idx_to_node = {i: node for i, node in enumerate(self.graph.nodes())}
            
            # Extract communities
            for node, module in im.modules:
                # Convert back to original node ID
                orig_node = idx_to_node[node]
                
                # Assign community
                self.communities[orig_node] = module
                self.community_to_nodes[module].append(orig_node)
            
            self.logger.info(f"Infomap method found {len(self.community_to_nodes)} communities")
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"Infomap not available or failed: {str(e)}")
            self._detect_louvain_communities(resolution=0.8)
    
    def _detect_infomap_in_subgraph(self, subgraph):
        """Run Infomap algorithm on a subgraph"""
        try:
            import infomap
            
            # Create Infomap instance
            im = infomap.Infomap("--two-level")
            
            # Create node index mapping
            node_to_idx = {node: i for i, node in enumerate(subgraph.nodes())}
            idx_to_node = {i: node for node, i in node_to_idx.items()}
            
            # Add nodes
            for i in range(len(node_to_idx)):
                im.add_node(i)
            
            # Add edges
            for u, v, data in subgraph.edges(data=True):
                weight = data.get("weight", 1.0)
                im.add_link(node_to_idx[u], node_to_idx[v], weight)
            
            # Run Infomap
            im.run()
            
            # Extract communities
            communities = {}
            for node_idx, module in im.modules:
                orig_node = idx_to_node[node_idx]
                communities[orig_node] = module
            
            return communities
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"Infomap on subgraph failed: {str(e)}")
            return self._detect_louvain_in_subgraph(subgraph, 0.8)
    
    def _detect_leiden_communities(self, resolution):
        """Detect communities using Leiden algorithm"""
        try:
            import leidenalg
            import igraph as ig
            
            self.logger.info(f"Running Leiden algorithm with resolution {resolution}")
            
            # Convert networkx graph to igraph
            g = ig.Graph()
            
            # Add vertices
            node_map = {node: i for i, node in enumerate(self.graph.nodes())}
            g.add_vertices(len(node_map))
            
            # Add edges
            edges = [(node_map[u], node_map[v]) for u, v in self.graph.edges()]
            g.add_edges(edges)
            
            # Add edge weights if available
            weights = [data.get("weight", 1.0) for _, _, data in self.graph.edges(data=True)]
            g.es["weight"] = weights
            
            # Run Leiden algorithm
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=42
            )
            
            # Clear existing assignments
            self.communities = {}
            self.community_to_nodes = defaultdict(list)
            
            # Reverse node mapping
            idx_to_node = {i: node for node, i in node_map.items()}
            
            # Extract communities
            for i, cluster in enumerate(partition):
                for node_idx in cluster:
                    orig_node = idx_to_node[node_idx]
                    self.communities[orig_node] = i
                    self.community_to_nodes[i].append(orig_node)
            
            self.logger.info(f"Leiden method found {len(self.community_to_nodes)} communities")
            
        except ImportError:
            self.logger.warning("leidenalg/igraph not available, falling back to Louvain")
            self._detect_louvain_communities(resolution)
        except Exception as e:
            self.logger.error(f"Error in Leiden algorithm: {str(e)}")
            self._detect_louvain_communities(resolution)
    
    def _detect_leiden_in_subgraph(self, subgraph, resolution):
        """Run Leiden algorithm on a subgraph"""
        try:
            import leidenalg
            import igraph as ig
            
            # Convert to igraph
            g = ig.Graph()
            
            # Add vertices and create mapping
            node_map = {node: i for i, node in enumerate(subgraph.nodes())}
            g.add_vertices(len(node_map))
            
            # Add edges
            edges = [(node_map[u], node_map[v]) for u, v in subgraph.edges()]
            g.add_edges(edges)
            
            # Add weights
            weights = [data.get("weight", 1.0) for _, _, data in subgraph.edges(data=True)]
            g.es["weight"] = weights
            
            # Run Leiden
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=42
            )
            
            # Reverse mapping
            idx_to_node = {i: node for node, i in node_map.items()}
            
            # Extract communities
            communities = {}
            for i, cluster in enumerate(partition):
                for node_idx in cluster:
                    orig_node = idx_to_node[node_idx]
                    communities[orig_node] = i
            
            return communities
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"Leiden on subgraph failed: {str(e)}")
            return self._detect_louvain_in_subgraph(subgraph, resolution)
    
    def _detect_label_propagation_communities(self):
        """Detect communities using Label Propagation"""
        self.logger.info("Running Label Propagation algorithm")
        
        try:
            # Use NetworkX's label propagation
            communities = nx.algorithms.community.label_propagation_communities(self.graph)
            
            # Clear existing assignments
            self.communities = {}
            self.community_to_nodes = defaultdict(list)
            
            # Convert to our format
            for i, comm in enumerate(communities):
                for node in comm:
                    self.communities[node] = i
                    self.community_to_nodes[i].append(node)
            
            self.logger.info(f"Label Propagation found {len(self.community_to_nodes)} communities")
            
        except Exception as e:
            self.logger.error(f"Error in Label Propagation: {str(e)}")
            self._detect_spectral_communities()
    
    def _detect_label_prop_in_subgraph(self, subgraph):
        """Run Label Propagation on a subgraph"""
        try:
            communities = nx.algorithms.community.label_propagation_communities(subgraph)
            
            community_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
            
            return community_map
            
        except Exception as e:
            self.logger.warning(f"Label Propagation on subgraph failed: {str(e)}")
            
            # Fallback: assign all to one community
            return {node: 0 for node in subgraph.nodes()}
    
    def _detect_spectral_communities(self):
        """Detect communities using Spectral Clustering"""
        self.logger.info("Running Spectral Clustering algorithm")
        
        try:
            # Determine number of clusters based on graph size
            n_clusters = min(100, max(3, int(np.sqrt(self.graph.number_of_nodes() / 5))))
            
            # Convert graph to adjacency matrix
            adj_matrix = nx.to_scipy_sparse_array(self.graph)
            
            # Create spectral clustering model
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                n_init=10,
                random_state=42
            )
            
            # Fit model
            node_list = list(self.graph.nodes())
            labels = model.fit_predict(adj_matrix.todense())
            
            # Clear existing assignments
            self.communities = {}
            self.community_to_nodes = defaultdict(list)
            
            # Convert to our format
            for i, node in enumerate(node_list):
                comm_id = labels[i]
                self.communities[node] = comm_id
                self.community_to_nodes[comm_id].append(node)
            
            self.logger.info(f"Spectral Clustering found {len(self.community_to_nodes)} communities")
            
        except Exception as e:
            self.logger.error(f"Error in Spectral Clustering: {str(e)}")
            
            # Fallback to connected components
            self._detect_connected_components()
    
    def _detect_spectral_in_subgraph(self, subgraph):
        """Run Spectral Clustering on a subgraph"""
        try:
            # Determine clusters based on subgraph size
            n_clusters = min(20, max(2, int(np.sqrt(subgraph.number_of_nodes() / 3))))
            
            # Convert to adjacency matrix
            adj_matrix = nx.to_scipy_sparse_array(subgraph)
            
            # Run spectral clustering
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                n_init=5,
                random_state=42
            )
            
            # Fit model
            node_list = list(subgraph.nodes())
            labels = model.fit_predict(adj_matrix.todense())
            
            # Convert to dictionary
            return {node_list[i]: labels[i] for i in range(len(node_list))}
            
        except Exception as e:
            self.logger.warning(f"Spectral Clustering on subgraph failed: {str(e)}")
            
            # Fallback to connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(subgraph)):
                for node in component:
                    communities[node] = i
            return communities
    
    def _detect_connected_components(self):
        """Fallback method using connected components"""
        self.logger.info("Falling back to connected components for community detection")
        
        # Clear existing assignments
        self.communities = {}
        self.community_to_nodes = defaultdict(list)
        
        # Assign connected components as communities
        for i, component in enumerate(nx.connected_components(self.graph)):
            for node in component:
                self.communities[node] = i
                self.community_to_nodes[i].append(node)
        
        self.logger.info(f"Connected Components found {len(self.community_to_nodes)} communities")
    
    def _force_community_merges(self):
        """Force merging of communities to reduce fragmentation"""
        self.logger.info("Forcing community merges to reduce fragmentation")
        
        # Calculate community similarities
        similarities = {}
        for comm1 in self.community_to_nodes:
            for comm2 in self.community_to_nodes:
                if comm1 >= comm2:
                    continue
                
                # Get nodes in each community
                nodes1 = set(self.community_to_nodes[comm1])
                nodes2 = set(self.community_to_nodes[comm2])
                
                # Count edges between communities
                edge_count = 0
                for u in nodes1:
                    for v in self.graph.neighbors(u):
                        if v in nodes2:
                            edge_count += 1
                
                # Store similarity if edges exist
                if edge_count > 0:
                    similarities[(comm1, comm2)] = edge_count
        
        # Sort similarities
        sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Determine number of merges needed
        target_communities = min(self.graph.number_of_nodes() // 20, 500)
        target_communities = max(target_communities, 10)  # At least 10 communities
        
        # Perform merges
        merges_needed = len(self.community_to_nodes) - target_communities
        merges_performed = 0
        
        if merges_needed <= 0:
            self.logger.info("No forced merges needed")
            return
        
        self.logger.info(f"Planning to perform {merges_needed} community merges")
        
        # Track which communities have been merged
        merged = set()
        merge_map = {}  # Maps old community IDs to new ones
        
        # Perform merges
        for (comm1, comm2), _ in sorted_pairs:
            # Skip if either community has been merged already
            if comm1 in merged or comm2 in merged:
                continue
            
            # Merge comm2 into comm1
            merge_map[comm2] = comm1
            merged.add(comm2)
            
            merges_performed += 1
            if merges_performed >= merges_needed:
                break
        
        # Apply merges
        if merge_map:
            # Update node assignments
            for node, comm_id in list(self.communities.items()):
                if comm_id in merge_map:
                    self.communities[node] = merge_map[comm_id]
            
            # Rebuild community_to_nodes
            self.community_to_nodes = defaultdict(list)
            for node, comm_id in self.communities.items():
                self.community_to_nodes[comm_id].append(node)
            
            self.logger.info(f"Performed {merges_performed} forced community merges")
    
    def _merge_small_communities(self, min_size):
        """Merge small communities into larger ones"""
        self.logger.info(f"Merging communities smaller than {min_size} nodes")
        
        # Identify small communities
        small_comms = [
            comm_id for comm_id, nodes in self.community_to_nodes.items()
            if len(nodes) < min_size
        ]
        
        if not small_comms:
            self.logger.info("No small communities to merge")
            return
        
        self.logger.info(f"Found {len(small_comms)} small communities to merge")
        
        # Process each small community
        for comm_id in small_comms:
            nodes = self.community_to_nodes[comm_id]
            
            # Find best community to merge with
            best_target = None
            max_connections = 0
            
            # Check connections to other communities
            for node in nodes:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in self.communities:
                        neighbor_comm = self.communities[neighbor]
                        
                        # Skip if same community or another small one
                        if neighbor_comm == comm_id or neighbor_comm in small_comms:
                            continue
                        
                        # Count this connection
                        connections = sum(1 for n in nodes if any(
                            self.communities.get(neigh) == neighbor_comm
                            for neigh in self.graph.neighbors(n)
                        ))
                        
                        if connections > max_connections:
                            max_connections = connections
                            best_target = neighbor_comm
            
            # If we found a good target, merge
            if best_target is not None:
                # Update node assignments
                for node in nodes:
                    self.communities[node] = best_target
                    self.community_to_nodes[best_target].append(node)
                
                # Remove old community
                del self.community_to_nodes[comm_id]
            else:
                # No good connections - keep as is
                self.logger.info(f"Small community {comm_id} with {len(nodes)} nodes has no good merge target")
        
        # Rebuild community_to_nodes to ensure consistency
        self.community_to_nodes = defaultdict(list)
        for node, comm_id in self.communities.items():
            self.community_to_nodes[comm_id].append(node)
        
        self.logger.info(f"After merging, {len(self.community_to_nodes)} communities remain")
    
    def _calculate_quality_metrics(self):
        """Calculate modularity and other quality metrics"""
        try:
            import community as community_louvain
            
            # Create partition dictionary
            partition = {node: comm_id for node, comm_id in self.communities.items()}
            
            # Calculate modularity
            self.modularity = community_louvain.modularity(partition, self.graph)
            
            # Verify modularity is reasonable
            if self.modularity < 0.05:
                self.logger.warning(f"Very low modularity detected: {self.modularity:.4f}")
                
        except ImportError:
            # Calculate approximate modularity
            self._calculate_approximate_modularity()
            
        # Calculate conductance for each community
        for comm_id, nodes in self.community_to_nodes.items():
            self.conductance[comm_id] = self._calculate_community_conductance(nodes)
    
    def _calculate_approximate_modularity(self):
        """Calculate approximate modularity without community-louvain package"""
        self.logger.info("Calculating approximate modularity")
        
        # Get total number of edges
        m = self.graph.number_of_edges()
        if m == 0:
            self.modularity = 0.0
            return
        
        # Initialize modularity
        q = 0.0
        
        # Calculate contribution of each community
        for comm_id, nodes in self.community_to_nodes.items():
            # Create subgraph for this community
            subgraph = self.graph.subgraph(nodes)
            
            # Count internal edges
            l_c = subgraph.number_of_edges()
            
            # Calculate total degree of nodes in community
            d_c = sum(self.graph.degree(n) for n in nodes)
            
            # Calculate expected number of edges
            expected = (d_c * d_c) / (4 * m)
            
            # Add contribution to modularity
            q += (l_c / m) - (expected / m)
        
        self.modularity = q
        self.logger.info(f"Approximate modularity: {self.modularity:.4f}")
    
    def _calculate_community_conductance(self, nodes):
        """Calculate conductance for a community"""
        nodes_set = set(nodes)
        
        # Count internal and external edges
        internal_edges = 0
        external_edges = 0
        
        for node in nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor in nodes_set:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Internal edges are counted twice
        internal_edges //= 2
        
        # Calculate conductance
        total_edges = internal_edges + external_edges
        if total_edges == 0:
            return 1.0  # No edges, worst conductance
        
        return external_edges / total_edges
    
    def _detect_hierarchical_structure(self, levels, min_size):
        """Detect hierarchical community structure with multiple resolution levels"""
        self.logger.info(f"Detecting hierarchical community structure with {levels} levels")
        
        self.dendrogram = {"levels": []}
        
        # Store current level as level 0
        current_level = {
            "level": 0,
            "resolution": 0.8,  # Default resolution used initially
            "communities": {
                comm_id: {
                    "size": len(nodes),
                    "parent": None,
                    "children": []
                }
                for comm_id, nodes in self.community_to_nodes.items()
            }
        }
        
        self.dendrogram["levels"].append(current_level)
        
        # Process higher levels with increasing resolution
        for level in range(1, levels):
            self.logger.info(f"Processing hierarchical level {level} with increased resolution")
            
            # Adjust resolution parameter - higher values create smaller communities
            resolution = 0.8 + level * 0.4
            
            # Process each community from previous level
            level_communities = {}
            
            # Get communities from previous level
            prev_level = self.dendrogram["levels"][-1]
            
            # Track subcommunity assignments
            subcommunities = {}
            subcommunity_to_nodes = defaultdict(list)
            next_subcommunity_id = 0
            
            # Process each community
            for parent_id, parent_data in prev_level["communities"].items():
                # Only subdivide sufficiently large communities
                if parent_data["size"] >= min_size * 2:
                    # Get nodes in this community
                    parent_nodes = self.community_to_nodes[parent_id]
                    
                    # Create subgraph
                    subgraph = self.graph.subgraph(parent_nodes).copy()
                    
                    # Detect communities in subgraph
                    try:
                        import community as community_louvain
                        
                        # Use higher resolution for finer-grained communities
                        partition = community_louvain.best_partition(
                            subgraph, 
                            resolution=resolution,
                            random_state=42
                        )
                        
                        # Remap community IDs to be globally unique
                        children_ids = []
                        
                        for node, local_comm_id in partition.items():
                            # Create global ID with parent prefix
                            global_comm_id = next_subcommunity_id + local_comm_id
                            
                            # Store mapping
                            subcommunities[node] = global_comm_id
                            subcommunity_to_nodes[global_comm_id].append(node)
                            
                            # Track child IDs for parent
                            if global_comm_id not in children_ids:
                                children_ids.append(global_comm_id)
                        
                        # Update parent's children
                        parent_data["children"] = children_ids
                        
                        # Add community data to this level
                        for child_id in children_ids:
                            level_communities[child_id] = {
                                "size": len(subcommunity_to_nodes[child_id]),
                                "parent": parent_id,
                                "children": []
                            }
                        
                        # Update next subcommunity ID
                        if partition:
                            next_subcommunity_id += max(partition.values()) + 1
                            
                    except Exception as e:
                        self.logger.warning(f"Error detecting subcommunities: {str(e)}")
                        # Keep as is
                        level_communities[parent_id] = {
                            "size": parent_data["size"],
                            "parent": parent_data["parent"],
                            "children": []
                        }
                else:
                    # Too small to subdivide, keep as is in this level
                    level_communities[parent_id] = {
                        "size": parent_data["size"],
                        "parent": parent_data["parent"],
                        "children": []
                    }
            
            # Add level to dendrogram
            self.dendrogram["levels"].append({
                "level": level,
                "resolution": resolution,
                "communities": level_communities
            })
            
            # Store subcommunities
            self.hierarchical_communities[level] = subcommunities
        
        self.logger.info(f"Completed hierarchical structure with {levels} levels")
    
    def _detect_overlapping_communities(self, threshold=0.3, max_communities=3):
        """Detect overlapping communities where nodes can belong to multiple groups"""
        self.logger.info(f"Detecting overlapping communities with threshold {threshold}")
        
        # Calculate node-community affinities
        node_affinities = defaultdict(lambda: defaultdict(float))
        
        # Initialize with primary communities
        for node, comm_id in self.communities.items():
            node_affinities[node][comm_id] = 1.0
        
        # Calculate affinities based on edges to other communities
        for u, v in self.graph.edges():
            if u not in self.communities or v not in self.communities:
                continue
                
            u_comm = self.communities[u]
            v_comm = self.communities[v]
            
            if u_comm == v_comm:
                continue
                
            # Get edge weight
            weight = self.graph.get_edge_data(u, v).get("weight", 1.0)
            
            # Update affinities
            node_affinities[u][v_comm] += weight
            node_affinities[v][u_comm] += weight
        
        # Normalize and assign overlapping communities
        for node, affinities in node_affinities.items():
            # Calculate total affinity
            total_affinity = sum(affinities.values())
            
            # Sort by affinity
            sorted_affinities = sorted(
                [(comm_id, aff/total_affinity) for comm_id, aff in affinities.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Assign to communities above threshold
            self.overlapping_communities[node] = [
                (comm_id, aff) for comm_id, aff in sorted_affinities
                if aff >= threshold
            ][:max_communities]
        
        self.logger.info(f"Assigned overlapping community memberships to {len(self.overlapping_communities)} nodes")
    
    def _detect_subcommunities(self, resolution=1.2):
        """Detect subcommunities within each main community"""
        self.logger.info(f"Detecting subcommunities with resolution {resolution}")
        
        self.subcommunities = {}
        
        try:
            import community as community_louvain
            
            for comm_id, nodes in self.community_to_nodes.items():
                # Skip small communities
                if len(nodes) < 10:
                    continue
                    
                # Create subgraph for this community
                subgraph = self.graph.subgraph(nodes).copy()
                
                # Run community detection on the subgraph
                try:
                    sub_partition = community_louvain.best_partition(
                        subgraph, 
                        resolution=resolution,
                        random_state=42
                    )
                    
                    # Renumber subcommunities to be unique
                    base_id = comm_id * 1000  # Use main community as prefix
                    
                    # Group nodes by subcommunity
                    sub_communities = defaultdict(list)
                    for node, sub_id in sub_partition.items():
                        global_sub_id = base_id + sub_id
                        sub_communities[global_sub_id].append(node)
                    
                    # Store subcommunities
                    self.subcommunities[comm_id] = dict(sub_communities)
                    
                    self.logger.info(f"Detected {len(sub_communities)} subcommunities in community {comm_id}")
                    
                except Exception as e:
                    self.logger.warning(f"Error detecting subcommunities for community {comm_id}: {str(e)}")
                
        except Exception as e:
            self.logger.warning(f"Error detecting subcommunities: {str(e)}")
    
    def _generate_community_profiles(self):
        """Generate profiles for each community"""
        self.logger.info("Generating community profiles")
        
        self.community_profiles = {}
        
        for comm_id, nodes in self.community_to_nodes.items():
            # Skip communities with too few nodes
            if len(nodes) < 3:
                continue
                
            # Count node types
            node_types = Counter()
            for node in nodes:
                if node in self.graph:
                    node_type = self.graph.nodes[node].get("node_type", "unknown")
                    node_types[node_type] += 1
            
            # Get main node type
            main_type = node_types.most_common(1)[0][0] if node_types else "unknown"
            
            # Calculate network metrics for this community
            subgraph = self.graph.subgraph(nodes)
            
            # Calculate density
            density = nx.density(subgraph)
            
            # Calculate centrality
            try:
                centrality = nx.degree_centrality(subgraph)
                betweenness = {k: v for k, v in sorted(
                    nx.betweenness_centrality(subgraph, k=min(10, len(nodes))).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]}
            except Exception:
                centrality = {node: 0.0 for node in subgraph.nodes()}
                betweenness = {node: 0.0 for node in subgraph.nodes()}
            
            # Find most central nodes
            key_nodes = [k for k, v in sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]]
            
            # Initialize profile
            profile = {
                "id": comm_id,
                "size": len(nodes),
                "main_type": main_type,
                "node_type_distribution": dict(node_types),
                "density": density,
                "conductance": self.conductance.get(comm_id, 0.0),
                "key_nodes": key_nodes,
                "betweenness_centrality": betweenness,
                "attributes": {},
                "temporal_data": {},
                "community_quality": {
                    "density": density,
                    "conductance": self.conductance.get(comm_id, 0.0),
                    "modularity_contribution": self._calculate_modularity_contribution(comm_id)
                }
            }
            
            # Extract community attributes based on main type
            if main_type == "movie":
                # For movie-dominated communities, extract genres and years
                self._extract_movie_community_attributes(nodes, profile)
            elif main_type == "user":
                # For user-dominated communities, extract demographics
                self._extract_user_community_attributes(nodes, profile)
            elif main_type == "actor":
                # For actor-dominated communities
                self._extract_actor_community_attributes(nodes, profile)
            elif main_type == "director":
                # For director-dominated communities
                self._extract_director_community_attributes(nodes, profile)
            elif main_type == "genre":
                # For genre-dominated communities
                self._extract_genre_community_attributes(nodes, profile)
            
            # Store profile
            self.community_profiles[comm_id] = profile
        
        self.logger.info(f"Generated profiles for {len(self.community_profiles)} communities")
    
    def _extract_movie_community_attributes(self, nodes, profile):
        """Extract attributes for movie-dominated communities"""
        genres = Counter()
        years = []
        ratings = []
        directors = Counter()
        actors = Counter()
        
        for node in nodes:
            if node in self.graph and self.graph.nodes[node].get("node_type") == "movie":
                # Extract genres
                node_genres = self.graph.nodes[node].get("genres", "[]")
                if isinstance(node_genres, str):
                    try:
                        genre_list = json.loads(node_genres)
                        for genre in genre_list:
                            genres[genre] += 1
                    except:
                        pass
                
                # Extract year
                year = self.graph.nodes[node].get("year")
                if year:
                    try:
                        years.append(int(year))
                    except:
                        pass
                
                # Extract average rating
                rating = self.graph.nodes[node].get("avg_rating")
                if rating:
                    try:
                        ratings.append(float(rating))
                    except:
                        pass
                
                # Extract directors and actors through edges
                for neighbor in self.graph.neighbors(node):
                    neighbor_type = self.graph.nodes[neighbor].get("node_type", "")
                    if neighbor_type == "director":
                        directors[self.graph.nodes[neighbor].get("name", neighbor)] += 1
                    elif neighbor_type == "actor":
                        actors[self.graph.nodes[neighbor].get("name", neighbor)] += 1
        
        # Add to profile
        profile["attributes"]["top_genres"] = [g for g, _ in genres.most_common(5)]
        profile["attributes"]["top_directors"] = [d for d, _ in directors.most_common(3)]
        profile["attributes"]["top_actors"] = [a for a, _ in actors.most_common(5)]
        
        if years:
            profile["attributes"]["avg_year"] = sum(years) / len(years)
            profile["attributes"]["year_range"] = [min(years), max(years)]
            
            # Add temporal distribution
            year_distribution = Counter(years)
            profile["temporal_data"]["year_distribution"] = dict(sorted(year_distribution.items()))
            
            # Calculate dominant era
            if len(years) >= 3:
                # Group into decades
                decades = [y // 10 * 10 for y in years]
                decade_counter = Counter(decades)
                dominant_decade = decade_counter.most_common(1)[0][0]
                profile["attributes"]["dominant_era"] = f"{dominant_decade}s"
        
        if ratings:
            profile["attributes"]["avg_rating"] = sum(ratings) / len(ratings)
            profile["attributes"]["rating_range"] = [min(ratings), max(ratings)]
    
    def _extract_user_community_attributes(self, nodes, profile):
        """Extract attributes for user-dominated communities"""
        states = Counter()
        jobs = Counter()
        age_groups = Counter()
        genders = Counter()
        favorite_genres = Counter()
        
        for node in nodes:
            if node in self.graph and self.graph.nodes[node].get("node_type") == "user":
                # Extract state
                state = self.graph.nodes[node].get("state")
                if state:
                    states[state] += 1
                
                # Extract job
                job = self.graph.nodes[node].get("job")
                if job:
                    jobs[job] += 1
                
                # Extract age
                age = self.graph.nodes[node].get("age")
                if age:
                    try:
                        age = int(age)
                        if age < 18:
                            age_group = "<18"
                        elif age < 25:
                            age_group = "18-24"
                        elif age < 35:
                            age_group = "25-34"
                        elif age < 45:
                            age_group = "35-44"
                        elif age < 55:
                            age_group = "45-54"
                        else:
                            age_group = "55+"
                        
                        age_groups[age_group] += 1
                    except:
                        pass
                
                # Extract gender
                gender = self.graph.nodes[node].get("gender")
                if gender:
                    genders[gender] += 1
                
                # Extract favorite genres through movie connections
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get("node_type") == "movie":
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        
                        # Check if positively rated
                        rating = edge_data.get("rating", 0)
                        if rating > 3:  # Consider positive ratings only
                            # Get movie genres
                            movie_genres = self.graph.nodes[neighbor].get("genres", "[]")
                            if isinstance(movie_genres, str):
                                try:
                                    genre_list = json.loads(movie_genres)
                                    for genre in genre_list:
                                        favorite_genres[genre] += 1
                                except:
                                    pass
        
        # Add to profile
        profile["attributes"]["top_states"] = [s for s, _ in states.most_common(5)]
        profile["attributes"]["top_jobs"] = [j for j, _ in jobs.most_common(5)]
        profile["attributes"]["age_distribution"] = dict(age_groups)
        profile["attributes"]["gender_distribution"] = dict(genders)
        profile["attributes"]["favorite_genres"] = [g for g, _ in favorite_genres.most_common(5)]
    
    def _extract_actor_community_attributes(self, nodes, profile):
        """Extract attributes for actor-dominated communities"""
        movie_decades = Counter()
        genres = Counter()
        co_actors = Counter()
        
        for node in nodes:
            if node in self.graph and self.graph.nodes[node].get("node_type") == "actor":
                # Look at movies this actor was in
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get("node_type") == "movie":
                        # Extract year/decade
                        year = self.graph.nodes[neighbor].get("year")
                        if year:
                            try:
                                decade = (int(year) // 10) * 10
                                movie_decades[f"{decade}s"] += 1
                            except:
                                pass
                        
                        # Extract genres
                        movie_genres = self.graph.nodes[neighbor].get("genres", "[]")
                        if isinstance(movie_genres, str):
                            try:
                                genre_list = json.loads(movie_genres)
                                for genre in genre_list:
                                    genres[genre] += 1
                            except:
                                pass
                    
                    # Track co-actors
                    elif self.graph.nodes[neighbor].get("node_type") == "movie":
                        # Find other actors in this movie
                        movie = neighbor
                        for actor in self.graph.neighbors(movie):
                            if (actor != node and 
                                actor in self.graph and 
                                self.graph.nodes[actor].get("node_type") == "actor"):
                                co_actors[self.graph.nodes[actor].get("name", actor)] += 1
        
        # Add to profile
        profile["attributes"]["top_genres"] = [g for g, _ in genres.most_common(5)]
        profile["attributes"]["era_distribution"] = dict(movie_decades)
        profile["attributes"]["common_co_actors"] = [a for a, _ in co_actors.most_common(5)]
        
        if movie_decades:
            profile["attributes"]["dominant_era"] = movie_decades.most_common(1)[0][0]
    
    def _extract_director_community_attributes(self, nodes, profile):
        """Extract attributes for director-dominated communities"""
        movie_decades = Counter()
        genres = Counter()
        actors = Counter()
        
        for node in nodes:
            if node in self.graph and self.graph.nodes[node].get("node_type") == "director":
                # Look at movies this director made
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get("node_type") == "movie":
                        # Extract year/decade
                        year = self.graph.nodes[neighbor].get("year")
                        if year:
                            try:
                                decade = (int(year) // 10) * 10
                                movie_decades[f"{decade}s"] += 1
                            except:
                                pass
                        
                        # Extract genres
                        movie_genres = self.graph.nodes[neighbor].get("genres", "[]")
                        if isinstance(movie_genres, str):
                            try:
                                genre_list = json.loads(movie_genres)
                                for genre in genre_list:
                                    genres[genre] += 1
                            except:
                                pass
                        
                        # Find actors in these movies
                        movie = neighbor
                        for actor in self.graph.neighbors(movie):
                            if (actor in self.graph and 
                                self.graph.nodes[actor].get("node_type") == "actor"):
                                actors[self.graph.nodes[actor].get("name", actor)] += 1
        
        # Add to profile
        profile["attributes"]["top_genres"] = [g for g, _ in genres.most_common(5)]
        profile["attributes"]["era_distribution"] = dict(movie_decades)
        profile["attributes"]["frequent_collaborators"] = [a for a, _ in actors.most_common(5)]
        
        if movie_decades:
            profile["attributes"]["dominant_era"] = movie_decades.most_common(1)[0][0]
    
    def _extract_genre_community_attributes(self, nodes, profile):
        """Extract attributes for genre-dominated communities"""
        related_genres = Counter()
        top_movies = Counter()
        top_directors = Counter()
        top_actors = Counter()
        decades = Counter()
        
        for node in nodes:
            if node in self.graph and self.graph.nodes[node].get("node_type") == "genre":
                # Look at movies in this genre
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get("node_type") == "movie":
                        movie = neighbor
                        movie_title = self.graph.nodes[movie].get("title", movie)
                        top_movies[movie_title] += 1
                        
                        # Extract year/decade
                        year = self.graph.nodes[movie].get("year")
                        if year:
                            try:
                                decade = (int(year) // 10) * 10
                                decades[f"{decade}s"] += 1
                            except:
                                pass
                        
                        # Find other genres of this movie
                        movie_genres = self.graph.nodes[movie].get("genres", "[]")
                        if isinstance(movie_genres, str):
                            try:
                                genre_list = json.loads(movie_genres)
                                for genre in genre_list:
                                    if genre != self.graph.nodes[node].get("name"):
                                        related_genres[genre] += 1
                            except:
                                pass
                        
                        # Find directors and actors of this movie
                        for person in self.graph.neighbors(movie):
                            person_type = self.graph.nodes[person].get("node_type", "")
                            person_name = self.graph.nodes[person].get("name", person)
                            
                            if person_type == "director":
                                top_directors[person_name] += 1
                            elif person_type == "actor":
                                top_actors[person_name] += 1
        
        # Add to profile
        profile["attributes"]["related_genres"] = [g for g, _ in related_genres.most_common(5)]
        profile["attributes"]["top_movies"] = [m for m, _ in top_movies.most_common(10)]
        profile["attributes"]["top_directors"] = [d for d, _ in top_directors.most_common(5)]
        profile["attributes"]["top_actors"] = [a for a, _ in top_actors.most_common(10)]
        profile["attributes"]["decade_distribution"] = dict(decades)
        
        if decades:
            profile["attributes"]["dominant_era"] = decades.most_common(1)[0][0]
    
    def _calculate_modularity_contribution(self, comm_id):
        """Calculate the contribution of a community to the overall modularity"""
        nodes = self.community_to_nodes[comm_id]
        
        # Create subgraph for this community
        subgraph = self.graph.subgraph(nodes)
        
        # Count internal edges
        internal_edges = subgraph.number_of_edges()
        
        # Count total degree of nodes in community
        total_degree = sum(self.graph.degree(n) for n in nodes)
        
        # Get total number of edges in graph
        m = self.graph.number_of_edges()
        if m == 0:
            return 0.0
        
        # Calculate modularity contribution
        expected = (total_degree * total_degree) / (4 * m)
        contribution = (internal_edges / m) - (expected / m)
        
        return contribution
    
    def get_community_for_node(self, node_id):
        """Get community ID for a node"""
        return self.communities.get(node_id)
    
    def get_overlapping_communities_for_node(self, node_id):
        """Get all community IDs for a node (overlapping)"""
        if node_id in self.overlapping_communities:
            return self.overlapping_communities[node_id]
        return [(self.communities.get(node_id), 1.0)] if node_id in self.communities else []
    
    def get_community_members(self, comm_id):
        """Get all nodes in a community"""
        return self.community_to_nodes.get(comm_id, [])
    
    def get_community_profile(self, comm_id):
        """Get profile for a community"""
        return self.community_profiles.get(comm_id, {})
    
    def get_node_community_profile(self, node_id):
        """Get community profile for a node"""
        comm_id = self.get_community_for_node(node_id)
        if comm_id is not None:
            return self.get_community_profile(comm_id)
        return {}
    
    def get_subcommunities_for_node(self, node_id):
        """Get subcommunity information for a node"""
        comm_id = self.get_community_for_node(node_id)
        if comm_id is None or comm_id not in self.subcommunities:
            return None
            
        # Find which subcommunity contains this node
        for sub_id, nodes in self.subcommunities[comm_id].items():
            if node_id in nodes:
                # Get parent community info
                parent_profile = self.get_community_profile(comm_id)
                
                # Calculate subcommunity info
                subcommunity_size = len(nodes)
                parent_size = parent_profile.get("size", 0)
                subcommunity_percent = (subcommunity_size / parent_size * 100) if parent_size else 0
                
                return {
                    "community_id": comm_id,
                    "subcommunity_id": sub_id,
                    "subcommunity_size": subcommunity_size,
                    "parent_size": parent_size,
                    "percent_of_parent": subcommunity_percent
                }
        
        return None
    
    def get_hierarchical_communities_for_node(self, node_id, max_levels=3):
        """Get hierarchical community path for a node"""
        # Get base community
        base_comm_id = self.get_community_for_node(node_id)
        if base_comm_id is None:
            return []
        
        hierarchy = [(0, base_comm_id)]
        
        # Add higher level communities if available
        for level in range(1, max_levels):
            if level in self.hierarchical_communities and node_id in self.hierarchical_communities[level]:
                hierarchy.append((level, self.hierarchical_communities[level][node_id]))
        
        return hierarchy
    
    def get_community_description(self, comm_id):
        """Generate natural language description of a community"""
        if comm_id not in self.community_profiles:
            return "Unknown community"
            
        profile = self.community_profiles[comm_id]
        
        # Start with basic info
        description = [f"Community {comm_id} has {profile['size']} members, primarily {profile['main_type']}s."]
        
        # Add type-specific information
        main_type = profile["main_type"]
        attrs = profile.get("attributes", {})
        
        if main_type == "movie":
            # Movie community description
            if "top_genres" in attrs and attrs["top_genres"]:
                genres = attrs["top_genres"]
                description.append(f"Top genres: {', '.join(genres[:3])}.")
            
            if "year_range" in attrs:
                year_range = attrs["year_range"]
                description.append(f"Movies from {year_range[0]} to {year_range[1]}.")
                
            if "dominant_era" in attrs:
                description.append(f"This community is dominated by {attrs['dominant_era']} movies.")
                
            if "top_directors" in attrs and attrs["top_directors"]:
                directors = attrs["top_directors"]
                description.append(f"Notable directors include: {', '.join(directors[:2])}.")
        
        elif main_type == "user":
            # User community description
            if "age_distribution" in attrs and attrs["age_distribution"]:
                description.append("Age groups: " + 
                               ", ".join(f"{k} ({v})" for k, v in attrs["age_distribution"].items()
                                        if v > 0)[:100] + ".")
            
            if "top_states" in attrs and attrs["top_states"]:
                states = attrs["top_states"]
                description.append(f"Members primarily from: {', '.join(states[:3])}.")
            
            if "favorite_genres" in attrs and attrs["favorite_genres"]:
                genres = attrs["favorite_genres"]
                description.append(f"Favorite genres: {', '.join(genres[:3])}.")
            
            if "top_jobs" in attrs and attrs["top_jobs"]:
                jobs = attrs["top_jobs"]
                description.append(f"Common occupations: {', '.join(jobs[:3])}.")
        
        elif main_type == "actor":
            if "top_genres" in attrs and attrs["top_genres"]:
                genres = attrs["top_genres"]
                description.append(f"These actors primarily appear in: {', '.join(genres[:3])}.")
                
            if "dominant_era" in attrs:
                description.append(f"Most active during the {attrs['dominant_era']}.")
                
            if "common_co_actors" in attrs and attrs["common_co_actors"]:
                co_actors = attrs["common_co_actors"]
                description.append(f"Frequently work with: {', '.join(co_actors[:3])}.")
                
        elif main_type == "director":
            if "top_genres" in attrs and attrs["top_genres"]:
                genres = attrs["top_genres"]
                description.append(f"These directors primarily work in: {', '.join(genres[:3])}.")
                
            if "dominant_era" in attrs:
                description.append(f"Most active during the {attrs['dominant_era']}.")
                
            if "frequent_collaborators" in attrs and attrs["frequent_collaborators"]:
                collaborators = attrs["frequent_collaborators"]
                description.append(f"Frequently cast: {', '.join(collaborators[:3])}.")
        
        elif main_type == "genre":
            if "related_genres" in attrs and attrs["related_genres"]:
                related = attrs["related_genres"]
                description.append(f"Often combined with: {', '.join(related[:3])}.")
                
            if "top_directors" in attrs and attrs["top_directors"]:
                directors = attrs["top_directors"]
                description.append(f"Notable directors: {', '.join(directors[:3])}.")
        
        # Add key nodes
        key_nodes = profile.get("key_nodes", [])
        if key_nodes:
            node_names = []
            for node in key_nodes[:3]:  # Limit to top 3
                if node in self.graph:
                    name = self.graph.nodes[node].get("name", 
                                                    self.graph.nodes[node].get("title", node))
                    node_names.append(name)
            
            if node_names:
                description.append(f"Key members include: {', '.join(node_names)}.")
                
        # Add community quality
        quality = profile.get("community_quality", {})
        if quality.get("modularity_contribution", 0) > 0.01:
            description.append(f"This is a well-defined community with strong internal connections.")
        
        return " ".join(description)
    
    def visualize_communities(self, output_path, max_nodes=1000, node_size_factor=10, 
                            show_labels=True, layout="spring"):
        """
        Visualize community structure
        
        Args:
            output_path: Path to save visualization
            max_nodes: Maximum number of nodes to display
            node_size_factor: Factor for node size calculation
            show_labels: Whether to show node labels
            layout: Layout algorithm ('spring', 'kamada_kawai', 'spectral')
        
        Returns:
            Path to saved visualization
        """
        self.logger.info(f"Visualizing community structure with {len(self.community_to_nodes)} communities")
        
        # Create subgraph if needed
        if self.graph.number_of_nodes() > max_nodes:
            # Sample nodes from each community
            sampled_nodes = []
            
            # Determine how many nodes to sample from each community
            total_communities = len(self.community_to_nodes)
            if total_communities == 0:
                return None
                
            nodes_per_community = max(3, min(50, max_nodes // total_communities))
            
            for comm_id, nodes in self.community_to_nodes.items():
                # Sample nodes from this community
                if len(nodes) <= nodes_per_community:
                    sampled_nodes.extend(nodes)
                else:
                    # Sample key nodes first
                    profile = self.community_profiles.get(comm_id, {})
                    key_nodes = profile.get("key_nodes", [])
                    
                    # Add key nodes
                    key_count = min(len(key_nodes), nodes_per_community // 2)
                    sampled_nodes.extend(key_nodes[:key_count])
                    
                    # Add random nodes
                    remaining = nodes_per_community - key_count
                    non_key_nodes = [n for n in nodes if n not in key_nodes]
                    if non_key_nodes and remaining > 0:
                        sampled_nodes.extend(random.sample(non_key_nodes, min(remaining, len(non_key_nodes))))
            
            # Create subgraph
            subgraph = self.graph.subgraph(sampled_nodes).copy()
        else:
            subgraph = self.graph
        
        # Set up colors for communities
        community_colors = {}
        for i, comm_id in enumerate(self.community_to_nodes.keys()):
            # Use colormap to get evenly distributed colors
            community_colors[comm_id] = i
        
        # Get node colors
        node_colors = []
        for node in subgraph.nodes():
            comm_id = self.communities.get(node, -1)
            color_idx = community_colors.get(comm_id, 0)
            node_colors.append(color_idx)
        
        # Get node sizes based on degree centrality
        centrality = nx.degree_centrality(subgraph)
        node_sizes = [node_size_factor + 100 * centrality[node] for node in subgraph.nodes()]
        
        # Create figure
        plt.figure(figsize=(20, 20))
        
        # Choose layout
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout == "spectral":
            pos = nx.spectral_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph, k=0.3, iterations=50)
        
        # Draw graph
        nx.draw_networkx(
            subgraph,
            pos=pos,
            node_color=node_colors,
            cmap=plt.cm.tab20,
            node_size=node_sizes,
            alpha=0.8,
            with_labels=False,
            edge_color="gray",
            width=0.5
        )
        
        # Add labels for important nodes
        if show_labels:
            # Label nodes with high centrality
            labels = {}
            for node in subgraph.nodes():
                if centrality[node] > 0.05:  # Label only important nodes
                    labels[node] = subgraph.nodes[node].get("name", 
                                                      subgraph.nodes[node].get("title", node))
            
            nx.draw_networkx_labels(
                subgraph,
                pos,
                labels=labels,
                font_size=10,
                font_color="black",
                font_weight="bold"
            )
        
        # Add community legend
        if len(community_colors) <= 20:  # Only show legend if not too many communities
            # Create legend elements
            legend_elements = []
            
            # Sort communities by size
            sorted_communities = sorted(
                [(comm_id, len(self.community_to_nodes[comm_id])) 
                 for comm_id in community_colors.keys()],
                key=lambda x: x[1],
                reverse=True
            )
            
            cmap = plt.cm.tab20
            for comm_id, size in sorted_communities[:10]:  # Top 10 communities
                color_idx = community_colors[comm_id]
                color = cmap(color_idx % 20)
                
                # Get community name
                profile = self.community_profiles.get(comm_id, {})
                main_type = profile.get("main_type", "unknown")
                
                # Create legend entry
                from matplotlib.lines import Line2D
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=10, label=f"Community {comm_id} ({size} {main_type}s)")
                )
            
            plt.legend(handles=legend_elements, loc='best', title="Top Communities")
        
        # Add title
        plt.title(f"Community Structure with {len(self.community_to_nodes)} Communities")
        plt.axis('off')
        
        # Save visualization
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Community visualization saved to {output_path}")
        return output_path
    
    def print_community_statistics(self):
        """Print detailed statistics about communities"""
        if not self.community_to_nodes:
            print("No communities detected.")
            return
        
        print(f"\n===== Community Detection Statistics =====")
        print(f"Total communities: {len(self.community_to_nodes)}")
        print(f"Overall modularity: {self.modularity:.4f}")
        
        # Size statistics
        sizes = [len(nodes) for nodes in self.community_to_nodes.values()]
        print(f"\nCommunity size statistics:")
        print(f"  Minimum size: {min(sizes)}")
        print(f"  Maximum size: {max(sizes)}")
        print(f"  Average size: {sum(sizes)/len(sizes):.1f}")
        
        # Size distribution
        size_dist = Counter()
        for size in sizes:
            if size < 10:
                size_dist["1-9"] += 1
            elif size < 50:
                size_dist["10-49"] += 1
            elif size < 100:
                size_dist["50-99"] += 1
            elif size < 500:
                size_dist["100-499"] += 1
            else:
                size_dist["500+"] += 1
        
        print("\nSize distribution:")
        for category, count in sorted(size_dist.items()):
            print(f"  {category}: {count} communities ({count/len(sizes)*100:.1f}%)")
        
        # Node type distribution
        main_types = Counter(profile.get("main_type", "unknown") 
                         for profile in self.community_profiles.values())
        
        print("\nCommunity main types:")
        for type_name, count in main_types.most_common():
            print(f"  {type_name}: {count} communities ({count/len(self.community_profiles)*100:.1f}%)")
        
        # Quality metrics
        print("\nQuality metrics:")
        conductances = [c for c in self.conductance.values()]
        if conductances:
            print(f"  Average conductance: {sum(conductances)/len(conductances):.4f}")
            print(f"  Best conductance: {min(conductances):.4f}")
        
        # Overlapping communities
        if self.overlapping_communities:
            memberships = [len(comms) for comms in self.overlapping_communities.values()]
            avg_membership = sum(memberships) / len(memberships)
            print(f"\nOverlapping communities:")
            print(f"  Nodes with multiple memberships: {len([m for m in memberships if m > 1])}")
            print(f"  Average memberships per node: {avg_membership:.2f}")
        
        # Hierarchical structure
        if self.dendrogram and self.dendrogram.get("levels"):
            print(f"\nHierarchical structure:")
            for level in self.dendrogram["levels"]:
                level_id = level["level"]
                comm_count = len(level["communities"])
                print(f"  Level {level_id}: {comm_count} communities (resolution: {level['resolution']:.2f})")
        
        # Top communities
        print("\nTop 5 communities by size:")
        top_communities = sorted(
            [(comm_id, len(nodes)) for comm_id, nodes in self.community_to_nodes.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for i, (comm_id, size) in enumerate(top_communities):
            profile = self.community_profiles.get(comm_id, {})
            main_type = profile.get("main_type", "unknown")
            print(f"  {i+1}. Community {comm_id}: {size} nodes, mainly {main_type}s")
            
            # Add brief description
            attrs = profile.get("attributes", {})
            if main_type == "movie" and "top_genres" in attrs:
                print(f"     Top genres: {', '.join(attrs['top_genres'][:3])}")
            elif main_type == "user" and "favorite_genres" in attrs:
                print(f"     Favorite genres: {', '.join(attrs['favorite_genres'][:3])}")

#############################################################
#            Quantum-Enhanced Recommendation Engine          #
#############################################################

class QuantumEnhancedRecommender:
    """
    Advanced recommendation engine with quantum-enhanced reasoning
    """
    
    def __init__(self, graph, embeddings=None, model=None):
        """
        Initialize recommendation engine
        
        Args:
            graph: NetworkX graph
            embeddings: Node embeddings dictionary
            model: PyTorch model (optional)
        """
        self.graph = graph
        self.embeddings = embeddings
        self.model = model
        self.logger = logging.getLogger("quantum_kg.recommender")
        
        # Initialize components
        self.meta_path_reasoner = AdvancedMetaPathReasoner(graph)
        self.community_detector = EnhancedCommunityDetector(graph)
        
        # Extract meta-path schemas
        self.meta_path_reasoner.extract_meta_path_schemas()
        
        # Ensure communities are detected
        
        self.community_detector.detect_communities(
            method="louvian",        # More reliable method
            resolution=0.7,          # Lower value for larger communities
            min_size=5,              # Minimum community size
            hierarchical=True,       # Enable hierarchical detection
            overlapping=True,        # Enable overlapping detection
            levels=3                 # Number of hierarchical levels
        )
        
        # Create index for movie nodes and user nodes
        self.movie_nodes = []
        self.user_nodes = []
        
        for node, data in graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            if node_type == "movie":
                self.movie_nodes.append(node)
            elif node_type == "user":
                self.user_nodes.append(node)
        
        self.logger.info(f"Quantum-Enhanced Recommender initialized with {len(self.movie_nodes)} movies and {len(self.user_nodes)} users")
    
    def recommend(self, user_id, n=10, diversity=0.5, 
                 novelty=0.5, include_watched=False, 
                 context=None, explanation_detail="high"):
        """
        Generate personalized recommendations
        
        Args:
            user_id: User node ID
            n: Number of recommendations
            diversity: Diversity parameter [0, 1]
            novelty: Novelty parameter [0, 1]
            include_watched: Whether to include previously watched items
            context: Conversation context dictionary
            explanation_detail: Level of detail in explanations
            
        Returns:
            List of recommendation objects with explanations
        """
        # First check if user exists
        if user_id not in self.graph:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        # Get user embedding if available
        user_embedding = None
        if self.embeddings and user_id in self.embeddings:
            user_embedding = self.embeddings[user_id]
        
        # Get user's community
        user_community = self.community_detector.get_community_for_node(user_id)
        
        # Get already watched movies
        watched_movies = set()
        rated_movies = {}
        
        for neighbor in self.graph.neighbors(user_id):
            if neighbor in self.movie_nodes:
                # Check edge type and rating
                edge_data = self.graph.get_edge_data(user_id, neighbor)
                edge_type = edge_data.get("edge_type", "").lower()
                
                if edge_type == "rated" or edge_type == "watched":
                    watched_movies.add(neighbor)
                    
                    # Get rating if available
                    rating = edge_data.get("rating", 0)
                    rated_movies[neighbor] = rating
        
        # Add context preferences if available
        context_preferences = {}
        if context:
            # Extract mentioned entities from context
            mentioned_genres = context.get("mentioned_genres", [])
            mentioned_actors = context.get("mentioned_actors", [])
            mentioned_directors = context.get("mentioned_directors", [])
            mentioned_movies = context.get("mentioned_movies", [])
            
            # Store preferences
            context_preferences = {
                "genres": mentioned_genres,
                "actors": mentioned_actors,
                "directors": mentioned_directors,
                "movies": mentioned_movies,
                "explicit_preferences": context.get("preferences", {}),
                "constraints": context.get("constraints", {})
            }
        
        # Get candidate movies
        candidates = []
        
        # Include all unwatched movies as candidates
        for movie in self.movie_nodes:
            if include_watched or movie not in watched_movies:
                candidates.append(movie)
        
        if not candidates:
            self.logger.warning(f"No candidate movies for user {user_id}")
            return []
        
        # Score candidates
        scored_candidates = []
        
        for movie in candidates:
            # Skip if movie doesn't exist (should not happen)
            if movie not in self.graph:
                continue
                
            # Compute recommendation score components
            
            # 1. Meta-path based score
            paths = self.meta_path_reasoner.find_paths(user_id, movie, max_length=4)
            meta_path_score = 0.0
            
            if paths:
                # Use best path score
                meta_path_score = paths[0]["score"]
            
            # 2. Embedding similarity score (if available)
            embedding_score = 0.0
            
            if user_embedding is not None and movie in self.embeddings:
                movie_embedding = self.embeddings[movie]
                
                similarity = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    movie_embedding.reshape(1, -1)
                )[0][0]
                
                embedding_score = max(0, similarity)
            
            # 3. Community-based score
            community_score = 0.0
            
            if user_community is not None:
                # Get all possible communities for this movie (if overlapping detection is used)
                movie_communities = []
                
                # Check for overlapping communities first
                if hasattr(self.community_detector, 'get_overlapping_communities_for_node'):
                    movie_comms = self.community_detector.get_overlapping_communities_for_node(movie)
                    movie_communities = [comm_id for comm_id, _ in movie_comms] if movie_comms else []
                
                # Fallback to primary community
                if not movie_communities:
                    movie_community = self.community_detector.get_community_for_node(movie)
                    if movie_community is not None:
                        movie_communities = [movie_community]
                
                # Calculate community score based on all possible communities
                if movie_communities:
                    # Check if any movie community matches the user's community
                    if user_community in movie_communities:
                        # Same community gets high score
                        community_score = 0.8
                    else:
                        # Compare community profiles for best matching community
                        max_similarity = 0.0
                        
                        for movie_community in movie_communities:
                            user_profile = self.community_detector.get_community_profile(user_community)
                            movie_profile = self.community_detector.get_community_profile(movie_community)
                            
                            # Calculate similarity between profiles
                            similarity = self._calculate_community_profile_similarity(user_profile, movie_profile)
                            max_similarity = max(max_similarity, similarity)
                        
                        community_score = 0.4 * max_similarity
            
            # 4. Neural model score (if available)
            model_score = 0.0
            
            if self.model is not None and hasattr(self.model, 'predict_rating'):
                try:
                    # Get all node features
                    node_features = []
                    for node in self.graph.nodes():
                        if node in self.embeddings:
                            node_features.append(self.embeddings[node])
                        else:
                            # Use placeholder
                            dim = next(iter(self.embeddings.values())).shape[0]
                            node_features.append(np.zeros(dim))
                    
                    # Convert to tensor
                    x = torch.tensor(node_features, dtype=torch.float)
                    
                    # Get node indices
                    user_idx = list(self.graph.nodes()).index(user_id)
                    movie_idx = list(self.graph.nodes()).index(movie)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        embeddings = self.model(x)
                        
                        # Predict rating
                        rating = self.model.predict_rating(
                            embeddings,
                            torch.tensor([user_idx]),
                            torch.tensor([movie_idx])
                        ).item()
                        
                        # Convert to [0, 1] range
                        model_score = (rating + 1) / 2
                
                except Exception as e:
                    self.logger.warning(f"Error using neural model: {str(e)}")
            
            # 5. Context preference score
            context_score = 0.0
            
            if context_preferences:
                # Calculate genre match
                if "genres" in context_preferences and context_preferences["genres"]:
                    movie_data = self.graph.nodes[movie]
                    movie_genres = movie_data.get("genres", "[]")
                    
                    if isinstance(movie_genres, str):
                        try:
                            movie_genre_list = json.loads(movie_genres)
                            
                            # Count matches
                            matches = sum(1 for g in movie_genre_list if g in context_preferences["genres"])
                            
                            if matches > 0:
                                context_score += 0.5 * (matches / len(context_preferences["genres"]))
                        except:
                            pass
                
                # Calculate actor/director match from meta-paths
                actor_match = False
                director_match = False
                
                for path in paths:
                    for i, node in enumerate(path["path"]):
                        if node in self.graph:
                            node_type = self.graph.nodes[node].get("node_type", "")
                            node_name = self.graph.nodes[node].get("name", "")
                            
                            if node_type == "actor" and node_name in context_preferences.get("actors", []):
                                actor_match = True
                            elif node_type == "director" and node_name in context_preferences.get("directors", []):
                                director_match = True
                
                if actor_match:
                    context_score += 0.3
                
                if director_match:
                    context_score += 0.3
            
            # 6. Novelty score - for items different from what user has seen
            novelty_score = 0.0
            
            if rated_movies and novelty > 0.1:
                # Calculate average embedding of positively rated movies
                liked_embeddings = []
                
                for movie_id, rating in rated_movies.items():
                    if rating > 0 and movie_id in self.embeddings:
                        liked_embeddings.append(self.embeddings[movie_id])
                
                if liked_embeddings and movie in self.embeddings:
                    # Calculate average liked embedding
                    avg_liked = np.mean(liked_embeddings, axis=0)
                    
                    # Calculate distance from average (higher = more novel)
                    movie_embedding = self.embeddings[movie]
                    distance = 1.0 - cosine_similarity(
                        avg_liked.reshape(1, -1),
                        movie_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # Normalize to [0, 1]
                    novelty_score = max(0, min(1, distance))
            
            # Combine scores with weights
            # Adjust weights based on diversity and novelty parameters
            path_weight = 0.25 * (1 - diversity)
            embedding_weight = 0.25 * (1 - diversity) 
            community_weight = 0.15 * (1 - diversity)
            model_weight = 0.15
            context_weight = 0.1 + (0.1 * diversity)
            novelty_weight = novelty * 0.2
            
            # Normalize weights
            total_weight = path_weight + embedding_weight + community_weight + model_weight + context_weight + novelty_weight
            path_weight /= total_weight
            embedding_weight /= total_weight
            community_weight /= total_weight
            model_weight /= total_weight
            context_weight /= total_weight
            novelty_weight /= total_weight
            
            # Compute final score
            final_score = (
                path_weight * meta_path_score +
                embedding_weight * embedding_score +
                community_weight * community_score +
                model_weight * model_score +
                context_weight * context_score +
                novelty_weight * novelty_score
            )
            
            # Store candidate with scores
            scored_candidates.append({
                'movie': movie,
                'final_score': final_score,
                'meta_path_score': meta_path_score,
                'embedding_score': embedding_score,
                'community_score': community_score,
                'model_score': model_score,
                'context_score': context_score,
                'novelty_score': novelty_score,
                'paths': paths
            })
        
        # Sort candidates by final score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity filtering if requested
        if diversity > 0.1:
            # Get top-2*n candidates as initial pool
            pool = scored_candidates[:2*n]
            
            # Rerank with diversity
            reranked = []
            
            while len(reranked) < n and pool:
                # Pick highest scored item first
                if not reranked:
                    reranked.append(pool.pop(0))
                    continue
                
                # For remaining items, balance score and diversity
                max_diversity_score = 0
                max_idx = 0
                
                for i, candidate in enumerate(pool):
                    # Calculate diversity from already selected items
                    diversity_score = 0
                    
                    for selected in reranked:
                        # Check path-based diversity
                        path_diversity = 1.0
                        
                        # Check for common entities in paths
                        candidate_paths = set()
                        selected_paths = set()
                        
                        for path in candidate["paths"]:
                            for node in path["path"]:
                                if node != user_id and node != candidate["movie"]:
                                    candidate_paths.add(node)
                        
                        for path in selected["paths"]:
                            for node in path["path"]:
                                if node != user_id and node != selected["movie"]:
                                    selected_paths.add(node)
                        
                        # Calculate Jaccard similarity of path entities
                        if candidate_paths and selected_paths:
                            overlap = len(candidate_paths & selected_paths)
                            union = len(candidate_paths | selected_paths)
                            path_diversity = 1.0 - (overlap / union)
                        
                        # Calculate embedding diversity
                        embedding_diversity = 1.0
                        if (selected["movie"] in self.embeddings and 
                            candidate["movie"] in self.embeddings):
                            # Use embedding distance as diversity measure
                            similarity = cosine_similarity(
                                self.embeddings[selected["movie"]].reshape(1, -1),
                                self.embeddings[candidate["movie"]].reshape(1, -1)
                            )[0][0]
                            
                            # Convert to diversity (1 - similarity)
                            embedding_diversity = 1.0 - similarity
                        
                        # Calculate community diversity
                        community_diversity = 1.0
                        selected_community = self.community_detector.get_community_for_node(selected["movie"])
                        candidate_community = self.community_detector.get_community_for_node(candidate["movie"])
                        
                        if selected_community is not None and candidate_community is not None:
                            if selected_community == candidate_community:
                                community_diversity = 0.2
                            else:
                                # Different communities are diverse
                                community_diversity = 1.0
                        
                        # Combine diversity metrics
                        item_diversity = (0.4 * path_diversity + 
                                        0.4 * embedding_diversity + 
                                        0.2 * community_diversity)
                        
                        diversity_score += item_diversity
                    
                    # Average diversity
                    diversity_score /= len(reranked)
                    
                    # Combine diversity and score
                    combined_score = (1 - diversity) * candidate['final_score'] + diversity * diversity_score
                    
                    if combined_score > max_diversity_score:
                        max_diversity_score = combined_score
                        max_idx = i
                
                # Add most diverse candidate
                reranked.append(pool.pop(max_idx))
            
            scored_candidates = reranked
        
        # Limit to top-n
        top_n = scored_candidates[:n]
        
        # Generate detailed recommendations with explanations
        recommendations = []
        
        for i, candidate in enumerate(top_n):
            movie = candidate['movie']
            movie_data = self.graph.nodes[movie]
            
            # Extract basic movie information
            title = movie_data.get("title", movie)
            year = movie_data.get("year", "")
            
            # Get genres
            genres = []
            genres_str = movie_data.get("genres", "[]")
            if isinstance(genres_str, str):
                try:
                    genres = json.loads(genres_str)
                except:
                    pass
            
            # Generate explanation based on best path
            explanation = "No explanation available"
            if candidate['paths']:
                explanation = self.meta_path_reasoner.generate_explanation(
                    candidate['paths'][0],
                    personalize=True
                )
            
            # Get community information
            community_info = None
            movie_community = self.community_detector.get_community_for_node(movie)
            if movie_community is not None:
                community_info = {
                    "id": movie_community,
                    "description": self.community_detector.get_community_description(movie_community)
                }
            
            # Assemble recommendation object
            rec = {
                "id": movie,
                "title": title,
                "year": year,
                "genres": genres,
                "rank": i + 1,
                "score": candidate['final_score'],
                "explanation": explanation,
                "meta_paths": [p["path_name"] for p in candidate["paths"][:3]],
                "community": community_info,
                "score_components": {
                    "meta_path_score": candidate['meta_path_score'],
                    "embedding_score": candidate['embedding_score'],
                    "community_score": candidate['community_score'],
                    "model_score": candidate['model_score'],
                    "context_score": candidate['context_score'],
                    "novelty_score": candidate['novelty_score']
                }
            }
            
            recommendations.append(rec)
        
        self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def explain_recommendation(self, user_id, movie_id, detail_level="high"):
        """
        Generate detailed explanation for a recommendation
        
        Args:
            user_id: User node ID
            movie_id: Movie node ID
            detail_level: Detail level for explanation
            
        Returns:
            Dictionary with explanation components
        """
        # Find meta-paths
        paths = self.meta_path_reasoner.find_paths(user_id, movie_id)
        
        if not paths:
            return {"explanation": "I don't have a good explanation for this recommendation."}
        
        # Generate primary explanation
        primary_explanation = self.meta_path_reasoner.generate_explanation(paths[0], personalize=True)
        
        # Get alternative explanations from different paths
        alternative_explanations = []
        for path in paths[1:3]:  # Use next 2 paths
            explanation = self.meta_path_reasoner.generate_explanation(path, personalize=True)
            alternative_explanations.append(explanation)
        
        # Get community-based explanation
        community_explanation = None
        movie_community = self.community_detector.get_community_for_node(movie_id)
        user_community = self.community_detector.get_community_for_node(user_id)
        
        if movie_community is not None:
            community_profile = self.community_detector.get_community_profile(movie_community)
            
            if user_community == movie_community:
                community_explanation = f"This movie is from a community that matches your viewing patterns closely. {self.community_detector.get_community_description(movie_community)}"
            else:
                community_explanation = f"This movie is from a different community than your usual preferences, which might provide a fresh perspective. {self.community_detector.get_community_description(movie_community)}"
        
        # Get similarity-based explanation
        similarity_explanation = None
        if self.embeddings and user_id in self.embeddings and movie_id in self.embeddings:
            similarity = cosine_similarity(
                self.embeddings[user_id].reshape(1, -1),
                self.embeddings[movie_id].reshape(1, -1)
            )[0][0]
            
            if similarity > 0.7:
                similarity_explanation = f"This movie is very similar to your overall preferences ({similarity:.1%} match)."
            elif similarity > 0.5:
                similarity_explanation = f"This movie has moderate similarity to your preferences ({similarity:.1%} match)."
            else:
                similarity_explanation = f"This movie is somewhat different from your usual preferences ({similarity:.1%} match), which might offer a new experience."
        
        # Assemble complete explanation
        explanation = {
            "explanation": primary_explanation,
            "alternative_explanations": alternative_explanations,
            "community_explanation": community_explanation,
            "similarity_explanation": similarity_explanation,
            "meta_paths": [p["path_name"] for p in paths[:3]],
            "confidence": paths[0]["score"]
        }
        
        return explanation
    
    def find_diverse_recommendations(self, user_id, n=10, explanation_detail="high"):
        """
        Find diverse recommendations across different reasons/facets
        
        Args:
            user_id: User node ID
            n: Number of recommendations
            explanation_detail: Level of detail in explanations
            
        Returns:
            List of diverse recommendation objects
        """
        # Get user's community
        user_community = self.community_detector.get_community_for_node(user_id)
        
        # Track facets we've used
        used_facets = set()
        diverse_recs = []
        
        # Get initial recommendations with high diversity
        initial_recs = self.recommend(
            user_id, 
            n=n*2,  # Get more than needed for filtering
            diversity=0.8,
            novelty=0.6,
            explanation_detail=explanation_detail
        )
        
        if not initial_recs:
            return []
        
        # Find recommendations with different meta-path types
        path_type_recs = defaultdict(list)
        
        for rec in initial_recs:
            for path_name in rec.get("meta_paths", []):
                # Extract key relationship
                path_elements = path_name.split("->")
                
                # Identify key entity type in path
                key_entity = None
                if "actor" in path_name:
                    key_entity = "actor"
                elif "director" in path_name:
                    key_entity = "director"
                elif "genre" in path_name:
                    key_entity = "genre"
                
                if key_entity:
                    path_type_recs[key_entity].append(rec)
        
        # Add one recommendation from each key path type
        for entity_type in ["actor", "director", "genre"]:
            if entity_type in path_type_recs and path_type_recs[entity_type]:
                # Get first recommendation of this type not already included
                for rec in path_type_recs[entity_type]:
                    if rec["id"] not in [r["id"] for r in diverse_recs]:
                        diverse_recs.append(rec)
                        used_facets.add(entity_type)
                        break
        
        # Add community-based recommendation (from different community)
        if user_community is not None:
            for rec in initial_recs:
                if "community" in rec and rec["community"]:
                    rec_community = rec["community"].get("id")
                    
                    if rec_community != user_community and rec["id"] not in [r["id"] for r in diverse_recs]:
                        diverse_recs.append(rec)
                        used_facets.add("community")
                        break
        
        # Add novelty-based recommendation
        novelty_candidates = sorted(initial_recs, key=lambda x: x["score_components"]["novelty_score"], reverse=True)
        
        for rec in novelty_candidates:
            if rec["id"] not in [r["id"] for r in diverse_recs]:
                diverse_recs.append(rec)
                used_facets.add("novelty")
                break
        
        # Fill remaining with top recommendations
        for rec in initial_recs:
            if len(diverse_recs) >= n:
                break
                
            if rec["id"] not in [r["id"] for r in diverse_recs]:
                diverse_recs.append(rec)
        
        # Limit to n items
        diverse_recs = diverse_recs[:n]
        
        # Add diversity category to each recommendation
        for i, rec in enumerate(diverse_recs):
            if i < len(used_facets):
                rec["diversity_category"] = list(used_facets)[i]
            else:
                rec["diversity_category"] = "general"
        
        return diverse_recs
    
    def get_similar_items(self, item_id, n=5, item_type="movie"):
        """
        Find similar items based on embeddings and meta-paths
        
        Args:
            item_id: Item node ID
            n: Number of similar items to return
            item_type: Type of item ("movie", "actor", "director", etc.)
            
        Returns:
            List of similar item objects
        """
        if item_id not in self.graph:
            return []
        
        # Get all items of the requested type
        items = []
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type", "").lower() == item_type.lower() and node != item_id:
                items.append(node)
        
        # Calculate similarity using embeddings
        embedding_similarities = []
        
        if self.embeddings and item_id in self.embeddings:
            item_embedding = self.embeddings[item_id]
            
            for other_item in items:
                if other_item in self.embeddings:
                    similarity = cosine_similarity(
                        item_embedding.reshape(1, -1),
                        self.embeddings[other_item].reshape(1, -1)
                    )[0][0]
                    
                    embedding_similarities.append((other_item, similarity))
            
            # Sort by similarity
            embedding_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate similarity using meta-paths
        path_similarities = []
        
        for other_item in items:
            # Find paths between items
            paths = self.meta_path_reasoner.find_paths(item_id, other_item, max_length=3)
            
            if paths:
                # Use highest path score as similarity
                path_similarities.append((other_item, paths[0]["score"]))
        
        # Sort by path similarity
        path_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Combine similarities
        combined_similarities = {}
        
        # Add embedding similarities
        for item, sim in embedding_similarities[:n*2]:
            combined_similarities[item] = {"embedding_similarity": sim, "path_similarity": 0.0}
        
        # Add path similarities
        for item, sim in path_similarities[:n*2]:
            if item in combined_similarities:
                combined_similarities[item]["path_similarity"] = sim
            else:
                combined_similarities[item] = {"embedding_similarity": 0.0, "path_similarity": sim}
        
        # Calculate combined score
        for item, sims in combined_similarities.items():
            combined_similarities[item]["combined_score"] = 0.6 * sims["embedding_similarity"] + 0.4 * sims["path_similarity"]
        
        # Sort by combined score
        sorted_items = sorted(combined_similarities.items(), key=lambda x: x[1]["combined_score"], reverse=True)
        
        # Build result objects
        similar_items = []
        
        for item, scores in sorted_items[:n]:
            # Get item data
            item_data = self.graph.nodes[item]
            
            # Build result based on item type
            if item_type.lower() == "movie":
                result = {
                    "id": item,
                    "title": item_data.get("title", item),
                    "year": item_data.get("year", ""),
                    "genres": json.loads(item_data.get("genres", "[]")) if isinstance(item_data.get("genres"), str) else item_data.get("genres", []),
                    "similarity": scores["combined_score"],
                    "embedding_similarity": scores["embedding_similarity"],
                    "path_similarity": scores["path_similarity"]
                }
            else:
                result = {
                    "id": item,
                    "name": item_data.get("name", item),
                    "similarity": scores["combined_score"],
                    "embedding_similarity": scores["embedding_similarity"],
                    "path_similarity": scores["path_similarity"]
                }
            
            # Add explanation
            if scores["path_similarity"] > 0:
                # Find paths for explanation
                paths = self.meta_path_reasoner.find_paths(item_id, item, max_length=3)
                
                if paths:
                    path_explanation = self.meta_path_reasoner.generate_explanation(
                        paths[0],
                        personalize=False
                    )
                    result["explanation"] = path_explanation
            elif scores["embedding_similarity"] > 0:
                # Use embedding similarity for explanation
                orig_item_data = self.graph.nodes[item_id]
                orig_name = orig_item_data.get("name", orig_item_data.get("title", item_id))
                
                similarity_pct = int(scores["embedding_similarity"] * 100)
                result["explanation"] = f"This is {similarity_pct}% similar to {orig_name} based on overall characteristics."
            
            similar_items.append(result)
        
        return similar_items

#############################################################
#                  UnifiedQuantumKG                         #
#############################################################

class UnifiedQuantumKG:
    """
    Unified Quantum Knowledge Graph for advanced recommendations with conversation support
    """
    
    def __init__(self):
        """Initialize the unified quantum knowledge graph"""
        self.graph = None
        self.textual_embeddings = None
        self.neural_embeddings = None
        self.combined_embeddings = None
        self.model = None
        self.meta_path_reasoner = None
        self.community_detector = None
        self.recommender = None
        self.dialogue_history = {}
        self.dialogue_context = {}
        
        self.logger = logging.getLogger("quantum_kg")
    
    def load_from_files(self, graph_path, textual_embeddings_path=None, 
                       neural_embeddings_path=None, model_path=None):
        """
        Load knowledge graph from files
        
        Args:
            graph_path: Path to NetworkX graph file (pickle or GraphML)
            textual_embeddings_path: Path to textual embeddings file (pickle)
            neural_embeddings_path: Path to neural embeddings file (pickle)
            model_path: Path to PyTorch model file (pt)
            
        Returns:
            Self for method chaining
        """
        # Load graph
        self.logger.info(f"Loading graph from {graph_path}")
        
        if graph_path.endswith(".pkl") or graph_path.endswith(".pickle"):
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
        elif graph_path.endswith(".graphml"):
            self.graph = nx.read_graphml(graph_path)
        else:
            raise ValueError(f"Unsupported graph file format: {graph_path}")
        
        self.logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Load textual embeddings if provided
        if textual_embeddings_path:
            self.logger.info(f"Loading textual embeddings from {textual_embeddings_path}")
            
            try:
                with open(textual_embeddings_path, 'rb') as f:
                    self.textual_embeddings = pickle.load(f)
                
                self.logger.info(f"Loaded textual embeddings for {len(self.textual_embeddings)} nodes")
            except Exception as e:
                self.logger.error(f"Error loading textual embeddings: {str(e)}")
        
        # Load neural embeddings if provided
        if neural_embeddings_path:
            self.logger.info(f"Loading neural embeddings from {neural_embeddings_path}")
            
            try:
                with open(neural_embeddings_path, 'rb') as f:
                    self.neural_embeddings = pickle.load(f)
                
                self.logger.info(f"Loaded neural embeddings for {len(self.neural_embeddings)} nodes")
            except Exception as e:
                self.logger.error(f"Error loading neural embeddings: {str(e)}")
        
        # Load model if provided
        if model_path:
            self.logger.info(f"Loading model from {model_path}")
            
            try:
                if torch.cuda.is_available():
                    self.model = torch.load(model_path)
                else:
                    self.model = torch.load(model_path, map_location=torch.device('cpu'))
                
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
        
        # Combine embeddings
        self._combine_embeddings()
        
        # Initialize components
        self._initialize_components()
        
        return self
    
    def load_from_kg_builder(self, kg_builder, neural_model_dir=None):
        """
        Load from BollywoodKGBuilder instance and neural model
        
        Args:
            kg_builder: BollywoodKGBuilder instance
            neural_model_dir: Directory with neural model outputs
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Loading from KG Builder")
        
        # Copy graph and textual embeddings
        self.graph = kg_builder.graph
        self.textual_embeddings = kg_builder.embeddings
        
        self.logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        self.logger.info(f"Loaded textual embeddings for {len(self.textual_embeddings)} nodes")
        
        # Load neural model if directory provided
        if neural_model_dir:
            # Find required files
            model_path = os.path.join(neural_model_dir, "enhanced_hybrid_model.pt")
            embeddings_path = os.path.join(neural_model_dir, "node_embeddings.pkl")
            
            # Load neural embeddings if available
            if os.path.exists(embeddings_path):
                self.logger.info(f"Loading neural embeddings from {embeddings_path}")
                
                try:
                    with open(embeddings_path, 'rb') as f:
                        self.neural_embeddings = pickle.load(f)
                    
                    self.logger.info(f"Loaded neural embeddings for {len(self.neural_embeddings)} nodes")
                except Exception as e:
                    self.logger.error(f"Error loading neural embeddings: {str(e)}")
            
            # Load model if available
            if os.path.exists(model_path):
                self.logger.info(f"Loading model from {model_path}")
                
                try:
                    if torch.cuda.is_available():
                        self.model = torch.load(model_path)
                    else:
                        self.model = torch.load(model_path, map_location=torch.device('cpu'))
                    
                    self.logger.info("Model loaded successfully")
                except Exception as e:
                    self.logger.error(f"Error loading model: {str(e)}")
        
        # Combine embeddings
        self._combine_embeddings()
        
        # Initialize components
        self._initialize_components()
        
        return self
    
    def _combine_embeddings(self):
        """Combine textual and neural embeddings for enhanced representations"""
        # If we only have one embedding type, use that
        if self.textual_embeddings and not self.neural_embeddings:
            self.combined_embeddings = self.textual_embeddings
            return
        
        if self.neural_embeddings and not self.textual_embeddings:
            self.combined_embeddings = self.neural_embeddings
            return
        
        if not self.textual_embeddings and not self.neural_embeddings:
            self.combined_embeddings = {}
            return
        
        # Combine embeddings using quantum-inspired superposition
        self.logger.info("Combining textual and neural embeddings")
        
        combined = {}
        
        # Process nodes present in both embedding sets
        common_nodes = set(self.textual_embeddings.keys()) & set(self.neural_embeddings.keys())
        
        for node in common_nodes:
            text_emb = self.textual_embeddings[node]
            neural_emb = self.neural_embeddings[node]
            
            # Normalize embeddings
            text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-9)
            neural_emb = neural_emb / (np.linalg.norm(neural_emb) + 1e-9)
            
            # Determine optimal dimensionality
            text_dim = text_emb.shape[0]
            neural_dim = neural_emb.shape[0]
            
            # Resize if needed
            if text_dim != neural_dim:
                # Resize to smaller dimension
                min_dim = min(text_dim, neural_dim)
                text_emb = text_emb[:min_dim]
                neural_emb = neural_emb[:min_dim]
            
            # Quantum-inspired combination (superposition with phase)
            # Generate random phase angle
            phase = 0.25 * np.pi  # 45 degrees provides equal weighting
            
            # Apply phase rotation to create superposition
            combined_emb = np.cos(phase) * text_emb + np.sin(phase) * neural_emb
            
            # Normalize
            combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-9)
            
            combined[node] = combined_emb
        
        # Add nodes only in textual embeddings
        for node in set(self.textual_embeddings.keys()) - set(self.neural_embeddings.keys()):
            combined[node] = self.textual_embeddings[node]
        
        # Add nodes only in neural embeddings
        for node in set(self.neural_embeddings.keys()) - set(self.textual_embeddings.keys()):
            combined[node] = self.neural_embeddings[node]
        
        self.combined_embeddings = combined
        self.logger.info(f"Created combined embeddings for {len(combined)} nodes")
    
    def _initialize_components(self):
        """Initialize all components with the unified embeddings"""
        # Initialize meta-path reasoner
        self.meta_path_reasoner = AdvancedMetaPathReasoner(self.graph)
        
        # Extract meta-path schemas
        self.meta_path_reasoner.extract_meta_path_schemas()
        
        # Initialize community detector
        self.community_detector = EnhancedCommunityDetector(self.graph)
        
        
        self.community_detector.detect_communities(
            method="louvian",        # More reliable method
            resolution=0.7,          # Lower value for larger communities
            min_size=5,              # Minimum community size
            hierarchical=True,       # Enable hierarchical detection
            overlapping=True,        # Enable overlapping detection
            levels=3                 # Number of hierarchical levels
        )
        
        # Initialize recommender
        self.recommender = QuantumEnhancedRecommender(
            self.graph, 
            self.combined_embeddings, 
            self.model
        )
        
        # Initialize dialogue history
        self.dialogue_history = {}
        self.dialogue_context = {}
        
        self.logger.info("All components initialized")
    
    def create_quantum_model(self):
        """Create a new quantum-enhanced model from graph structure"""
        try:
            self.logger.info("Creating new quantum-enhanced GNN model")
            
            # Extract node features
            node_features = []
            node_types = []
            for node in self.graph.nodes():
                # Use embeddings as features if available
                if node in self.combined_embeddings:
                    node_features.append(self.combined_embeddings[node])
                else:
                    # Create dummy features
                    dim = next(iter(self.combined_embeddings.values())).shape[0] if self.combined_embeddings else 128
                    node_features.append(np.zeros(dim))
                
                # Extract node type
                node_data = self.graph.nodes[node]
                node_type = node_data.get("node_type", "unknown")
                
                # Convert to numeric type index
                if node_type == "movie":
                    type_idx = 0
                elif node_type == "user":
                    type_idx = 1
                elif node_type == "actor":
                    type_idx = 2
                elif node_type == "director":
                    type_idx = 3
                elif node_type == "genre":
                    type_idx = 4
                else:
                    type_idx = 5
                
                node_types.append(type_idx)
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            node_type_tensor = torch.tensor(node_types, dtype=torch.long)
            
            # Extract edge information
            edge_list = []
            edge_attr_list = []
            edge_type_list = []
            
            for u, v, data in self.graph.edges(data=True):
                # Get node indices
                u_idx = list(self.graph.nodes()).index(u)
                v_idx = list(self.graph.nodes()).index(v)
                
                # Add edge
                edge_list.append([u_idx, v_idx])
                
                # Get edge type
                edge_type = data.get("edge_type", "unknown")
                
                # Convert to numeric type index
                if edge_type == "rated":
                    type_idx = 0
                elif edge_type == "acted_in":
                    type_idx = 1
                elif edge_type == "directed":
                    type_idx = 2
                elif edge_type == "has_genre":
                    type_idx = 3
                else:
                    type_idx = 4
                
                edge_type_list.append(type_idx)
                
                # Create edge attributes
                edge_attr = [0.0] * 8  # 8-dimensional edge attributes
                
                # Add weight if available
                weight = data.get("weight", 1.0)
                edge_attr[0] = float(weight)
                
                # Add rating if available
                rating = data.get("rating", 0.0)
                edge_attr[1] = float(rating)
                
                # Add one-hot encoding of edge type
                edge_attr[2 + type_idx] = 1.0
                
                edge_attr_list.append(edge_attr)
            
            # Convert to tensors
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            edge_type = torch.tensor(edge_type_list, dtype=torch.long)
            
            # Create model
            in_dim = x.size(1)
            hidden_dim = 256
            out_dim = 128
            
            model = QuantumEnhancedHybridGNN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_type=edge_type,
                node_type=node_type_tensor,
                time_steps=6,
                q_layers=4,
                dropout=0.1
            )
            
            self.model = model
            self.logger.info(f"Created quantum-enhanced GNN model with {in_dim} input dim, {hidden_dim} hidden dim, {out_dim} output dim")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating quantum model: {str(e)}")
            traceback.print_exc()
            return None
    
    def save(self, output_dir):
        """
        Save unified quantum knowledge graph to files
        
        Args:
            output_dir: Output directory
            
        Returns:
            Self for method chaining
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save graph
        graph_path = os.path.join(output_dir, "quantum_unified_graph.pkl")
        self.logger.info(f"Saving graph to {graph_path}")
        
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save graph in GraphML format for visualization
        graphml_path = os.path.join(output_dir, "quantum_unified_graph.graphml")
        self.logger.info(f"Saving graph in GraphML format to {graphml_path}")
        
        try:
            # Need to convert node attributes to be GraphML compatible
            graph_copy = self.graph.copy()
            
            for node, data in graph_copy.nodes(data=True):
                # Convert non-string attributes
                for key, value in list(data.items()):
                    if isinstance(value, (list, dict, set)):
                        try:
                            data[key] = json.dumps(value)
                        except:
                            data[key] = str(value)
                    elif value is None:
                        data[key] = ""
                    elif not isinstance(value, (str, int, float, bool)):
                        data[key] = str(value)
            
            # Convert edge attributes
            for u, v, data in graph_copy.edges(data=True):
                for key, value in list(data.items()):
                    if isinstance(value, (list, dict, set)):
                        try:
                            data[key] = json.dumps(value)
                        except:
                            data[key] = str(value)
                    elif value is None:
                        data[key] = ""
                    elif not isinstance(value, (str, int, float, bool)):
                        data[key] = str(value)
            
            nx.write_graphml(graph_copy, graphml_path)
            
        except Exception as e:
            self.logger.error(f"Error saving GraphML: {str(e)}")
            
            # Try with a simplified graph
            try:
                simple_graph = nx.Graph()
                
                for node, data in self.graph.nodes(data=True):
                    simple_attrs = {
                        "name": str(data.get("name", data.get("title", node))),
                        "node_type": str(data.get("node_type", "unknown"))
                    }
                    
                    if "community_id" in data:
                        simple_attrs["community_id"] = str(data["community_id"])
                    
                    simple_graph.add_node(node, **simple_attrs)
                
                for u, v, data in self.graph.edges(data=True):
                    simple_attrs = {
                        "edge_type": str(data.get("edge_type", "unknown"))
                    }
                    
                    simple_graph.add_edge(u, v, **simple_attrs)
                
                nx.write_graphml(simple_graph, graphml_path)
                self.logger.info("Saved simplified GraphML")
                
            except Exception as e2:
                self.logger.error(f"Error saving simplified GraphML: {str(e2)}")
        
        # Save embeddings
        if self.textual_embeddings:
            textual_path = os.path.join(output_dir, "textual_embeddings.pkl")
            with open(textual_path, 'wb') as f:
                pickle.dump(self.textual_embeddings, f)
        
        if self.neural_embeddings:
            neural_path = os.path.join(output_dir, "neural_embeddings.pkl")
            with open(neural_path, 'wb') as f:
                pickle.dump(self.neural_embeddings, f)
        
        if self.combined_embeddings:
            combined_path = os.path.join(output_dir, "quantum_combined_embeddings.pkl")
            with open(combined_path, 'wb') as f:
                pickle.dump(self.combined_embeddings, f)
        
        # Save model if available
        if self.model:
            model_path = os.path.join(output_dir, "quantum_unified_model.pt")
            self.logger.info(f"Saving model to {model_path}")
            
            torch.save(self.model, model_path)
        
        # Save community information
        community_path = os.path.join(output_dir, "quantum_community_profiles.json")
        self.logger.info(f"Saving community profiles to {community_path}")
        
        community_data = {}
        for comm_id, profile in self.community_detector.community_profiles.items():
            # Convert numpy arrays to lists for JSON serialization
            serializable_profile = {}
            for key, value in profile.items():
                if isinstance(value, np.ndarray):
                    serializable_profile[key] = value.tolist()
                elif isinstance(value, set):
                    serializable_profile[key] = list(value)
                else:
                    serializable_profile[key] = value
            
            community_data[str(comm_id)] = serializable_profile
        
        with open('output.json', 'w') as f:
            json.dump(community_data, f, indent=2, cls=NumpyEncoder)
        
        # Save meta-path schemas
        metapath_path = os.path.join(output_dir, "quantum_metapath_schemas.json")
        self.logger.info(f"Saving meta-path schemas to {metapath_path}")
        
        metapath_data = {
            "schemas": {k: list(v) for k, v in self.meta_path_reasoner.meta_path_schemas.items()}
        }
        
        with open(metapath_path, 'w') as f:
            json.dump(metapath_data, f, indent=2)

        community_viz_path = os.path.join(output_dir, "quantum_community_structure.png")
        if hasattr(self.community_detector, 'visualize_communities'):
            self.community_detector.visualize_communities(
                output_path=community_viz_path,
                max_nodes=2000,
                node_size_factor=20,
                show_labels=True
            )
        
        # Add community statistics to a text file
        stats_path = os.path.join(output_dir, "community_statistics.txt")
        with open(stats_path, 'w') as f:
            # Redirect stdout to file temporarily
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            if hasattr(self.community_detector, 'print_community_statistics'):
                self.community_detector.print_community_statistics()
            sys.stdout = old_stdout
        
        self.logger.info(f"Unified Quantum Knowledge Graph saved to {output_dir}")
        return self
    def recommend_for_user(self, user_id, n=10, diversity=0.5, 
                         novelty=0.5, include_watched=False, 
                         context=None, explanation_detail="high"):
        """
        Generate personalized recommendations
        
        Args:
            user_id: User node ID
            n: Number of recommendations
            diversity: Diversity parameter [0, 1]
            novelty: Novelty parameter [0, 1]
            include_watched: Whether to include previously watched items
            context: Conversation context dictionary
            explanation_detail: Level of detail in explanations
            
        Returns:
            List of recommendation objects with explanations
        """
        return self.recommender.recommend(
            user_id, 
            n=n, 
            diversity=diversity,
            novelty=novelty,
            include_watched=include_watched,
            context=context or self.dialogue_context.get(user_id, {}),
            explanation_detail=explanation_detail
        )
    
    def find_diverse_recommendations(self, user_id, n=10):
        """
        Find diverse recommendations across different reasons
        
        Args:
            user_id: User node ID
            n: Number of recommendations
            
        Returns:
            List of diverse recommendation objects
        """
        return self.recommender.find_diverse_recommendations(
            user_id, 
            n=n, 
            explanation_detail="high"
        )
    
    def explain_recommendation(self, user_id, movie_id, detail_level="high"):
        """
        Generate detailed explanation for a recommendation
        
        Args:
            user_id: User node ID
            movie_id: Movie node ID
            detail_level: Detail level for explanation
            
        Returns:
            Dictionary with explanation components
        """
        return self.recommender.explain_recommendation(
            user_id, 
            movie_id, 
            detail_level=detail_level
        )
    
    def add_dialogue_turn(self, user_id, user_utterance, system_response=None, entities=None):
        """
        Add a dialogue turn for a user
        
        Args:
            user_id: User ID
            user_utterance: User's message
            system_response: System's response (optional)
            entities: Dictionary of entities mentioned (optional)
            
        Returns:
            Updated dialogue context
        """
        # Initialize history if needed
        if user_id not in self.dialogue_history:
            self.dialogue_history[user_id] = []
            self.dialogue_context[user_id] = {
                "mentioned_genres": [],
                "mentioned_actors": [],
                "mentioned_directors": [],
                "mentioned_movies": [],
                "preferences": {},
                "constraints": {},
                "session_ratings": {}
            }
        
        # Create turn object
        turn = {
            "user_utterance": user_utterance,
            "system_response": system_response,
            "entities": entities or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.dialogue_history[user_id].append(turn)
        
        # Limit history size
        max_history = 10
        if len(self.dialogue_history[user_id]) > max_history:
            self.dialogue_history[user_id] = self.dialogue_history[user_id][-max_history:]
        
        # Update context
        self._update_dialogue_context(user_id, turn)
        
        return self.dialogue_context[user_id]
    
    def _update_dialogue_context(self, user_id, turn):
        """
        Update dialogue context based on turn
        
        Args:
            user_id: User ID
            turn: Dialogue turn object
        """
        context = self.dialogue_context[user_id]
        entities = turn.get("entities", {})
        
        # Update mentioned entities
        for entity_type, entity_list in entities.items():
            if entity_type == "genre":
                for entity in entity_list:
                    genre = entity.get("name")
                    if genre and genre not in context["mentioned_genres"]:
                        context["mentioned_genres"].append(genre)
            
            elif entity_type == "actor":
                for entity in entity_list:
                    actor = entity.get("name")
                    if actor and actor not in context["mentioned_actors"]:
                        context["mentioned_actors"].append(actor)
            
            elif entity_type == "director":
                for entity in entity_list:
                    director = entity.get("name")
                    if director and director not in context["mentioned_directors"]:
                        context["mentioned_directors"].append(director)
            
            elif entity_type == "movie":
                for entity in entity_list:
                    movie = entity.get("id")
                    if movie and movie not in context["mentioned_movies"]:
                        context["mentioned_movies"].append(movie)
        
        # Update preferences
        if "preferences" in entities:
            for pref in entities["preferences"]:
                pref_type = pref.get("type")
                pref_value = pref.get("value")
                
                if pref_type and pref_value:
                    context["preferences"][pref_type] = pref_value
        
        # Update constraints
        if "constraints" in entities:
            for constraint in entities["constraints"]:
                constraint_type = constraint.get("type")
                constraint_value = constraint.get("value")
                
                if constraint_type and constraint_value:
                    context["constraints"][constraint_type] = constraint_value
        
        # Update session ratings
        if "ratings" in entities:
            for rating in entities["ratings"]:
                movie_id = rating.get("movie_id")
                rating_value = rating.get("value")
                
                if movie_id and rating_value is not None:
                    context["session_ratings"][movie_id] = rating_value
        
        # Update context
        self.dialogue_context[user_id] = context
    
    def get_dialogue_context(self, user_id):
        """
        Get dialogue context for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Context dictionary
        """
        return self.dialogue_context.get(user_id, {})
    
    def get_dialogue_context_summary(self, user_id):
        """
        Get summary of dialogue context
        
        Args:
            user_id: User ID
            
        Returns:
            Context summary string
        """
        context = self.get_dialogue_context(user_id)
        
        if not context:
            return "No context available."
            
        summary = []
        
        # Mentioned entities
        if context["mentioned_genres"]:
            summary.append(f"Genres: {', '.join(context['mentioned_genres'][:3])}")
        
        if context["mentioned_actors"]:
            summary.append(f"Actors: {', '.join(context['mentioned_actors'][:3])}")
        
        if context["mentioned_directors"]:
            summary.append(f"Directors: {', '.join(context['mentioned_directors'][:3])}")
        
        if context["mentioned_movies"]:
            summary.append(f"Movies: {len(context['mentioned_movies'])} mentioned")
        
        # Preferences
        if context["preferences"]:
            prefs = []
            for pref_type, pref_value in context["preferences"].items():
                prefs.append(f"{pref_type}: {pref_value}")
            
            summary.append(f"Preferences: {'; '.join(prefs[:3])}")
        
        # Constraints
        if context["constraints"]:
            constraints = []
            for constraint_type, constraint_value in context["constraints"].items():
                constraints.append(f"{constraint_type}: {constraint_value}")
            
            summary.append(f"Constraints: {'; '.join(constraints[:3])}")
        
        # Ratings
        if context["session_ratings"]:
            summary.append(f"Session ratings: {len(context['session_ratings'])} movies rated")
        
        if not summary:
            return "No meaningful context available."
            
        return " | ".join(summary)
    
    def visualize_graph(self, output_path, max_nodes=1000, community_colors=True):
        """
        Visualize graph with communities
        
        Args:
            output_path: Output file path
            max_nodes: Maximum number of nodes to include
            community_colors: Whether to color nodes by community
            
        Returns:
                Self for method chaining
            """
        self.logger.info(f"Visualizing graph with max {max_nodes} nodes")
        
        # Use community detector's visualization
        if community_colors and hasattr(self.community_detector, 'visualize_communities'):
            self.community_detector.visualize_communities(
                output_path=output_path,
                max_nodes=max_nodes,
                node_size_factor=30,
                show_labels=True
            )
            return self
        
        # Original visualization code as fallback
        # Create a smaller graph if needed
        if self.graph.number_of_nodes() > max_nodes:
            # Get most important nodes based on degree centrality
            centrality = nx.degree_centrality(self.graph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            
            # Extract node IDs
            node_ids = [node for node, _ in top_nodes]
            
            # Create subgraph
            subgraph = self.graph.subgraph(node_ids).copy()
        else:
            subgraph = self.graph
        
        # Prepare visualization
        plt.figure(figsize=(20, 20))
        
        # Get node colors
        if community_colors:
            # Color by community
            node_colors = []
            community_map = {}
            
            for node in subgraph.nodes():
                comm_id = subgraph.nodes[node].get("community_id", 0)
                
                if comm_id not in community_map:
                    community_map[comm_id] = len(community_map)
                
                node_colors.append(community_map[comm_id])
        else:
            # Color by node type
            node_colors = []
            node_type_map = {
                "movie": 0,
                "user": 1,
                "actor": 2,
                "director": 3,
                "genre": 4,
                "person": 5
            }
            
            for node in subgraph.nodes():
                node_type = subgraph.nodes[node].get("node_type", "unknown").lower()
                node_colors.append(node_type_map.get(node_type, 6))
        
        # Get node sizes based on degree
        node_sizes = [20 + 10 * subgraph.degree(node) for node in subgraph.nodes()]
        
        # Use spring layout
        pos = nx.spring_layout(subgraph, k=0.3, iterations=50, seed=42)
        
        # Draw graph
        nx.draw_networkx(
            subgraph,
            pos=pos,
            node_color=node_colors,
            cmap=plt.cm.tab20,
            node_size=node_sizes,
            alpha=0.8,
            with_labels=False,
            edge_color="gray",
            width=0.5
        )
        
        # Add labels for important nodes
        top_nodes = {}
        
        for node in subgraph.nodes():
            # Label nodes with high degree
            if subgraph.degree(node) > subgraph.number_of_nodes() / 20:
                name = subgraph.nodes[node].get("name", 
                                            subgraph.nodes[node].get("title", node))
                top_nodes[node] = name
        
        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels=top_nodes,
            font_size=12,
            font_color="black",
            font_weight="bold"
        )
        
        # Add title
        if community_colors:
            plt.title(f"Quantum Unified Knowledge Graph with {len(community_map)} Communities")
        else:
            plt.title(f"Quantum Unified Knowledge Graph by Node Types")
        
        plt.axis('off')
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Graph visualization saved to {output_path}")
        return self
    
    def analyze_communities(self, output_dir=None):
        """
        Analyze community structure and generate visualizations
        
        Args:
            output_dir: Output directory for visualizations (optional)
            
        Returns:
            Dictionary with community statistics
        """
        stats = {}
        
        # Get basic statistics
        stats["total_communities"] = len(self.community_detector.community_to_nodes)
        stats["modularity"] = self.community_detector.modularity
        
        # Calculate size statistics
        community_sizes = [len(nodes) for nodes in self.community_detector.community_to_nodes.values()]
        stats["min_size"] = min(community_sizes) if community_sizes else 0
        stats["max_size"] = max(community_sizes) if community_sizes else 0
        stats["avg_size"] = sum(community_sizes) / len(community_sizes) if community_sizes else 0
        
        # Calculate type distribution
        type_distribution = Counter()
        for profile in self.community_detector.community_profiles.values():
            main_type = profile.get("main_type", "unknown")
            type_distribution[main_type] += 1
        
        stats["type_distribution"] = dict(type_distribution)
        
        # Generate visualizations if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Community structure visualization
            if hasattr(self.community_detector, 'visualize_communities'):
                viz_path = os.path.join(output_dir, "community_structure.png")
                self.community_detector.visualize_communities(
                    output_path=viz_path,
                    max_nodes=2000
                )
                stats["structure_viz"] = viz_path
            
            # Size distribution plot
            size_viz_path = os.path.join(output_dir, "community_size_distribution.png")
            plt.figure(figsize=(10, 6))
            plt.hist(community_sizes, bins=50)
            plt.xlabel('Community Size')
            plt.ylabel('Count')
            plt.title('Community Size Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(size_viz_path)
            plt.close()
            stats["size_viz"] = size_viz_path
            
            # Type distribution plot
            type_viz_path = os.path.join(output_dir, "community_type_distribution.png")
            plt.figure(figsize=(10, 6))
            plt.bar(type_distribution.keys(), type_distribution.values())
            plt.xlabel('Node Type')
            plt.ylabel('Count')
            plt.title('Community Main Type Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(type_viz_path)
            plt.close()
            stats["type_viz"] = type_viz_path
        
        return stats

#############################################################
#                   Utility Functions                       #
#############################################################

def build_quantum_unified_kg(graph_path, textual_embeddings_path=None, 
                           neural_embeddings_path=None, model_path=None, 
                           output_dir="quantum_kg_output"):
    """
    Build unified quantum knowledge graph from inputs
    
    Args:
        graph_path: Path to graph file
        textual_embeddings_path: Path to textual embeddings file
        neural_embeddings_path: Path to neural embeddings file
        model_path: Path to model file
        output_dir: Output directory
        
    Returns:
        UnifiedQuantumKG instance
    """
    kg = UnifiedQuantumKG()
    
    # Load from files
    kg.load_from_files(
        graph_path=graph_path,
        textual_embeddings_path=textual_embeddings_path,
        neural_embeddings_path=neural_embeddings_path,
        model_path=model_path
    )
    
    # Create quantum model if needed
    if kg.model is None:
        kg.create_quantum_model()
    
    # Save to output directory
    kg.save(output_dir)
    
    # Create visualization
    kg.visualize_graph(
        output_path=os.path.join(output_dir, "quantum_graph_visualization.png"),
        max_nodes=1000,
        community_colors=True
    )
    
    return kg



def build_from_kg_builder_and_neural_model(kg_builder, neural_model_dir, output_dir="quantum_kg_output"):
    """
    Build unified quantum knowledge graph from KG Builder and neural model
    
    Args:
        kg_builder: BollywoodKGBuilder instance
        neural_model_dir: Directory with neural model
        output_dir: Output directory
        
    Returns:
        UnifiedQuantumKG instance
    """
    kg = UnifiedQuantumKG()
    
    # Load from KG builder and neural model
    kg.load_from_kg_builder(
        kg_builder=kg_builder,
        neural_model_dir=neural_model_dir
    )
    
    # Create quantum model if needed
    if kg.model is None:
        kg.create_quantum_model()
    
    # Save to output directory
    kg.save(output_dir)
    
    # Create visualization
    kg.visualize_graph(
        output_path=os.path.join(output_dir, "quantum_graph_visualization.png"),
        max_nodes=1000,
        community_colors=True
    )
    
    return kg

#############################################################
#                       Main Function                       #
#############################################################

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Unified Quantum Knowledge Graph")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph file (.pkl or .graphml)")
    parser.add_argument("--textual-embeddings", type=str, help="Path to textual embeddings file (.pkl)")
    parser.add_argument("--neural-embeddings", type=str, help="Path to neural embeddings file (.pkl)")
    parser.add_argument("--model", type=str, help="Path to model file (.pt)")
    parser.add_argument("--output", type=str, default="quantum_kg_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Build unified KG
    kg = build_quantum_unified_kg(
        graph_path=args.graph,
        textual_embeddings_path=args.textual_embeddings,
        neural_embeddings_path=args.neural_embeddings,
        model_path=args.model,
        output_dir=args.output
    )
    community_stats = kg.analyze_communities(os.path.join(args.output, "community_analysis"))
    
    # Print summary
    print(f"Unified Quantum Knowledge Graph created with:")
    print(f"  - {kg.graph.number_of_nodes()} nodes")
    print(f"  - {kg.graph.number_of_edges()} edges")
    print(f"  - {len(kg.textual_embeddings) if kg.textual_embeddings else 0} textual embeddings")
    print(f"  - {len(kg.neural_embeddings) if kg.neural_embeddings else 0} neural embeddings")
    print(f"  - {len(kg.combined_embeddings) if kg.combined_embeddings else 0} combined quantum embeddings")
    print(f"  - {community_stats['total_communities']} communities with modularity {community_stats['modularity']:.4f}")
    print(f"  - {len(kg.meta_path_reasoner.meta_path_schemas)} meta-path schemas")
    print(f"All outputs saved to {args.output}")
    