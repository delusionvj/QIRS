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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Tuple, Optional, List, Union
from tqdm.auto import tqdm

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import (
    GATConv, GCNConv, SAGEConv, GATv2Conv,
    TransformerConv, HypergraphConv,
    HANConv, HGTConv, Linear
)
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch

# ------------------ Added for Mixed Precision (AMP) ------------------
from torch.cuda.amp import autocast, GradScaler
# ---------------------------------------------------------------------

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    # Fallback if torchdiffeq is not installed
    def odeint(*args, **kwargs):
        raise ImportError("torchdiffeq is not installed.")

###############################################################################
#                              Logging Setup                                  #
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='advanced_model_log.log',
    filemode='w'
)
logger = logging.getLogger("advanced_hybrid_model")

# Optional: fix seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

###############################################################################
#                     Enhanced Graph Data Processor                           #
###############################################################################

class AdvancedGraphDataProcessor:
    """Loads GraphML with enhanced support for heterogeneous graphs and hyperedges"""

    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.logger = logging.getLogger("advanced_hybrid_model.processor")

    def load_graphml(self, path: str) -> nx.Graph:
        """Load graph from GraphML file with robustness against attribute errors"""
        self.logger.info(f"Loading graph from {path}")

        try:
            return nx.read_graphml(path)
        except Exception as e:
            self.logger.warning(f"Error loading GraphML: {str(e)}. Trying with node_type as string...")
            # Try with node_type enforcement
            return nx.read_graphml(path, node_type=str)

    def process_nx_graph(self, nx_graph: nx.Graph, existing_embeddings: Dict = None) -> Tuple[Data, Dict]:
        """
        Convert NetworkX to PyG data with enhanced feature processing

        Args:
            nx_graph: Input NetworkX graph
            existing_embeddings: Optional dictionary of pre-computed embeddings

        Returns:
            data: PyG Data object
            node_type_map: Dictionary mapping node indices to node types
        """
        node_list = list(nx_graph.nodes())
        node_mapping = {node: i for i, node in enumerate(node_list)}

        # Enhanced node type management - create integer mapping
        node_types = []
        node_type_to_idx = {}
        node_idx_to_type = {}

        # Collect node types
        for node, data in nx_graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            if node_type not in node_type_to_idx:
                node_type_to_idx[node_type] = len(node_type_to_idx)

            node_idx = node_mapping[node]
            node_idx_to_type[node_idx] = node_type
            node_types.append(node_type_to_idx[node_type])

        # Track entities by type for community augmentation
        nodes_by_type = {t: [] for t in node_type_to_idx.keys()}
        for node, data in nx_graph.nodes(data=True):
            node_type = data.get("node_type", "unknown").lower()
            nodes_by_type[node_type].append(node_mapping[node])

        # Create comprehensive node features
        node_features = {}
        node_years = {}
        community_ids = {}

        # Process node embeddings
        all_embeddings = []
        target_embed_dim = 384  # Increased dimension for better representation

        for node, data in nx_graph.nodes(data=True):
            node_idx = node_mapping[node]
            feats = []

            # 1. Node type encoding - one-hot across all node types
            node_type = data.get("node_type", "unknown").lower()
            type_oh = [0] * len(node_type_to_idx)
            type_oh[node_type_to_idx[node_type]] = 1
            feats.extend(type_oh)

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

            # Try alternative date formats
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

            # 4. Process existing embeddings if available
            if existing_embeddings and node in existing_embeddings:
                embedding = existing_embeddings[node]
                all_embeddings.append((node_idx, embedding))
                # Add placeholder (will be replaced later)
                feats.extend([0] * target_embed_dim)
            else:
                # Check for embeddings in node attributes
                embedding_hex = data.get("embedding")
                if embedding_hex:
                    try:
                        if isinstance(embedding_hex, str):
                            # Maybe hex string
                            raw_bytes = bytes.fromhex(embedding_hex)
                            embedding = np.frombuffer(raw_bytes, dtype=np.float32)
                            all_embeddings.append((node_idx, embedding))
                            # Add placeholder
                            feats.extend([0] * target_embed_dim)
                        elif isinstance(embedding_hex, list):
                            embedding = np.array(embedding_hex, dtype=np.float32)
                            all_embeddings.append((node_idx, embedding))
                            # Add placeholder
                            feats.extend([0] * target_embed_dim)
                        else:
                            # Default to zeros
                            feats.extend([0] * target_embed_dim)
                    except Exception as e:
                        self.logger.warning(f"Error processing embedding: {e}")
                        feats.extend([0] * target_embed_dim)
                else:
                    # No embedding available
                    feats.extend([0] * target_embed_dim)

            # 5. Additional categorical features for specific node types
            if node_type == "user":
                # User-specific features
                gender_map = {"male": 1, "female": 2, "other": 3}
                gender_value = data.get("gender", "")
                gender = str(gender_value).lower() if gender_value is not None else ""
                gender_val = gender_map.get(gender, 0)

                # One-hot encode gender
                gender_oh = [0, 0, 0, 0]  # unknown, male, female, other
                gender_oh[gender_val] = 1
                feats.extend(gender_oh)

                # Process age (normalized)
                age = data.get("age")
                if age and isinstance(age, (int, float)) and 0 <= age <= 100:
                    norm_age = age / 100
                else:
                    norm_age = 0.5  # Default
                feats.append(norm_age)

                # Language count (normalized)
                languages = data.get("languages", "[]")
                if isinstance(languages, str):
                    try:
                        lang_list = json.loads(languages)
                        lang_count = min(len(lang_list), 10) / 10  # Normalize to 0-1
                    except:
                        lang_count = 0.2  # Default
                else:
                    lang_count = 0.2
                feats.append(lang_count)

                # Rating activity (normalized)
                pref_consistency = data.get("preference_consistency")
                if pref_consistency is not None:
                    try:
                        pref_consistency = float(pref_consistency)
                    except:
                        pref_consistency = 0.5
                else:
                    pref_consistency = 0.5
                feats.append(pref_consistency)

            elif node_type == "movie":
                # Movie-specific features
                language = data.get("language", "").lower()
                lang_map = {"hindi": 1, "english": 2, "tamil": 3, "telugu": 4, "malayalam": 5, "bengali": 6}
                lang_val = lang_map.get(language, 0)

                # One-hot encode language
                lang_oh = [0] * (len(lang_map) + 1)  # +1 for unknown
                lang_oh[lang_val] = 1
                feats.extend(lang_oh)

                # Genre count (normalized)
                genres = data.get("genres", "[]")
                if isinstance(genres, str):
                    try:
                        genre_list = json.loads(genres)
                        if isinstance(genre_list, list):
                            genre_count = min(len(genre_list), 10) / 10  # Normalize to 0-1
                        else:
                            genre_count = 0.3
                    except:
                        genre_count = 0.3
                else:
                    genre_count = 0.3
                feats.append(genre_count)

                # Has box office info
                has_boxoffice = 1.0 if data.get("box_office") else 0.0
                feats.append(has_boxoffice)

                # Has critical reception
                has_reception = 1.0 if data.get("critical_reception") else 0.0
                feats.append(has_reception)

            elif node_type in ["person", "actor", "director", "writer"]:
                # Person-specific features
                known_for = data.get("known_for", "[]")
                if isinstance(known_for, str):
                    try:
                        known_list = json.loads(known_for)
                        known_count = min(len(known_list), 20) / 20  # Normalize to 0-1
                    except:
                        known_count = 0.2
                else:
                    known_count = 0.2
                feats.append(known_count)

                # Career span placeholder
                feats.append(0.5)
                feats.extend([0.0] * 8)  # Padding to match other node types

            elif node_type == "genre":
                # Genre-specific features
                feats.extend([0.0] * 10)

            else:
                # Default padding
                feats.extend([0.0] * 10)

            node_features[node_idx] = feats

        # Process pre-computed embeddings for the placeholders
        if all_embeddings:
            self.logger.info(f"Processing {len(all_embeddings)} node embeddings")

            emb_indices = [idx for idx, _ in all_embeddings]
            emb_arrays = [emb for _, emb in all_embeddings]

            emb_dim = emb_arrays[0].shape[0]

            # If embeddings are too large, reduce dimensionality
            if emb_dim > target_embed_dim:
                try:
                    from sklearn.decomposition import PCA
                    stacked_arrays = np.vstack(emb_arrays)

                    pca = PCA(n_components=target_embed_dim)
                    reduced_embeddings = pca.fit_transform(stacked_arrays)

                    self.logger.info(f"Reduced embeddings from {emb_dim} to {target_embed_dim} dimensions")

                    for i, node_idx in enumerate(emb_indices):
                        feats = node_features[node_idx]
                        embed_start = len(feats) - target_embed_dim
                        feats[embed_start:] = reduced_embeddings[i].tolist()
                        node_features[node_idx] = feats

                except Exception as e:
                    self.logger.warning(f"Error reducing embedding dimensions: {e}")
                    # Fallback to truncation or averaging
                    for i, node_idx in enumerate(emb_indices):
                        feats = node_features[node_idx]
                        embed_start = len(feats) - target_embed_dim
                        if emb_dim >= 2 * target_embed_dim:
                            chunks = np.array_split(emb_arrays[i], target_embed_dim)
                            avg_chunks = [chunk.mean() for chunk in chunks]
                            feats[embed_start:] = avg_chunks
                        else:
                            feats[embed_start:] = emb_arrays[i][:target_embed_dim].tolist()
                        node_features[node_idx] = feats
            else:
                # Embeddings already have correct dimensionality
                for i, node_idx in enumerate(emb_indices):
                    feats = node_features[node_idx]
                    embed_start = len(feats) - target_embed_dim
                    emb_padded = np.pad(emb_arrays[i], (0, max(0, target_embed_dim - len(emb_arrays[i]))))
                    feats[embed_start:] = emb_padded[:target_embed_dim].tolist()
                    node_features[node_idx] = feats

        # Check for dimension consistency
        feature_lengths = [len(node_features[i]) for i in range(len(node_list))]
        if len(set(feature_lengths)) > 1:
            max_length = max(feature_lengths)
            logger.warning(f"Inconsistent feature dimensions detected: {set(feature_lengths)}")
            logger.warning(f"Padding all features to length {max_length}")
            for i in range(len(node_list)):
                if len(node_features[i]) < max_length:
                    padding = [0.0] * (max_length - len(node_features[i]))
                    node_features[i].extend(padding)
            assert all(len(node_features[i]) == max_length for i in range(len(node_list))), "Padding failed"

        x = torch.tensor([node_features[i] for i in range(len(node_list))], dtype=torch.float)

        # Process edges
        edge_list = []
        edge_attr_list = []
        edge_type_list = []

        edge_type_to_idx = {}

        for u, v, edata in nx_graph.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]

            e_type = edata.get("edge_type", "unknown").lower()
            if e_type not in edge_type_to_idx:
                edge_type_to_idx[e_type] = len(edge_type_to_idx)
            e_type_idx = edge_type_to_idx[e_type]

            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])

            edge_type_list.append(e_type_idx)
            edge_type_list.append(e_type_idx)

            weight = float(edata.get("weight", edata.get("edge_weight", edata.get("strength", 1.0))))
            rating = 0.0
            if e_type == "rated" and "rating" in edata:
                try:
                    rating = float(edata["rating"])
                except (ValueError, TypeError):
                    rating = 0.0
            timestamp = float(edata.get("timestamp", 0.0))
            rating_mask = 1.0 if e_type == "rated" and "rating" in edata else 0.0

            e_feats = [weight, rating, timestamp, rating_mask]
            edge_attr_list.append(e_feats)
            edge_attr_list.append(e_feats)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        if edge_attr.size(1) != 4:
            self.logger.warning(f"Edge attributes have {edge_attr.size(1)} features instead of 4.")
            fixed_edge_attr = torch.zeros((edge_attr.size(0), 4), dtype=torch.float)
            for i in range(edge_attr.size(0)):
                for j in range(min(edge_attr.size(1), 4)):
                    fixed_edge_attr[i, j] = edge_attr[i, j]
            edge_attr = fixed_edge_attr
            self.logger.info("Edge attribute dimensions fixed to size 4")
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
        data.node_type = torch.tensor(node_types, dtype=torch.long)

        if community_ids:
            max_community = max(community_ids.values()) + 1
            communities = torch.zeros(len(node_list), dtype=torch.long)
            for node_idx, comm_id in community_ids.items():
                communities[node_idx] = comm_id
            data.node_community = communities

        data.node_mapping = node_mapping
        data.node_list = node_list
        data.node_type_map = node_idx_to_type
        data.edge_type_map = {idx: e_type for e_type, idx in edge_type_to_idx.items()}

        data.original_feature_dim = x.size(1) - target_embed_dim
        data.embedding_dim = target_embed_dim

        self.logger.info(f"Processed graph: {len(node_list)} nodes, {edge_index.size(1)} edges, "
                         f"{len(edge_type_to_idx)} edge types, {len(node_type_to_idx)} node types")

        return data, node_idx_to_type

    def add_user_ratings(self, data: Data, ratings_df: pd.DataFrame,
                         movie_mapping: Dict[str, str],
                         user2idx: Dict[str, int]) -> Data:
        """
        Add user-movie ratings as additional edges/data
        """
        user_items = []
        rating_list = []

        for _, row in ratings_df.iterrows():
            user_id = str(row["userId"])
            movie_id = str(row["movieId"])
            rating = row["rating"]

            if movie_id in movie_mapping:
                graph_movie_id = movie_mapping[movie_id]
                if graph_movie_id in data.node_mapping:
                    movie_idx = data.node_mapping[graph_movie_id]
                    if user_id in user2idx:
                        u_idx = user2idx[user_id]
                        user_items.append([u_idx, movie_idx])
                        rating_list.append(rating)

        if not user_items:
            self.logger.warning("No user-movie ratings found with given mappings")
            return data

        data.user_items = torch.tensor(user_items, dtype=torch.long)
        data.ratings = torch.tensor(rating_list, dtype=torch.float)

        self.logger.info(f"Added {len(rating_list)} user-movie ratings to data")
        return data

    def split_data(self, data: Data, val_ratio=0.1, test_ratio=0.1,
                  stratify_by_user=True, min_user_ratings=3):
        """
        Split data into train/val/test with enhanced stratification
        """
        if not hasattr(data, "user_items"):
            self.logger.warning("No ratings in data, returning single dataset only")
            return data, None, None

        user_items = data.user_items.numpy()
        ratings = data.ratings.numpy()

        if stratify_by_user and len(user_items) > 0:
            user_to_items = {}
            for i, (user, item) in enumerate(user_items):
                if user not in user_to_items:
                    user_to_items[user] = []
                user_to_items[user].append((i, item, ratings[i]))

            train_idx, val_idx, test_idx = [], [], []
            for user, items in user_to_items.items():
                if len(items) < min_user_ratings:
                    train_idx.extend([i for i, _, _ in items])
                    continue
                random.shuffle(items)
                n_test = max(1, int(len(items) * test_ratio))
                n_val = max(1, int(len(items) * val_ratio))
                n_train = len(items) - n_test - n_val
                if n_train <= 0:
                    n_train = 1
                    if n_val > 1:
                        n_val -= 1
                    else:
                        n_test -= 1
                train_idx.extend([i for i, _, _ in items[:n_train]])
                val_idx.extend([i for i, _, _ in items[n_train:n_train+n_val]])
                test_idx.extend([i for i, _, _ in items[n_train+n_val:]])
        else:
            n = len(user_items)
            indices = list(range(n))
            random.shuffle(indices)
            test_size = int(test_ratio * n)
            val_size = int(val_ratio * n)
            train_size = n - test_size - val_size

            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)

        def subset_data(base, subset_idx):
            d = Data()
            for key in base.keys():
                if key not in ['user_items', 'ratings']:
                    d[key] = base[key]
            d.user_items = base.user_items[subset_idx]
            d.ratings = base.ratings[subset_idx]
            return d

        train_data = subset_data(data, train_idx)
        val_data = subset_data(data, val_idx)
        test_data = subset_data(data, test_idx)

        self.logger.info(f"Split data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return train_data, val_data, test_data

###############################################################################
#                       Hyperbolic Embedding Layer                            #
###############################################################################

class HyperbolicEmbedding(nn.Module):
    """Hyperbolic embedding layer for hierarchical data"""

    def __init__(self, num_embeddings, embedding_dim, c=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.001, 0.001)

    def exponential_map(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-10)
        c_tensor = torch.tensor(self.c, device=x.device, dtype=x.dtype)
        exp_map = torch.tanh(torch.sqrt(c_tensor) * norm) * x / (torch.sqrt(c_tensor) * norm)
        mask = (norm < 1e-7).expand_as(exp_map)
        return torch.where(mask, x, exp_map)

    def forward(self, indices):
        tangent_emb = self.weight[indices]
        return self.exponential_map(tangent_emb)

    def distance(self, x, y):
        sqrt_c = torch.sqrt(self.c)
        x_norm = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm = torch.sum(y * y, dim=-1, keepdim=True)
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True) * 2
        alpha = 1 - self.c * x_norm
        beta = 1 - self.c * y_norm
        gamma = 1 - self.c * xy_inner + self.c**2 * x_norm * y_norm
        distance = torch.acosh(gamma / (alpha * beta).sqrt() + 1e-8) / sqrt_c
        return distance

###############################################################################
#                     Heterogeneous Message Passing                           #
###############################################################################

class HeterogeneousMessagePassing(MessagePassing):
    """Message passing layer with separate processing by edge type"""

    def __init__(self, in_dim, out_dim, num_edge_types, aggr="add",
                 self_loops=True, bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.linear_by_type = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_edge_types)
        ])
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.self_loops = self_loops
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.linear_by_type:
            lin.reset_parameters()
        self.self_loop.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type):
        if self.self_loops:
            out = self.self_loop(x)
        else:
            out = torch.zeros((x.size(0), self.self_loop.out_features), device=x.device, dtype=x.dtype)
        out = out + self.propagate(edge_index, x=x, edge_type=edge_type)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, edge_type):
        messages = []
        for i, lin in enumerate(self.linear_by_type):
            mask = (edge_type == i)
            if not mask.any():
                continue
            transformed = lin(x_j[mask])
            typed_message = torch.zeros((edge_type.size(0), lin.out_features),
                                        device=x_j.device, dtype=x_j.dtype)
            typed_message[mask] = transformed
            messages.append(typed_message)
        if not messages:
            return torch.zeros((edge_type.size(0), self.linear_by_type[0].out_features),
                               device=x_j.device, dtype=x_j.dtype)
        return torch.stack(messages).sum(dim=0)

class HeterogeneousGATConv(MessagePassing):
    """Graph attention with separate processing by edge type"""

    def __init__(self, in_dim, out_dim, num_edge_types, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=True, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_by_type = nn.ModuleList([
            nn.Linear(in_dim, heads * out_dim, bias=False) for _ in range(num_edge_types)
        ])
        self.lin_self = nn.Linear(in_dim, heads * out_dim, bias=False)

        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_dim))
        self.norm = nn.LayerNorm(out_dim * heads if concat else out_dim)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.edge_dim = None
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lin_by_type:
            lin.reset_parameters()
        self.lin_self.reset_parameters()
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, edge_attr=None, return_attention_weights=False):
        x_self = self.lin_self(x).view(-1, self.heads, self.out_dim)
        out = x_self
        all_attention_weights = {}

        for i, lin in enumerate(self.lin_by_type):
            mask = (edge_type == i)
            if not mask.any():
                continue
            edge_index_i = edge_index[:, mask]
            x_i = lin(x).view(-1, self.heads, self.out_dim)
            if edge_attr is not None:
                edge_attr_i = edge_attr[mask]
                out_i, att_weights_i = self._edge_type_forward(
                    x_i, edge_index_i, edge_attr_i, return_attention_weights
                )
            else:
                out_i, att_weights_i = self._edge_type_forward(
                    x_i, edge_index_i, None, return_attention_weights
                )
            out = out + out_i
            if return_attention_weights:
                all_attention_weights[i] = (edge_index_i, att_weights_i)

        if self.concat:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        out = self.norm(out)

        if return_attention_weights:
            return out, all_attention_weights
        else:
            return out

    def _edge_type_forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        if edge_index.numel() == 0:
            return torch.zeros_like(x), None
        out, attention_weights = self.propagate(
            edge_index, x=x, edge_attr=edge_attr,
            return_attention_weights=True
        )
        if return_attention_weights:
            return out, attention_weights
        else:
            return out, None

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i, return_attention_weights):
        x_j = x_j.view(-1, self.heads, self.out_dim)
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)

        if return_attention_weights:
            return out, alpha
        else:
            return out

    def aggregate(self, inputs, index, dim_size=None, ptr=None):
        if isinstance(inputs, tuple):
            out, alpha = inputs
            return super().aggregate(out, index, dim_size=dim_size, ptr=ptr), alpha
        else:
            return super().aggregate(inputs, index, dim_size=dim_size, ptr=ptr)

###############################################################################
#                     Advanced Neural ODE Components                          #
###############################################################################

class AdvancedODEFunction(nn.Module):
    """Advanced ODE function with multi-scale feature learning and heterogeneous support"""

    def __init__(self, hidden_dim, edge_index, edge_type, num_edge_types,
                 node_type=None, num_node_types=None,
                 dropout=0.1, attention_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_edge_types = num_edge_types
        self.node_type = node_type
        self.num_node_types = num_node_types

        self.het_conv = HeterogeneousGATConv(
            hidden_dim,
            hidden_dim // attention_heads,
            num_edge_types,
            heads=attention_heads,
            concat=True,
            dropout=dropout
        )
        self.trans_conv = TransformerConv(
            hidden_dim,
            hidden_dim,
            heads=attention_heads,
            dropout=dropout,
            concat=False
        )
        self.skip_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim)

        if node_type is not None and num_node_types is not None:
            self.use_node_types = True
            self.node_type_embeddings = nn.Parameter(
                torch.Tensor(num_node_types, hidden_dim // 4)
            )
            self.node_type_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
            nn.init.xavier_uniform_(self.node_type_embeddings)
        else:
            self.use_node_types = False

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, x):
        identity = x
        h1 = self.het_conv(x, self.edge_index, self.edge_type)
        h1 = self.norm1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        h2 = self.trans_conv(h1, self.edge_index)
        if self.use_node_types:
            node_type_emb = self.node_type_embeddings[self.node_type]
            h2 = torch.cat([h2, node_type_emb], dim=1)
            h2 = self.node_type_proj(h2)

        h2 = self.norm2(h2)
        gate = self.skip_gate(identity)
        skip = self.skip_proj(identity)
        h_out = gate * h2 + (1 - gate) * skip
        h_out = self.act(h_out)
        h_out = self.dropout(h_out)
        return h_out

class AdvancedNeuralODEBlock(nn.Module):
    """Neural ODE block with edge type awareness and multi-level integration"""

    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_type,
                 num_edge_types, node_type=None, num_node_types=None,
                 time_steps=16, dropout=0.1, adaptive_steps=True):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.odefunc = AdvancedODEFunction(
            hidden_dim=hidden_dim,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=num_edge_types,
            node_type=node_type,
            num_node_types=num_node_types,
            dropout=dropout
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.merge_gate = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.Sigmoid()
        )

        self.time_steps = time_steps
        self.adaptive_steps = adaptive_steps
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = nn.Parameter(torch.randn(time_steps))

    def forward(self, x):
        h = self.in_proj(x)
        t_space = torch.linspace(0, 1, self.time_steps).to(x.device)

        try:
            if self.adaptive_steps:
                traj = odeint(
                    self.odefunc, h, t_space,
                    method='dopri5',
                    rtol=1e-4, atol=1e-5,
                    adjoint_options={'norm': 'seminorm'}
                )
            else:
                traj = odeint(
                    self.odefunc, h, t_space,
                    method='rk4'
                )
        except Exception as e:
            logger.warning(f"ODE solver failed: {str(e)}. Falling back to Euler method.")
            dt = 1.0/(self.time_steps-1)
            traj_list = [h]
            cur = h
            for _ in range(self.time_steps-1):
                dx = self.odefunc(0, cur)
                cur = cur + dx*dt
                traj_list.append(cur)
            traj = torch.stack(traj_list, dim=0)

        attn = F.softmax(self.attn_weights, dim=0).view(-1, 1, 1)
        h_weighted = (traj * attn).sum(dim=0)

        ode_out = self.out_proj(h_weighted)
        skip_out = self.skip_proj(x)
        gate_input = torch.cat([h_weighted, x], dim=1)
        gate = self.merge_gate(gate_input)
        out = gate * ode_out + (1 - gate) * skip_out
        return out

###############################################################################
#                    Advanced Quantum-Inspired Blocks                         #
###############################################################################

class PositionalEncodingProjection(nn.Module):
    """Positional encoding for quantum phase shifts"""

    def __init__(self, hidden_dim, max_seq_len=100):
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, positions):
        positions = positions.clamp(max=self.pe.size(0) - 1)
        pos_enc = self.pe[positions]
        return self.proj(x + pos_enc)

class MultiHeadQuantumLayer(nn.Module):
    """Multi-head quantum-inspired layer with advanced phase control"""

    def __init__(self, hidden_dim, heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.phase_shifts = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)

        phase = self.phase_shifts.view(1, 1, self.heads, self.head_dim)
        q_complex = torch.complex(
            q * torch.cos(phase),
            q * torch.sin(phase)
        )
        k_complex = torch.complex(
            k * torch.cos(phase),
            k * torch.sin(phase)
        )

        q_complex = q_complex.transpose(1, 2)
        k_complex = k_complex.transpose(1, 2)
        v = v.transpose(1, 2)

        # ---------------- Here is the big matmul that can cause OOM ----------------
        scores = torch.matmul(q_complex, k_complex.transpose(-2, -1).conj())
        # ---------------------------------------------------------------------------

        scores = torch.abs(scores) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(output)
        output = self.norm(output)
        return output

class EnhancedQuantumBlock(nn.Module):
    """Enhanced quantum-inspired block for graph data"""

    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, edge_type,
                 num_edge_types, node_type=None, num_node_types=None,
                 n_layers=8, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.q_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.q_layers.append(nn.ModuleList([
                HeterogeneousGATConv(
                    hidden_dim,
                    hidden_dim // 4,
                    num_edge_types,
                    heads=4,
                    concat=True,
                    dropout=dropout
                ),
                MultiHeadQuantumLayer(
                    hidden_dim,
                    heads=8,
                    dropout=dropout
                ),
                nn.LayerNorm(hidden_dim)
            ]))

        self.phase_shifts = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * 0.02) for _ in range(n_layers)
        ])

        self.final_conv = HeterogeneousGATConv(
            hidden_dim,
            hidden_dim // 4,
            num_edge_types,
            heads=4,
            concat=True,
            dropout=dropout
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, out_dim)
        )

        if node_type is not None and num_node_types is not None:
            self.use_node_types = True
            self.node_type_embeddings = nn.Parameter(
                torch.Tensor(num_node_types, hidden_dim // 4)
            )
            self.node_type_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
            nn.init.xavier_uniform_(self.node_type_embeddings)
        else:
            self.use_node_types = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.in_proj(x)
        batch_size = 1
        seq_len = h.size(0)

        for i, (gat, quantum, norm) in enumerate(self.q_layers):
            h_orig = h
            h_gat = gat(h, self.edge_index, self.edge_type)
            h_q = h_gat.view(batch_size, seq_len, -1)
            h_q = quantum(h_q)
            h_q = h_q.view(seq_len, -1)

            phase = self.phase_shifts[i]
            h_complex = torch.complex(
                h_q * torch.cos(phase),
                h_q * torch.sin(phase)
            )
            h_phase = torch.abs(h_complex)
            h = norm(h_phase + h_orig)
            h = self.dropout(h)

        h = self.final_conv(h, self.edge_index, self.edge_type)

        if self.use_node_types:
            node_type_emb = self.node_type_embeddings[self.node_type]
            h = torch.cat([h, node_type_emb], dim=1)
            h = self.node_type_proj(h)

        out = self.out_proj(h)
        return out

###############################################################################
#                      Advanced Hybrid Model Architecture                     #
###############################################################################

class AdvancedHybridModel(nn.Module):
    """
    Advanced hybrid architecture combining Neural ODE and Quantum GNN with
    hyperbolic embeddings and graph structure learning
    """

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 edge_index,
                 edge_type,
                 num_edge_types,
                 node_type=None,
                 num_node_types=None,
                 num_communities=None,
                 time_steps=16,
                 q_layers=8,
                 dropout=0.15,
                 use_hyperbolic=True,
                 embedding_dim=64):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_hyperbolic = use_hyperbolic

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.node_type = node_type

        if use_hyperbolic:
            num_nodes = edge_index.max().item() + 1
            self.hyperbolic_embedding = HyperbolicEmbedding(
                num_embeddings=num_nodes,
                embedding_dim=embedding_dim,
                c=1.0
            )
            self.hyp_proj = nn.Linear(embedding_dim, hidden_dim)
            self.in_proj = nn.Linear(in_dim + hidden_dim, hidden_dim)
        else:
            self.in_proj = nn.Linear(in_dim, hidden_dim)

        if num_communities is not None:
            self.use_communities = True
            self.community_embedding = nn.Embedding(num_communities, hidden_dim // 4)
            self.community_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        else:
            self.use_communities = False

        self.ode_block = AdvancedNeuralODEBlock(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=num_edge_types,
            node_type=node_type,
            num_node_types=num_node_types,
            time_steps=time_steps,
            dropout=dropout
        )

        self.quantum_block = EnhancedQuantumBlock(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=num_edge_types,
            node_type=node_type,
            num_node_types=num_node_types,
            n_layers=q_layers,
            dropout=dropout
        )

        self.skip_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, node_indices=None, community_ids=None):
        if self.use_hyperbolic:
            if node_indices is None:
                node_indices = torch.arange(x.size(0), device=x.device)
            hyp_emb = self.hyperbolic_embedding(node_indices)
            hyp_proj = self.hyp_proj(hyp_emb)
            h_in = torch.cat([x, hyp_proj], dim=1)
            h = self.in_proj(h_in)
        else:
            h = self.in_proj(x)

        if self.use_communities and community_ids is not None:
            comm_emb = self.community_embedding(community_ids)
            h = torch.cat([h, comm_emb], dim=1)
            h = self.community_proj(h)

        h_ode = self.ode_block(h)
        h_q = self.quantum_block(h_ode)
        h_skip = self.skip_proj(h)

        gate_input = torch.cat([h_ode, h], dim=1)
        gates = self.gate_net(gate_input)
        output = gates[:, 0].unsqueeze(1) * h_q + gates[:, 1].unsqueeze(1) * h_skip

        return output

###############################################################################
#                      Advanced Recommender Architecture                      #
###############################################################################

class AdvancedHybridRecommender(nn.Module):
    """
    Advanced recommender with hybrid graph neural networks and sophisticated
    rating prediction layers.
    """

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 edge_index,
                 edge_type,
                 num_edge_types,
                 node_type=None,
                 num_node_types=None,
                 num_communities=None,
                 time_steps=16,
                 q_layers=8,
                 dropout=0.15,
                 use_hyperbolic=True,
                 embedding_dim=64,
                 rating_mode="regression",
                 rating_min=-1.0,
                 rating_max=1.0):
        super().__init__()

        self.hybrid_model = AdvancedHybridModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=num_edge_types,
            node_type=node_type,
            num_node_types=num_node_types,
            num_communities=num_communities,
            time_steps=time_steps,
            q_layers=q_layers,
            dropout=dropout,
            use_hyperbolic=use_hyperbolic,
            embedding_dim=embedding_dim
        )

        self.rating_mode = rating_mode
        self.rating_min = rating_min
        self.rating_max = rating_max

        self.rating_predictor = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        if rating_mode == "regression":
            self.rating_head = nn.Linear(hidden_dim // 4, 1)
        elif rating_mode == "classification":
            self.rating_head = nn.Linear(hidden_dim // 4, 3)
        else:
            raise ValueError(f"Unsupported rating mode: {rating_mode}")

    def forward(self, x, node_indices=None, community_ids=None):
        return self.hybrid_model(x, node_indices, community_ids)

    def predict_rating(self, node_embeddings, user_idx, movie_idx):
        user_emb = node_embeddings[user_idx]
        movie_emb = node_embeddings[movie_idx]
        combined = torch.cat([user_emb, movie_emb], dim=1)

        features = self.rating_predictor(combined)
        raw_output = self.rating_head(features)

        if self.rating_mode == "regression":
            if self.rating_min == -1.0 and self.rating_max == 1.0:
                return torch.tanh(raw_output).squeeze(1)
            else:
                range_width = self.rating_max - self.rating_min
                return (torch.sigmoid(raw_output) * range_width + self.rating_min).squeeze(1)
        else:
            return raw_output

    def forward_rating(self, x, user_idx, movie_idx, node_indices=None, community_ids=None):
        node_embs = self.forward(x, node_indices, community_ids)
        return self.predict_rating(node_embs, user_idx, movie_idx)

###############################################################################
#                     Advanced Training with Curriculum                       #
###############################################################################

class AdvancedRecommenderTrainer:
    """
    Enhanced trainer with curriculum learning, mixed loss functions,
    learning rate scheduling, and early stopping.
    """

    def __init__(self,
                 model: AdvancedHybridRecommender,
                 learning_rate=3e-4,
                 device=None,
                 rating_mode="regression"):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.rating_mode = rating_mode

        # ------------------- Mixed Precision and GradScaler -------------------
        torch.set_float32_matmul_precision('medium')  # helps reduce memory usage in large matmuls
        self.scaler = GradScaler()                    # for AMP gradient scaling
        # ----------------------------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        self.logger = logging.getLogger("advanced_hybrid_model.trainer")
        self.best_model_path = None

    def _get_loss_function(self, epoch, total_epochs):
        if self.rating_mode == "regression":
            progress = epoch / total_epochs
            def combined_loss(pred, target):
                mse_loss = F.mse_loss(pred, target)
                mae_loss = F.l1_loss(pred, target)
                huber_loss = F.smooth_l1_loss(pred, target, beta=0.1)
                mse_weight = max(0.4, 1.0 - progress)
                mae_weight = min(0.4, progress)
                huber_weight = 0.2
                return (mse_weight * mse_loss +
                        mae_weight * mae_loss +
                        huber_weight * huber_loss)
            return combined_loss
        else:
            return F.cross_entropy

    def _get_rating_metrics(self, predictions, targets):
        if self.rating_mode == "regression":
            mse = F.mse_loss(predictions, targets).item()
            rmse = math.sqrt(mse)
            mae = F.l1_loss(predictions, targets).item()
            return {"mse": mse, "rmse": rmse, "mae": mae}
        else:
            preds = torch.argmax(predictions, dim=1)
            targets = (targets + 1).long()
            correct = (preds == targets).sum().item()
            accuracy = correct / targets.size(0)
            from sklearn.metrics import f1_score
            f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='macro')
            return {"accuracy": accuracy, "f1": f1}

    def _get_curriculum_weights(self, data, epoch, total_epochs):
        user_items = data.user_items
        ratings = data.ratings
        if self.rating_mode == "regression":
            rating_difficulty = 1.0 - torch.abs(ratings)
        else:
            rating_difficulty = torch.zeros_like(ratings)
            rating_difficulty[ratings == 0] = 1.0

        progress = min(1.0, epoch / (total_epochs * 0.5))
        weights = 1.0 - (1.0 - progress) * rating_difficulty * 0.8
        return weights

    def train(self,
              data: Data,
              val_data: Optional[Data]=None,
              epochs=70,
              batch_size=128,
              patience=12,
              curriculum_learning=True,
              label_smoothing=0.1,
              clip_grad_norm=1.0):
        if not hasattr(data, "user_items"):
            raise ValueError("No user_items in data - can't train a recommender!")

        self.model.train()

        x = data.x.to(self.device)
        user_items = data.user_items.to(self.device)
        ratings = data.ratings.to(self.device)
        node_indices = torch.arange(x.size(0), device=self.device)
        community_ids = data.node_community.to(self.device) if hasattr(data, "node_community") else None

        history = {
            "train_loss": [],
            "train_rmse": [],
            "train_mae": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "learning_rate": []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        epoch_bar = tqdm(range(1, epochs+1), desc="Training")

        for ep in epoch_bar:
            loss_fn = self._get_loss_function(ep, epochs)
            if curriculum_learning:
                sample_weights = self._get_curriculum_weights(data, ep, epochs).to(self.device)
            else:
                sample_weights = None

            current_lr = self.optimizer.param_groups[0]['lr']
            history["learning_rate"].append(current_lr)

            if sample_weights is not None:
                weights_np = sample_weights.cpu().numpy()
                weights_np = np.maximum(weights_np, 1e-6)
                weights_np = weights_np / weights_np.sum()
                n_samples = len(user_items)
                idx = np.random.choice(n_samples, size=n_samples, p=weights_np)
                idx = torch.tensor(idx, device=self.device)
            else:
                idx = torch.randperm(len(user_items), device=self.device)

            ui_shuf = user_items[idx]
            r_shuf = ratings[idx]

            epoch_loss = 0.0
            all_preds = []
            all_targets = []
            n_batches = (len(ui_shuf) + batch_size - 1)//batch_size
            batch_bar = tqdm(range(n_batches), desc="Batches", leave=False)

            for b in batch_bar:
                start = b*batch_size
                end = min(start+batch_size, len(ui_shuf))
                b_ui = ui_shuf[start:end]
                b_r = r_shuf[start:end]

                self.optimizer.zero_grad()

                # ---------------------- AMP autocast block ----------------------
                with autocast():
                    u_idx = b_ui[:, 0]
                    m_idx = b_ui[:, 1]
                    preds = self.model.forward_rating(x, u_idx, m_idx, node_indices, community_ids)

                    if self.rating_mode == "classification":
                        targets = (b_r + 1).long()
                        loss = F.cross_entropy(preds, targets, label_smoothing=label_smoothing)
                    else:
                        loss = loss_fn(preds, b_r)
                # --------------------------------------------------------------

                self.scaler.scale(loss).backward()

                if clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.rating_mode == "regression":
                    batch_metrics = self._get_rating_metrics(preds.detach(), b_r)
                    batch_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'rmse': f'{batch_metrics["rmse"]:.4f}',
                        'mae': f'{batch_metrics["mae"]:.4f}'
                    })
                    epoch_loss += loss.item() * len(b_r)
                    all_preds.append(preds.detach())
                    all_targets.append(b_r)
                else:
                    batch_metrics = self._get_rating_metrics(preds.detach(), b_r)
                    batch_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{batch_metrics["accuracy"]:.4f}',
                        'f1': f'{batch_metrics["f1"]:.4f}'
                    })
                    epoch_loss += loss.item() * len(b_r)
                    all_preds.append(preds.detach())
                    all_targets.append(b_r)

            epoch_loss /= len(user_items)
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            epoch_metrics = self._get_rating_metrics(all_preds, all_targets)

            if self.rating_mode == "regression":
                history["train_loss"].append(epoch_loss)
                history["train_rmse"].append(epoch_metrics["rmse"])
                history["train_mae"].append(epoch_metrics["mae"])
                train_metrics_str = (
                    f'loss={epoch_loss:.4f}, '
                    f'rmse={epoch_metrics["rmse"]:.4f}, '
                    f'mae={epoch_metrics["mae"]:.4f}'
                )
            else:
                history["train_loss"].append(epoch_loss)
                history["train_rmse"].append(0)
                history["train_mae"].append(0)
                train_metrics_str = (
                    f'loss={epoch_loss:.4f}, '
                    f'acc={epoch_metrics["accuracy"]:.4f}, '
                    f'f1={epoch_metrics["f1"]:.4f}'
                )

            val_metrics = {}
            val_metrics_str = ""

            if val_data and hasattr(val_data, "user_items"):
                val_loss, val_metrics = self.evaluate(val_data)
                if self.rating_mode == "regression":
                    history["val_loss"].append(val_loss)
                    history["val_rmse"].append(val_metrics["rmse"])
                    history["val_mae"].append(val_metrics["mae"])
                    val_metrics_str = (
                        f'val_loss={val_loss:.4f}, '
                        f'val_rmse={val_metrics["rmse"]:.4f}, '
                        f'val_mae={val_metrics["mae"]:.4f}'
                    )
                else:
                    history["val_loss"].append(val_loss)
                    history["val_rmse"].append(0)
                    history["val_mae"].append(0)
                    val_metrics_str = (
                        f'val_loss={val_loss:.4f}, '
                        f'val_acc={val_metrics["accuracy"]:.4f}, '
                        f'val_f1={val_metrics["f1"]:.4f}'
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {
                        'epoch': ep,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }
                    self.best_model_path = 'best_advanced_model.pt'
                    torch.save(best_model_state, self.best_model_path)
                    epoch_bar.set_postfix_str(f"{train_metrics_str}, {val_metrics_str}, best model saved")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {ep}")
                        epoch_bar.set_postfix_str(f"{train_metrics_str}, {val_metrics_str}, early stopped")
                        break
                    epoch_bar.set_postfix_str(f"{train_metrics_str}, {val_metrics_str}, patience {patience_counter}/{patience}")
            else:
                epoch_bar.set_postfix_str(f"{train_metrics_str}, lr={current_lr:.6f}")

            self.scheduler.step()
            self.logger.info(
                f"Epoch {ep}/{epochs}, {train_metrics_str}, {val_metrics_str}, lr={current_lr:.6f}"
            )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state["model_state_dict"])
            self.logger.info(
                f"Loaded best model from epoch {best_model_state['epoch']} "
                f"with val_loss={best_model_state['val_loss']:.4f}"
            )

        return history

    def evaluate(self, data: Data) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        x = data.x.to(self.device)
        user_items = data.user_items.to(self.device)
        ratings = data.ratings.to(self.device)
        node_indices = torch.arange(x.size(0), device=self.device)
        community_ids = data.node_community.to(self.device) if hasattr(data, "node_community") else None

        with torch.no_grad():
            # --------------- AMP autocast for evaluation ---------------
            with autocast():
                preds = self.model.forward_rating(
                    x, user_items[:, 0], user_items[:, 1], node_indices, community_ids
                )
            # -----------------------------------------------------------

            if self.rating_mode == "regression":
                loss = F.mse_loss(preds, ratings)
            else:
                targets = (ratings + 1).long()
                loss = F.cross_entropy(preds, targets)

            metrics = self._get_rating_metrics(preds, ratings)

        return loss.item(), metrics

    def get_node_embeddings(self, data: Data) -> Dict[str, np.ndarray]:
        self.model.eval()
        x = data.x.to(self.device)
        node_indices = torch.arange(x.size(0), device=self.device)
        community_ids = data.node_community.to(self.device) if hasattr(data, "node_community") else None

        with torch.no_grad():
            # --------------- AMP autocast for embeddings ---------------
            with autocast():
                embeddings = self.model(x, node_indices, community_ids).cpu().float().numpy()
            # -----------------------------------------------------------

        emb_dict = {}
        for i, node_id in enumerate(data.node_list):
            emb_dict[node_id] = embeddings[i]
        return emb_dict

    def save(self, path: str, data: Optional[Data]=None, embed_path=None):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "rating_mode": self.rating_mode,
            "model_config": {
                "in_dim": self.model.hybrid_model.in_dim,
                "hidden_dim": self.model.hybrid_model.hidden_dim,
                "out_dim": self.model.hybrid_model.out_dim,
                "use_hyperbolic": self.model.hybrid_model.use_hyperbolic
            }
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved model to {path}")

        if embed_path and data is not None:
            emb_dict = self.get_node_embeddings(data)
            with open(embed_path, "wb") as f:
                pickle.dump(emb_dict, f)
            self.logger.info(f"Saved node embeddings to {embed_path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "rating_mode" in checkpoint:
            self.rating_mode = checkpoint["rating_mode"]
        self.logger.info(f"Loaded model from {path}")

###############################################################################
#              Utility to load & map movies and ratings                       #
###############################################################################

def map_movie_ids_to_graph(movies_df: pd.DataFrame, nx_graph: nx.Graph) -> Dict[str, str]:
    logger.info("Building mapping from movie IDs to graph nodes")
    mapping = {}
    graph_movies = {}
    graph_movie_ids = {}

    for n, d in nx_graph.nodes(data=True):
        if d.get("node_type", "").lower() == "movie":
            title = str(d.get("title", "")).lower()
            if title:
                graph_movies[title] = n
            imdb_id = str(d.get("imdb_id", "")).lower()
            if imdb_id:
                graph_movie_ids[imdb_id] = n

    matched = 0
    for _, row in movies_df.iterrows():
        ml_id = str(row.get("movie_id", row.get("movieId", "")))
        if not ml_id:
            continue
        found = False
        for id_col in ["imdb_id", "tt_id", "id"]:
            if id_col in row and row[id_col]:
                movie_id = str(row[id_col]).lower()
                if movie_id in graph_movie_ids:
                    mapping[ml_id] = graph_movie_ids[movie_id]
                    found = True
                    break
        if found:
            matched += 1
            continue
        ml_title = str(row.get("title", row.get("name", ""))).lower()
        if ml_title in graph_movies:
            mapping[ml_id] = graph_movies[ml_title]
            matched += 1
            continue
        best_match = None
        best_score = 0
        for graph_title, node_id in graph_movies.items():
            if abs(len(ml_title) - len(graph_title)) > min(len(ml_title), len(graph_title)) * 0.5:
                continue
            ml_tokens = set(ml_title.split())
            graph_tokens = set(graph_title.split())
            if ml_tokens and graph_tokens:
                intersection = ml_tokens.intersection(graph_tokens)
                union = ml_tokens.union(graph_tokens)
                score = len(intersection) / len(union)
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = node_id
        if best_match:
            mapping[ml_id] = best_match
            matched += 1

    logger.info(f"Mapped {matched}/{len(movies_df)} movies to graph nodes")
    return mapping

def load_ratings_json(ratings_path: str) -> pd.DataFrame:
    with open(ratings_path, 'r') as f:
        data = json.load(f)
    records = []
    if isinstance(data, list) and len(data) > 0 and "_id" in data[0] and "rated" in data[0]:
        for user_block in data:
            user_id = user_block["_id"]
            rated = user_block.get("rated", {})
            for movie_id, rating_info in rated.items():
                if movie_id.lower() == "submit":
                    continue
                if isinstance(rating_info, list) and len(rating_info) > 0:
                    try:
                        rating = float(rating_info[0])
                        records.append({
                            "userId": user_id,
                            "movieId": movie_id,
                            "rating": rating
                        })
                    except (ValueError, TypeError):
                        pass
    elif isinstance(data, list) and len(data) > 0 and "userId" in data[0] and "movieId" in data[0]:
        for item in data:
            try:
                records.append({
                    "userId": item["userId"],
                    "movieId": item["movieId"],
                    "rating": float(item["rating"])
                })
            except (KeyError, ValueError, TypeError):
                pass
    elif isinstance(data, dict) and "users" in data:
        users = data["users"]
        for user_id, user_data in users.items():
            if "ratings" in user_data:
                ratings = user_data["ratings"]
                for movie_id, rating in ratings.items():
                    try:
                        records.append({
                            "userId": user_id,
                            "movieId": movie_id,
                            "rating": float(rating)
                        })
                    except (ValueError, TypeError):
                        pass
    if not records and isinstance(data, dict):
        for user_id, ratings in data.items():
            if isinstance(ratings, dict):
                for movie_id, rating in ratings.items():
                    try:
                        records.append({
                            "userId": user_id,
                            "movieId": movie_id,
                            "rating": float(rating)
                        })
                    except (ValueError, TypeError):
                        pass
    logger.info(f"Loaded {len(records)} ratings from JSON file")
    return pd.DataFrame(records)

###############################################################################
#                     Main Training Pipeline                                  #
###############################################################################

def train_advanced_hybrid_model(
    graphml_path: str,
    ratings_path: Optional[str] = None,
    movies_path: Optional[str] = None,
    existing_embeddings_path: Optional[str] = None,
    output_dir: str = "./output_advanced",
    epochs=100,
    hidden_dim=512,
    out_dim=256,
    time_steps=16,
    q_layers=8,
    dropout=0.15,
    lr=1e-4,
    batch_size=256,
    patience=15,
    use_hyperbolic=True,
    curriculum_learning=True,
    rating_mode="regression",
    rating_min=-1.0,
    rating_max=1.0,
    device=None,
    seed=42) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    processor = AdvancedGraphDataProcessor(device=device)

    existing_embeddings = None
    if existing_embeddings_path and os.path.exists(existing_embeddings_path):
        logger.info(f"Loading existing embeddings from {existing_embeddings_path}")
        try:
            with open(existing_embeddings_path, "rb") as f:
                existing_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(existing_embeddings)} existing embeddings")
        except Exception as e:
            logger.warning(f"Error loading embeddings: {str(e)}")

    nx_graph = processor.load_graphml(graphml_path)
    logger.info(f"Loaded graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")

    data, node_type_map = processor.process_nx_graph(nx_graph, existing_embeddings)

    num_users = 0
    train_data, val_data, test_data = data, None, None

    if ratings_path and movies_path:
        if ratings_path.endswith('.json'):
            ratings_df = load_ratings_json(ratings_path)
        else:
            ratings_df = pd.read_csv(ratings_path)

        movie_mapping = {}
        movies_df = None
        if movies_path.endswith('.json'):
            with open(movies_path, 'r') as f:
                movie_mapping = json.load(f)
        else:
            movies_df = pd.read_csv(movies_path)
            if 'movie_id' in movies_df.columns and 'node_id' in movies_df.columns:
                movie_mapping = dict(zip(movies_df['movie_id'], movies_df['node_id']))
            elif 'movie_id' in movies_df.columns and 'title' in movies_df.columns:
                for _, row in movies_df.iterrows():
                    movie_id = str(row['movie_id'])
                    title = row['title']
                    title_normalized = title.lower().replace(' ', '_')
                    node_id = f"movie_{title_normalized}"
                    movie_mapping[movie_id] = node_id

        unique_users = ratings_df['userId'].unique().tolist()
        user2idx = {u: i for i, u in enumerate(unique_users)}
        num_users = len(user2idx)

        data = processor.add_user_ratings(data, ratings_df, movie_mapping, user2idx)
        train_data, val_data, test_data = processor.split_data(
            data,
            val_ratio=0.1,
            test_ratio=0.1,
            stratify_by_user=True
        )
    else:
        logger.warning("No ratings or movies given, will skip training recommender model")

    in_dim = data.x.size(1)
    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device) if hasattr(data, "edge_type") else None
    node_type = data.node_type.to(device) if hasattr(data, "node_type") else None

    num_edge_types = edge_type.max().item() + 1 if edge_type is not None else 1
    num_node_types = node_type.max().item() + 1 if node_type is not None else None
    num_communities = data.node_community.max().item() + 1 if hasattr(data, "node_community") else None

    model = AdvancedHybridRecommender(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=num_edge_types,
        node_type=node_type,
        num_node_types=num_node_types,
        num_communities=num_communities,
        time_steps=time_steps,
        q_layers=q_layers,
        dropout=dropout,
        use_hyperbolic=use_hyperbolic,
        embedding_dim=64,
        rating_mode=rating_mode,
        rating_min=rating_min,
        rating_max=rating_max
    )

    logger.info(f"Initialized advanced hybrid model with {in_dim} input dim, "
                f"{hidden_dim} hidden dim, {out_dim} output dim")
    logger.info(f"Model has {num_edge_types} edge types" +
                (f", {num_node_types} node types" if num_node_types else "") +
                (f", {num_communities} communities" if num_communities else ""))

    trainer = AdvancedRecommenderTrainer(
        model=model,
        learning_rate=lr,
        device=device,
        rating_mode=rating_mode
    )

    train_history = None
    test_metrics = None
    train_time = 0

    if hasattr(train_data, "user_items"):
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        train_history = trainer.train(
            train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            curriculum_learning=curriculum_learning
        )

        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")

        if test_data and hasattr(test_data, "user_items"):
            logger.info("Evaluating on test set")
            test_loss, test_metrics = trainer.evaluate(test_data)
            if rating_mode == "regression":
                logger.info(f"Test metrics: MSE={test_metrics['mse']:.4f}, "
                            f"RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}")
            else:
                logger.info(f"Test metrics: Loss={test_loss:.4f}, "
                            f"Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}")
    else:
        logger.info("No user-item ratings, skipping training")

    model_path = os.path.join(output_dir, "advanced_hybrid_model.pt")
    embeddings_path = os.path.join(output_dir, "advanced_node_embeddings.pkl")
    trainer.save(model_path, data, embeddings_path)

    if train_history:
        history_path = os.path.join(output_dir, "advanced_train_history.json")
        serializable_history = {}
        for k, v in train_history.items():
            serializable_history[k] = [float(x) for x in v]
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

    summary = {
        "model_type": "advanced_hybrid_ode_quantum",
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "time_steps": time_steps,
        "q_layers": q_layers,
        "dropout": dropout,
        "use_hyperbolic": use_hyperbolic,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "learning_rate": lr,
        "rating_mode": rating_mode,
        "num_nodes": nx_graph.number_of_nodes(),
        "num_edges": nx_graph.number_of_edges(),
        "num_edge_types": num_edge_types,
        "train_time_sec": train_time,
        "model_path": model_path,
        "embeddings_path": embeddings_path
    }

    if test_metrics:
        summary.update({
            "test_metrics": test_metrics
        })

    summary_path = os.path.join(output_dir, "advanced_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved model, embeddings and summary to {output_dir}")

    return {
        "model": model,
        "trainer": trainer,
        "data": data,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_history": train_history,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "embeddings_path": embeddings_path,
        "summary_path": summary_path
    }

def run_advanced_model(
    graphml_path: str,
    output_dir: str = "./output_advanced",
    ratings_path: Optional[str] = None,
    movies_path: Optional[str] = None,
    existing_embeddings_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    default_config = {
        "epochs": 100,
        "hidden_dim": 512,
        "out_dim": 256,
        "time_steps": 16,
        "q_layers": 8,
        "dropout": 0.15,
        "lr": 1e-4,
        "batch_size": 256,
        "patience": 15,
        "use_hyperbolic": True,
        "curriculum_learning": True,
        "rating_mode": "regression",
        "rating_min": -1.0,
        "rating_max": 1.0,
        "seed": 42
    }

    if config:
        default_config.update(config)

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used, "
                    f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (GPU not available)")

    default_config["device"] = device
    results = train_advanced_hybrid_model(
        graphml_path=graphml_path,
        ratings_path=ratings_path,
        movies_path=movies_path,
        existing_embeddings_path=existing_embeddings_path,
        output_dir=output_dir,
        **default_config
    )

    end_time = time.time()
    duration = end_time - start_time
    results["total_duration"] = duration
    logger.info(f"Total execution time: {duration/60:.2f} minutes")

    if results.get("test_metrics"):
        test_metrics = results["test_metrics"]
        if default_config["rating_mode"] == "regression":
            logger.info(f"Final Test RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"Final Test MAE: {test_metrics.get('mae', 'N/A'):.4f}")
        else:
            logger.info(f"Final Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"Final Test F1: {test_metrics.get('f1', 'N/A'):.4f}")

    return results


# ------------------------------------------------------------------------------
# Example usage in a notebook or script (Colab-friendly):
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    graphml_path             = "/content/drive/MyDrive/uploadfile/ODE_RATING/text_bollywood_kg_neural.graphml"
    ratings_path             = "/content/drive/MyDrive/uploadfile/ODE_RATING/ratings_array.json"
    movies_path              = "/content/drive/MyDrive/uploadfile/ODE_RATING/movie_id_mapping.json"
    existing_embeddings_path = None
    output_dir               = "/content/drive/MyDrive/uploadfile/ODE_RATING/output_advanced"

    hidden_dim           = 64
    out_dim              = 32
    time_steps           = 4
    q_layers             = 4
    epochs               = 100
    batch_size           = 1
    lr                   = 1e-4
    dropout              = 0.15
    use_hyperbolic       = True
    curriculum_learning  = True
    rating_mode          = "regression"
    seed                 = 42

    config = {
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "time_steps": time_steps,
        "q_layers": q_layers,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "use_hyperbolic": use_hyperbolic,
        "curriculum_learning": curriculum_learning,
        "rating_mode": rating_mode,
        "seed": seed
    }

    run_advanced_model(
        graphml_path=graphml_path,
        output_dir=output_dir,
        ratings_path=ratings_path,
        movies_path=movies_path,
        existing_embeddings_path=existing_embeddings_path,
        config=config
    )
