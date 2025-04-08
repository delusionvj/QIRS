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
import asyncio
import aiohttp
import sqlite3
import traceback
import yaml
import warnings
import matplotlib.pyplot as plt
import hashlib
import sys
import math
import glob
import shutil
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, TransformerConv, HGTConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, negative_sampling
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torchdiffeq import odeint
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Iterator
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm.asyncio import tqdm as async_tqdm

# for suppressing warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# for loading movie tt id

def load_tt_mapping(mapping_json: str) -> Dict[str, str]:
    """Load movie ID mapping from tt IDs to movie names"""
    try:
        with open(mapping_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load movie mapping from {mapping_json}: {str(e)}")
        return {}
    
# logging setup

def setup_logging(log_dir="logs", level=logging.WARNING):
    """Set up logging configuration with detailed formatting"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bollywood_kg_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create specific loggers
    loggers = {
        "data_collection": logging.getLogger("data_collection"),
        "graph_builder": logging.getLogger("graph_builder"),
        "embeddings": logging.getLogger("embeddings"),
        "community": logging.getLogger("community"),
        "hypergraph": logging.getLogger("hypergraph"),
        "temporal": logging.getLogger("temporal"),
        "metrics": logging.getLogger("metrics"),
        "api": logging.getLogger("api"),
        "user_processing": logging.getLogger("user_processing"),
        "genre_processing": logging.getLogger("genre_processing"),
        "quantum": logging.getLogger("quantum"),
        "ode": logging.getLogger("ode"),
        "transformer": logging.getLogger("transformer")
    }
    
    for name, logger in loggers.items():
        logger.setLevel(level)
        
    return loggers


# Enhanced Data collection and API client

class EnhancedAPIClient:
    """Enhanced OpenAI API client with rate limiting, caching, and retries"""
    
    def __init__(self, api_key: str, cache_dir: str = "cache", max_retries: int = 5, 
                 retry_delay: float = 1.0, rate_limit_per_minute: int = 20,
                 default_model: str = "gpt-4o-mini", embeddings_rate_limit: int = 100):
        """Initialize the API client"""
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit = rate_limit_per_minute
        self.default_model = default_model  
        self.call_timestamps = []
        self.embedding_timestamps = []
        self.embeddings_rate_limit = embeddings_rate_limit

        # Set up client and cache
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_db = os.path.join(cache_dir, "api_cache.db")
        self._setup_cache_db()
        
        self.logger = logging.getLogger("api")
        self.logger.info(f"Initialized API client with rate limit of {rate_limit_per_minute} calls/minute")
    
    def _setup_cache_db(self):
        """Set up SQLite database for caching API responses"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            request_hash TEXT PRIMARY KEY,
            model TEXT,
            request_type TEXT,
            request_data TEXT,
            response_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, model: str, request_type: str, data: Dict) -> str:
        """Generate a unique hash for the request"""
        request_str = json.dumps({"model": model, "type": request_type, "data": data}, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if request is in cache and return response if found"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute("SELECT response_data FROM api_cache WHERE request_hash = ?", (cache_key,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return json.loads(result[0])
        
        self.logger.debug(f"Cache miss for key: {cache_key[:8]}...")
        return None
    
    def _update_cache(self, cache_key: str, model: str, request_type: str, 
                     request_data: Dict, response_data: Dict):
        """Update cache with new response"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO api_cache (request_hash, model, request_type, request_data, response_data) VALUES (?, ?, ?, ?, ?)",
            (
                cache_key,
                model,
                request_type,
                json.dumps(request_data),
                json.dumps(response_data)
            )
        )
        conn.commit()
        conn.close()
        
        self.logger.debug(f"Updated cache for key: {cache_key[:8]}...")
    
    async def _enforce_embedding_rate_limit(self):
        """Enforce API rate limiting for embeddings (with higher limits)"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.embedding_timestamps = [ts for ts in self.embedding_timestamps 
                                if current_time - ts < 60]
        
        # If we've hit the rate limit, wait until we can make another call
        if len(self.embedding_timestamps) >= self.embeddings_rate_limit:
            wait_time = 60 - (current_time - self.embedding_timestamps[0])
            if wait_time > 0:
                # Use a shorter wait by dividing by a factor - this can help when
                # the timestamps aren't perfectly spaced
                adjusted_wait = wait_time / 2
                self.logger.info(f"Embedding rate limit reached, waiting {adjusted_wait:.2f} seconds...")
                await asyncio.sleep(adjusted_wait)
                # Recursive call to check again after waiting
                await self._enforce_embedding_rate_limit()
        
        # Add current timestamp to the list
        self.embedding_timestamps.append(time.time())
    
    async def _enforce_rate_limit(self):
        """Enforce API rate limiting"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.call_timestamps = [ts for ts in self.call_timestamps 
                              if current_time - ts < 60]
        
        # If we've hit the rate limit, wait until we can make another call
        if len(self.call_timestamps) >= self.rate_limit:
            wait_time = 60 - (current_time - self.call_timestamps[0])
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                # Recursive call to check again after waiting
                await self._enforce_rate_limit()
        
        # Add current timestamp to the list
        self.call_timestamps.append(time.time())
    
    async def chat_completion(self, messages: List[Dict], model: Optional[str] = None,
                         temperature: float = 0.3, 
                         max_tokens: Optional[int] = None,
                         use_cache: bool = True) -> Dict:
        """Make an async ChatCompletion API call with caching and retries"""
        
        # Use default model if not specified
        model = model or self.default_model
        
        request_data = {
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens:
            request_data["max_tokens"] = max_tokens
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(model, "chat_completion", request_data)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limit
                await self._enforce_rate_limit()
                
                # Make API call using aiohttp
                async with aiohttp.ClientSession() as session:
                    self.logger.debug(f"Making ChatCompletion API call (attempt {attempt+1})")
                    
                    # Prepare headers and payload
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "response_format": {"type": "json_object"}
                    }
                    if max_tokens:
                        payload["max_tokens"] = max_tokens
                    
                    # Make API call
                    async with session.post("https://api.openai.com/v1/chat/completions", 
                                         headers=headers, 
                                         json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            self.logger.warning(f"API error (status {response.status}): {error_text}")
                            if attempt < self.max_retries - 1:
                                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                                self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                return {"error": f"API error: {error_text}"}
                        
                        # Parse response
                        result = await response.json()
                        
                        # Extract content
                        response_content = result["choices"][0]["message"]["content"]
                        
                        try:
                            # Try to parse JSON response
                            parsed_response = json.loads(response_content)
                            
                            # Update cache if enabled
                            if use_cache:
                                self._update_cache(cache_key, model, "chat_completion", 
                                                 request_data, parsed_response)
                            
                            return parsed_response
                            
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse JSON response: {response_content[:100]}...")
                            # If last attempt, return the raw response
                            if attempt == self.max_retries - 1:
                                return {"error": "JSON parsing failed", "content": response_content}
            
            except Exception as e:
                self.logger.warning(f"API call failed (attempt {attempt+1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries reached. API call failed: {str(e)}")
                    return {"error": str(e)}
        
        return {"error": "Max retries reached"}
    
    async def embedding(self, model: str, input_text: Union[str, List[str]], 
                  use_cache: bool = True) -> Dict:
        """Get embeddings for text with caching and retries"""
        request_data = {"input": input_text}
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(model, "embedding", request_data)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                # Enforce embedding-specific rate limit
                await self._enforce_embedding_rate_limit()
                
                # Make API call
                async with aiohttp.ClientSession() as session:
                    self.logger.debug(f"Making Embedding API call (attempt {attempt+1})")
                    
                    # Prepare headers and payload
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    payload = {
                        "model": model,
                        "input": input_text
                    }
                    
                    # Make API call
                    async with session.post("https://api.openai.com/v1/embeddings", 
                                         headers=headers, 
                                         json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            self.logger.warning(f"Embedding API error (status {response.status}): {error_text}")
                            if attempt < self.max_retries - 1:
                                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                                self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                return {"error": f"API error: {error_text}"}
                        
                        # Parse response
                        result = await response.json()
                        
                        # Extract embeddings
                        if isinstance(input_text, list):
                            embeddings = [item["embedding"] for item in result["data"]]
                        else:
                            embeddings = result["data"][0]["embedding"]
                        
                        result_data = {"embeddings": embeddings}
                        
                        # Update cache if enabled
                        if use_cache:
                            self._update_cache(cache_key, model, "embedding", request_data, result_data)
                        
                        return result_data
                
            except Exception as e:
                self.logger.warning(f"Embedding API call failed (attempt {attempt+1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries reached. Embedding API call failed: {str(e)}")
                    return {"error": str(e)}
        
        return {"error": "Max retries reached"}
    

# Data collector class

class FlexibleDataCollector:
    "Enhanced data collector"

    def __init__(self, api_client: EnhancedAPIClient, data_dir: str = "data"):
        "Initializing data collector"

        self.api_client = api_client
        self.data_dir = data_dir

        os.makedirs(os.path.join(data_dir, "movies"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "people"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "relationships"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "context"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "users"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "genres"), exist_ok=True)

        self.name_normalization_cache = {}

        self.fuzzy_match_cache = {}

        self.logger = logging.getLogger("data_collection")
        self.logger.info("Flexible data collector initialized")


    def normalize_entity_name(self, entity_name: str) -> str:
        "Normalized entity name"

        if entity_name in self.name_normalization_cache:
            return self.name_normalization_cache[entity_name]
        
        # for none or empty string
        if not entity_name:
            return ""
        
        normalized = entity_name.lower()

        normalized = normalized.replace(".", "").replace(",", "").replace(":", "")
        normalized = normalized.replace("'", "").replace("\"", "").replace("-", "_")
        
        normalized = normalized.replace(" ", "_")

        while "__" in normalized:
            normalized = normalized.replace("__", "_")

        normalized = normalized.rstrip("_")

        self.name_normalization_cache[entity_name] = normalized

        return normalized
    
    def _get_file_path(self, entity_type: str, entity_name: str) -> str:
        "generate file path for entity data"

        if entity_name.startswith(('person_', 'movie_', 'genre_')):
            entity_name = entity_name.split('_', 1)[1]

        sanitized_name = self.normalize_entity_name(entity_name)

        if entity_type == "movie":
            return os.path.join(self.data_dir, "movies", f"{sanitized_name}.json")
        elif entity_type in ["actor", "director", "writer", "music_director", "cinematographer", "people"]:
            return os.path.join(self.data_dir, "people", f"{sanitized_name}.json")
        elif entity_type == "relationship":
            return os.path.join(self.data_dir, "relationships", f"{sanitized_name}.json")
        elif entity_type == "context":
            return os.path.join(self.data_dir, "context", f"{sanitized_name}.json")
        elif entity_type == "user":
            return os.path.join(self.data_dir, "users", f"{sanitized_name}.json")
        elif entity_type == "genre":
            return os.path.join(self.data_dir, "genres", f"{sanitized_name}.json")
        else:
            return os.path.join(self.data_dir, f"{entity_type}_{sanitized_name}.json") 
        
    def _load_data(self, entity_type: str, entity_name: str) -> Optional[Dict]:
        "Load entity data from file with flexible matching"

        file_path = self._get_file_path(entity_type, entity_name)

        #loading data from direct file path
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.debug(f"Loaded {entity_type} data for '{entity_name}' from {file_path}")
                return data
            except Exception as e:
                self.logger.warning(f"Failed to load {entity_type} data for '{entity_name}': {str(e)}")
       
        return None
    
    def _compute_word_similarity(self, str1: str, str2 : str) -> float:

        words1 = set(str1.lower().replace("_", " ").split())
        words2 = set(str2.lower().replace("_", " ").split())
        
        # Handle empty sets
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity: intersection size / union size
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    

    def get_collected_entities(self) -> Dict[str, List[str]]:
        "Get all collected entitites by type"

        entities = {
            "movies": [],
            "people": {
                "actor": [],
                "director": [],
                "writer": [],
                "music_director": [],
                "cinematographer": [],
                "people": []
            },
            "contexts": [],
            "users": [],
            "genres": []
        }

        movie_files = os.listdir(os.path.join(self.data_dir, "movies"))
        entities["movies"] = [os.path.splitext(f)[0].replace("_", " ") for f in movie_files 
                            if f.endswith(".json")]
        
        # Get people
        people_files = os.listdir(os.path.join(self.data_dir, "people"))

        for file_name in people_files:
            if file_name.endswith(".json"):
                try:
                    file_path = os.path.join(self.data_dir, "people", file_name)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    role = data.get("role", "")
                    name = data.get("name", "")
                    
                    if role and name:
                        if role in entities["people"]:
                            entities["people"][role].append(name)
                        else:
                            entities["people"][role] = [name]
                except Exception as e:
                    self.logger.warning(f"Error processing people file {file_name}: {str(e)}")
        
        # Get contexts
        context_files = os.listdir(os.path.join(self.data_dir, "context"))
        entities["contexts"] = [os.path.splitext(f)[0].replace("decade_", "") + "s" 
                              for f in context_files if f.endswith(".json")]
        
        # Get users
        if os.path.exists(os.path.join(self.data_dir, "users")):
            user_files = os.listdir(os.path.join(self.data_dir, "users"))
            entities["users"] = [os.path.splitext(f)[0] for f in user_files 
                               if f.endswith(".json") and not f.endswith("_persona.json")]
        
        # Get genres
        if os.path.exists(os.path.join(self.data_dir, "genres")):
            genre_files = os.listdir(os.path.join(self.data_dir, "genres"))
            entities["genres"] = [os.path.splitext(f)[0].replace("_", " ") 
                                for f in genre_files if f.endswith(".json")]
        
        return entities
    
    def get_movie_data_by_tt_id(self, tt_id: str, movie_tt_mapping: Dict[str, str]) -> Dict:
        """Get movie data by IMDB tt ID with enhanced matching"""
        if not tt_id:
            return {}
            
        # Normalize the input tt_id
        normalized_tt_id = tt_id.strip().lower()
        
        # Check if in mapping
        if normalized_tt_id in movie_tt_mapping:
            mapped_movie_title = movie_tt_mapping[normalized_tt_id]
            
            # Remove movie_ prefix if present
            if mapped_movie_title.startswith("movie_"):
                mapped_movie_title = mapped_movie_title[len("movie_"):]
            
            mapped_movie_title = self.normalize_entity_name(mapped_movie_title)
            # Try to load data
            data = self._load_data("movie", mapped_movie_title)
            if data:
                return data
        
        return {}
            


# Enhanced hypergraph and graph strucuture

class HyperEdge:
    "Represntation of a hyperedge in the hypergraph with enhanced attributes"

    def __init__(self, edge_id: str, edge_type: str, nodes: List[str], 
                 attributes: Optional[Dict] = None, temporal_info: Optional[Dict] = None,
                 weights: Optional[Dict[str, float]] = None ):
        "Initiliazing a hyperedge"

        self.edge_id = edge_id
        self.edge_type = edge_type
        self.nodes = set(nodes)
        self.attributes = attributes or {}
        self.temporal_info = temporal_info or {}

        self.weights = weights or {node: 1.0 for node in nodes}

        creation_time = temporal_info.get("creation_time") if temporal_info else None
        self.creation_time = creation_time or datetime.now().isoformat()
        self.last_updated = self.creation_time

    
    def add_node(self, node_id: str, weight: float = 1.0):
        "Add a node p the hyperedge"
        self.nodes.add(node_id)
        self.weights[node_id] = weight
        self.last_updated = datetime.now().isoformat()

    def remove_node(self, node_id: str):
        """Remove a node from the hyperedge"""
        if node_id in self.nodes:
            self.nodes.remove(node_id)
            if node_id in self.weights:
                del self.weights[node_id]
            self.last_updated = datetime.now().isoformat()

    def update_node_weight(self, node_id: str, weight: float):
        """Update the participation weight of a node"""
        if node_id in self.nodes:
            self.weights[node_id] = weight
            self.last_updated = datetime.now().isoformat()
    
    def update_attributes(self, attributes: Dict):
        """Update edge attributes"""
        self.attributes.update(attributes)
        self.last_updated = datetime.now().isoformat()
    
    def update_temporal_info(self, temporal_info: Dict):
        """Update temporal information"""
        self.temporal_info.update(temporal_info)
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert hyperedge to dictionary representation"""
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "nodes": list(self.nodes),
            "attributes": self.attributes,
            "temporal_info": self.temporal_info,
            "weights": self.weights,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HyperEdge':
        """Create hyperedge from dictionary representation"""
        edge = cls(
            edge_id=data["edge_id"],
            edge_type=data["edge_type"],
            nodes=data["nodes"],
            attributes=data["attributes"],
            temporal_info=data["temporal_info"],
            weights=data.get("weights")
        )
        
        # Set timestamps if available
        if "creation_time" in data:
            edge.creation_time = data["creation_time"]
        if "last_updated" in data:
            edge.last_updated = data["last_updated"]
            
        return edge
    

# Qunatum Inspired Tensor network

class QuantumInspiredTensorNetwork:
    """Implements quantum-inspired tensor network for efficient 
    representation of high-dimensional entity relationships with improved stability"""

    def __init__(self, dim: int, rank: int = 10, device = None):
        """
        Initialize tensor network
        
        Args:
            dim: Dimension of original embedding space
            rank: Rank of tensor decomposition
            device: Computing device (CPU/GPU)
        """

        self.dim = dim
        self.rank = min(rank, dim //2)
        self.logger = logging.getLogger("quantum")
        self.matrices = None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.logger.info(f"Initialized Quantum-Inspired Tensor Network with dim={dim}, rank={self.rank}")


    def decompose(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Decompose high-dimensional data into tensor train format
        
        Args:
            data: Input data of shape (n_samples, dim)
            
        Returns:
            List of TT-cores in the decomposition
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            self.logger.error("Input must be a numpy array")
            return []
            
        if len(data.shape) != 2:
            self.logger.error(f"Input must be 2D, got shape {data.shape}")
            return []
            
        if data.shape[1] != self.dim:
            self.logger.warning(f"Input dimension {data.shape[1]} doesn't match expected dimension {self.dim}")
            # Resize data to match expected dimension
            if data.shape[1] > self.dim:
                data = data[:, :self.dim]
            else:
                pad_width = ((0, 0), (0, self.dim - data.shape[1]))
                data = np.pad(data, pad_width, mode='constant')
        
        n_samples = data.shape[0]
        self.logger.info(f"Decomposing {n_samples} samples of dimension {self.dim} using TT decomposition")
        
        try:
            # Convert to torch tensors for efficient computation
            data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            
            # Determine number of cores in decomposition
            # For TT decomposition we can use more cores than in sequential SVD
            n_cores = max(3, int(np.ceil(np.log2(self.dim))))
            
            # Compute balanced dimensions for each core
            core_dims = self._compute_core_dimensions(n_cores)
            
            # Effective rank (capped by data dimensions)
            effective_rank = min(self.rank, min(n_samples, self.dim))
            
            # Initialize tensor train cores list
            tt_cores = []
            
            # Initialize ranks between cores (r₀=1, r_d=1, and ranks in between)
            ranks = [1] + [effective_rank] * (n_cores - 1) + [1]
            
            # Reshape original tensor to match expected TT format dimensions
            reshaped_data = data_tensor.reshape(n_samples, -1)
            curr_data = reshaped_data
            
            # Left-to-right sweep (TT-SVD algorithm)
            for k in range(n_cores - 1):
                # Reshape tensor for current core separation
                mode_size = core_dims[k]
                mode_matrix = curr_data.reshape(-1, mode_size * ranks[k+1])
                
                # SVD to separate current core
                try:
                    U, S, V = torch.svd(mode_matrix)
                    
                    # Truncate to effective rank
                    trunc_rank = min(effective_rank, S.shape[0])
                    U = U[:, :trunc_rank]
                    S = S[:trunc_rank]
                    V = V[:, :trunc_rank]
                    
                    # Construct current core tensor
                    curr_core = U.reshape(n_samples, ranks[k], core_dims[k], ranks[k+1])
                    tt_cores.append(curr_core.cpu().numpy())
                    
                    # Update remaining data
                    curr_data = torch.matmul(torch.diag(S), V.t()).t()
                
                except Exception as e:
                    self.logger.error(f"SVD failed at core {k}: {str(e)}")
                    # Attempt recovery with randomized approach
                    if k == 0:
                        return []  # If first core fails, abort
                    break
            
            # Add the last core
            if len(tt_cores) == n_cores - 1:
                last_core_shape = (n_samples, ranks[-2], core_dims[-1], ranks[-1])
                last_core = curr_data.reshape(last_core_shape)
                tt_cores.append(last_core.cpu().numpy())
            
            self.matrices = tt_cores
            self.logger.info(f"TT decomposition complete, created {len(tt_cores)} cores")
            return tt_cores
            
        except Exception as e:
            self.logger.error(f"Error in TT decomposition: {str(e)}")
            traceback.print_exc()
            return []

    def _compute_core_dimensions(self, n_cores):
        """Compute balanced dimensions for TT-cores"""
        # For TT we want more balanced dimensions across cores
        remaining_dim = self.dim
        core_dims = []
        
        # Get approximately balanced dimensions
        for i in range(n_cores):
            if i == n_cores - 1:
                # Last core gets all remaining dimensions
                core_dims.append(remaining_dim)
            else:
                # For TT decomposition we want more equally sized cores
                # Using nth root to determine core size
                core_dim = max(2, int(np.ceil(self.dim ** (1/n_cores))))
                # Make sure we don't exceed remaining dimensions
                core_dim = min(core_dim, remaining_dim - (n_cores - i - 1))
                core_dims.append(core_dim)
                remaining_dim -= core_dim
        
        # Ensure dimensions sum to original dimension
        if sum(core_dims) != self.dim:
            diff = self.dim - sum(core_dims)
            core_dims[-1] += diff
            
        return core_dims
            
    def reconstruct(self, matrices: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Reconstruct original data from tensor train network
        
        Args:
            matrices: List of TT-cores in the decomposition
            
        Returns:
            Reconstructed data of shape (n_samples, dim)
        """
        if matrices is None:
            matrices = self.matrices
            
        if matrices is None or len(matrices) == 0:
            self.logger.error("No matrices available for reconstruction")
            return None
            
        try:
            # Use the first matrix to determine number of samples
            n_samples = matrices[0].shape[0]
            
            # Convert to torch tensors
            torch_matrices = [torch.tensor(m, dtype=torch.float32, device=self.device) for m in matrices]
            
            # For each sample, perform TT contraction
            results = []
            
            for s in range(n_samples):
                # Start with the first core
                result = torch_matrices[0][s]  # Shape: [r0, n1, r1]
                
                # Contract with subsequent cores
                for i in range(1, len(torch_matrices)):
                    # Get current core for this sample
                    curr_core = torch_matrices[i][s]  # Shape: [r{i-1}, ni, ri]
                    
                    # Contract along the rank dimension
                    # Reshape result to matrix for contraction
                    result_mat = result.reshape(-1, result.shape[-1])  # Shape: [r0*n1*...*n{i-1}, r{i-1}]
                    core_mat = curr_core.reshape(curr_core.shape[0], -1)  # Shape: [r{i-1}, ni*ri]
                    
                    # Matrix multiplication for contraction
                    result = torch.matmul(result_mat, core_mat)  # Shape: [r0*n1*...*n{i-1}, ni*ri]
                    
                    # Reshape to incorporate new core's dimensions
                    new_shape = list(result.shape[:-1]) + [curr_core.shape[1], curr_core.shape[2]]
                    result = result.reshape(*new_shape)  # Restore tensor format
                
                # Final result has shape [1, n1, n2, ..., nd, 1] for each sample
                # We need to reshape to a vector
                final_vec = result.reshape(-1)
                
                # Ensure dimension matches original input
                if final_vec.shape[0] < self.dim:
                    padded = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
                    padded[:final_vec.shape[0]] = final_vec
                    final_vec = padded
                else:
                    final_vec = final_vec[:self.dim]
                    
                results.append(final_vec)
                
            # Stack all sample results
            final_result = torch.stack(results)
            
            return final_result.cpu().numpy()
                
        except Exception as e:
            self.logger.error(f"Error in tensor train reconstruction: {str(e)}")
            traceback.print_exc()
            return None
            
    def compute_similarity(self, idx1: int, idx2: int) -> float:
        """
        Compute similarity between two entities directly using tensor network
        
        Args:
            idx1: Index of first entity
            idx2: Index of second entity
            
        Returns:
            Similarity score between entities
        """
        if self.matrices is None or len(self.matrices) == 0:
            self.logger.error("No matrices available for similarity computation")
            return 0.0
            
        try:
            # Compute similarity directly in tensor network space
            # This is more efficient than reconstructing full vectors
            similarity = 1.0
            
            for i in range(len(self.matrices)):
                # Get cores for each entity
                if idx1 >= self.matrices[i].shape[0] or idx2 >= self.matrices[i].shape[0]:
                    self.logger.error(f"Index out of bounds: got {idx1} and {idx2}, max is {self.matrices[i].shape[0]-1}")
                    return 0.0
                
                core1 = self.matrices[i][idx1]
                core2 = self.matrices[i][idx2]
                
                # Compute contribution to similarity from this core
                if i == 0 or i == len(self.matrices) - 1:
                    # For first and last cores, we use Frobenius inner product
                    core_similarity = np.sum(core1 * core2) / (
                        np.sqrt(np.sum(core1 * core1) + 1e-8) * 
                        np.sqrt(np.sum(core2 * core2) + 1e-8)
                    )
                else:
                    # For intermediate cores, we use normalized trace
                    core_similarity = np.trace(core1 @ core2.T) / (
                        np.sqrt(np.trace(core1 @ core1.T) + 1e-8) * 
                        np.sqrt(np.trace(core2 @ core2.T) + 1e-8)
                    )
                
                # Ensure similarity is valid
                core_similarity = np.clip(core_similarity, -1.0, 1.0)
                
                # Overall similarity is product of core similarities
                similarity *= max(0, core_similarity)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing tensor network similarity: {str(e)}")
            return 0.0


class GraphODELayer(nn.Module):
    """
    Neural ODE layer for graph evolution modeling with learnable parameters
    """
    
    def __init__(self, node_dim: int, time_dim: int = 8, hidden_dim: int = 64, device=None):
        """
        Initialize Graph ODE layer
        
        Args:
            node_dim: Dimension of node features
            time_dim: Dimension of time encoding
            hidden_dim: Hidden dimension for neural networks
            device: Torch device to use
        """
        super(GraphODELayer, self).__init__()
        self.node_dim = node_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural network to parameterize the ODE dynamics
        self.dynamics_nn = nn.Sequential(
            nn.Linear(node_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        ).to(self.device)
        
        # Time embedding network for more expressive time representation
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        ).to(self.device)
        
        # Initialize using Kaiming initialization for better training
        self._init_weights()
        
        self.logger = logging.getLogger("ode")
        self.logger.info(f"Initialized Graph ODE Layer with node dimension {node_dim}, time dimension {time_dim}")
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _get_time_embedding(self, t):
        """Generate time embedding using learned network"""
        # Reshape t for network input
        t_input = t.reshape(-1, 1) if torch.is_tensor(t) else torch.tensor([[t]], dtype=torch.float32, device=self.device)
        
        # Pass through the embedding network
        return self.time_embedding(t_input)
    
    def forward(self, t, x):
        """
        Forward pass computes the derivative of node features with respect to time
        
        Args:
            t: Current time point
            x: Node features at current time
            
        Returns:
            Time derivative of node features (dx/dt)
        """
        # Get time embedding
        if torch.is_tensor(t) and t.dim() > 0:
            time_input = t[0].item()  # Use the first element if t is a tensor
        else:
            time_input = t
        
        time_embedding = self._get_time_embedding(time_input)
        
        # Expand time embedding to match batch dimension of x
        batch_size = x.shape[0]
        time_embedding = time_embedding.expand(batch_size, -1)
        
        # Concatenate node features with time embedding
        x_with_time = torch.cat([x, time_embedding], dim=1)
        
        # Compute derivative using neural network
        dx_dt = self.dynamics_nn(x_with_time)
        
        return dx_dt
    
    def integrate(self, x0, t_span, method='dopri5', rtol=1e-3, atol=1e-4):
        """
        Integrate ODE from initial condition x0 over time span
        
        Args:
            x0: Initial node features
            t_span: Time points to evaluate at
            method: Integration method
            rtol: Relative tolerance for adaptive step size methods
            atol: Absolute tolerance for adaptive step size methods
            
        Returns:
            Node features at each time point in t_span
        """
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        
        if not isinstance(t_span, torch.Tensor):
            t_span = torch.tensor(t_span, dtype=torch.float32, device=self.device)
        
        try:
            # Use torchdiffeq for ODE integration
            # This solves the ODE defined by self.forward
            solution = odeint(
                self,
                x0,
                t_span,
                method=method,
                rtol=rtol,
                atol=atol
            )
            
            self.logger.info(f"Successfully integrated ODE over {len(t_span)} time points")
            return solution
            
        except Exception as e:
            self.logger.error(f"Error integrating ODE: {str(e)}")
            traceback.print_exc()
            
            # Fallback: linear interpolation
            self.logger.info("Using fallback linear interpolation")
            
            # Create simple linear interpolation between start and end
            t_start, t_end = t_span[0].item(), t_span[-1].item()
            result = torch.zeros((len(t_span), x0.shape[0], x0.shape[1]), device=self.device)
            
            for i, t in enumerate(t_span):
                # Linear interpolation factor
                alpha = (t.item() - t_start) / (t_end - t_start) if t_end > t_start else 0
                
                # Apply small random dynamics to make it look like ODE solution
                noise_scale = 0.01 * alpha * (1 - alpha)  # Peaks in the middle
                noise = torch.randn_like(x0) * noise_scale
                
                # Create interpolated result
                result[i] = x0 + alpha * self.forward(t, x0) + noise
            
            return result
    
    def train_temporal_model(self, node_features, time_points, target_features, 
                           num_epochs=100, learning_rate=0.001):
        """
        Train the ODE model using temporal data
        
        Args:
            node_features: Initial node features at time t0
            time_points: Time points for training
            target_features: Target node features at each time point
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training losses
        """
        # Convert to tensors
        x0 = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        t_span = torch.tensor(time_points, dtype=torch.float32, device=self.device)
        targets = torch.tensor(target_features, dtype=torch.float32, device=self.device)
        
        # Set model to training mode
        self.train()
        
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        
        self.logger.info(f"Beginning ODE model training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass - integrate ODE
            try:
                predictions = odeint(
                    self,
                    x0,
                    t_span,
                    method='dopri5',
                    rtol=1e-3,
                    atol=1e-4
                )
                
                # Compute loss (MSE)
                loss = F.mse_loss(predictions, targets)
                
                # Add regularization to ensure smooth trajectories
                # Compute finite difference approximation of second derivative
                if len(predictions) > 2:
                    second_derivatives = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
                    smoothness_loss = torch.mean(torch.sum(second_derivatives**2, dim=(1, 2)))
                    loss = loss + 0.01 * smoothness_loss
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Record loss
                losses.append(loss.item())
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                self.logger.error(f"Error during training epoch {epoch+1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Set model back to evaluation mode
        self.eval()
        
        self.logger.info(f"ODE model training completed, final loss: {losses[-1]:.6f}")
        
        return losses
    
class TemporalHypergraph:
    "Temporal hypergraph representation for knowledge graph with enhanced ODE capability"

    def __init__(self, name: str = "bollywood_hypergraph", ode_dim : int = 64, device = None):
        "Initialize temporal hypergraph"

        self.name = name
        self.nodes = {}
        self.hyperedges = {}
        self.node_to_edges = defaultdict(set)

        # temporal indices
        self.temporal_edges = defaultdict(list)
        self.temporal_nodes = defaultdict(set)

        # type indices
        self.node_types = defaultdict(set)
        self.edge_types = defaultdict(set)

        # advanced temporal modeling
        self.ode_dim = ode_dim
        self.ode_layer = None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_embeddings = {}  # node_id -> temporal embedding trajectory
        self.edge_embeddings = {}  # edge_id -> temporal embedding trajectory
        
        # ODE integration points
        self.integration_years = np.linspace(1950, 2025, 50)  # 50 points from 1950 to 2025
        
        # Statistics
        self.stats = {
            "node_count": 0,
            "edge_count": 0,
            "creation_time": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        self.logger = logging.getLogger("hypergraph")
        self.logger.info(f"Initialized temporal hypergraph: {name}")

    def add_node(self, node_id: str, node_type: str, attributes: Dict, 
                temporal_info: Optional[Dict] = None) -> bool:
        """Add a node to the hypergraph"""
        if node_id in self.nodes:
            self.logger.debug(f"Node {node_id} already exists, updating attributes")
            self.nodes[node_id].update(attributes)
            return False
        
        # Add node
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            **attributes
        }
        
        # Add to type index
        self.node_types[node_type].add(node_id)
        
        # Add temporal information if provided
        if temporal_info:
            self.nodes[node_id]["temporal_info"] = temporal_info
            
            # Add to temporal index
            if "year" in temporal_info:
                year = temporal_info["year"]
                if isinstance(year, int):
                    self.temporal_nodes[year].add(node_id)
                elif isinstance(year, str) and year.isdigit():
                    self.temporal_nodes[int(year)].add(node_id)
        
        # Update statistics
        self.stats["node_count"] += 1
        self.stats["last_modified"] = datetime.now().isoformat()
        
        self.logger.debug(f"Added node: {node_id} ({node_type})")
        return True
    
    def add_hyperedge(self, edge_id: str, edge_type: str, nodes: List[str], 
                     attributes: Optional[Dict] = None, 
                     temporal_info: Optional[Dict] = None,
                     weights: Optional[Dict[str, float]] = None) -> bool:
        """Add a hyperedge to the hypergraph with auto-creation of missing nodes"""
        if edge_id in self.hyperedges:
            self.logger.debug(f"Edge {edge_id} already exists, updating attributes")
            if attributes:
                self.hyperedges[edge_id].update_attributes(attributes)
            if temporal_info:
                self.hyperedges[edge_id].update_temporal_info(temporal_info)
            return False
        
        # Auto-create missing nodes
        valid_nodes = []
        for node_id in nodes:
            if node_id in self.nodes:
                valid_nodes.append(node_id)
            else:
                # Extract node type and name from ID
                if "_" in node_id:
                    parts = node_id.split("_", 1)
                    if len(parts) == 2:
                        node_type = parts[0]
                        node_name = parts[1].replace("_", " ").title()
                        
                        # Create basic node
                        self.add_node(
                            node_id=node_id,
                            node_type=node_type,
                            attributes={
                                "name": node_name,
                                "auto_created": True
                            }
                        )
                        valid_nodes.append(node_id)
                        self.logger.info(f"Auto-created missing node: {node_id}")
                        continue
                
                self.logger.warning(f"Node {node_id} does not exist, cannot add to hyperedge")
        
        # Only proceed if we have at least two valid nodes
        if len(valid_nodes) < 2:
            return False
        
        # Set default weights if not provided
        if weights is None:
            weights = {node_id: 1.0 for node_id in valid_nodes}
        else:
            # Ensure all nodes have weights
            for node_id in valid_nodes:
                if node_id not in weights:
                    weights[node_id] = 1.0
        
        # Create hyperedge with valid nodes
        edge = HyperEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            nodes=valid_nodes,
            attributes=attributes or {},
            temporal_info=temporal_info or {},
            weights=weights
        )
        
        # Add edge
        self.hyperedges[edge_id] = edge
        
        # Update node-edge index
        for node_id in valid_nodes:
            self.node_to_edges[node_id].add(edge_id)
        
        # Add to type index
        self.edge_types[edge_type].add(edge_id)
        
        # Add to temporal index if applicable
        if temporal_info and "year" in temporal_info:
            year = temporal_info["year"]
            if isinstance(year, int):
                self.temporal_edges[year].append(edge_id)
            elif isinstance(year, str) and year.isdigit():
                self.temporal_edges[int(year)].append(edge_id)
        
        # Update statistics
        self.stats["edge_count"] += 1
        self.stats["last_modified"] = datetime.now().isoformat()
        
        self.logger.debug(f"Added hyperedge: {edge_id} ({edge_type}) connecting {len(valid_nodes)} nodes")
        return True
    
    def initialize_ode(self, hidden_dim=128, device=None):
        """Initialize ODE layer for temporal modeling"""
        if self.ode_layer is not None:
            return
            
        device = device or self.device
        self.ode_layer = GraphODELayer(
            node_dim=self.ode_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        self.logger.info(f"Initialized ODE layer with dimension {self.ode_dim}, hidden dimension {hidden_dim}")
    
    def generate_temporal_embeddings(self, node_embeddings: Dict[str, np.ndarray], 
                                    years_range: Tuple[int, int] = (1950, 2025),
                                    train_model: bool = True) -> Dict[str, Dict]:
        """
        Generate temporal embeddings for nodes using Graph ODE
        
        Args:
            node_embeddings: Dictionary of node_id -> static embedding
            years_range: Range of years to model
            train_model: Whether to train the ODE model with temporal data
            
        Returns:
            Dictionary of node_id -> temporal embedding trajectories
        """
        self.logger.info(f"Generating temporal embeddings from {years_range[0]} to {years_range[1]}")
        
        # Initialize ODE layer if not already done
        self.initialize_ode()
        
        # Create node feature matrix
        node_ids = list(node_embeddings.keys())
        
        # Reduce dimension if needed
        embedding_dim = next(iter(node_embeddings.values())).shape[0]
        if embedding_dim > self.ode_dim:
            # Use PCA for dimension reduction
            pca = PCA(n_components=self.ode_dim)
            
            # Collect embeddings in a matrix
            X = np.stack([node_embeddings[node_id] for node_id in node_ids])
            
            # Fit PCA
            pca = PCA(n_components=self.ode_dim)
            X_reduced = pca.fit_transform(X)
            
            # Create reduced embeddings dictionary
            reduced_embeddings = {node_id: X_reduced[i] for i, node_id in enumerate(node_ids)}
        else:
            # Pad embeddings if dimension is smaller
            reduced_embeddings = {
                node_id: np.pad(
                    node_embeddings[node_id], 
                    (0, self.ode_dim - embedding_dim),
                    mode='constant'
                ) for node_id in node_ids
            }
        
        # Convert to tensor
        node_features = torch.tensor(
            np.stack([reduced_embeddings[node_id] for node_id in node_ids]),
            dtype=torch.float32,
            device=self.ode_layer.device
        )
        
        # Create time span for integration
        year_start, year_end = years_range
        t_span = torch.linspace(0, 1, 50, device=self.ode_layer.device)  # 50 points in [0, 1]
        
        # Map from integration time to actual years
        year_mapping = lambda t: year_start + t * (year_end - year_start)
        
        # If training data is available, train the ODE model
        if train_model:
            # Find nodes with temporal information for training
            training_nodes = []
            training_years = []
            training_features = []
            
            for i, node_id in enumerate(node_ids):
                if node_id not in self.nodes:
                    continue
                    
                # Get node data
                node_data = self.nodes[node_id]
                
                # Check if node has year information
                if "temporal_info" in node_data and "year" in node_data["temporal_info"]:
                    year = node_data["temporal_info"]["year"]
                    if isinstance(year, str) and year.isdigit():
                        year = int(year)
                    
                    if isinstance(year, int) and year_start <= year <= year_end:
                        # Normalize year to [0, 1]
                        normalized_year = (year - year_start) / (year_end - year_start)
                        
                        # Add to training data
                        training_nodes.append(i)
                        training_years.append(normalized_year)
                        training_features.append(node_features[i])
            
            # If we have enough training data, train the model
            if len(training_nodes) >= 10:  # Need a reasonable amount of data
                self.logger.info(f"Training ODE model with {len(training_nodes)} nodes with temporal information")
                
                # Create training data
                train_indices = torch.tensor(training_nodes, dtype=torch.long, device=self.ode_layer.device)
                train_years = torch.tensor(training_years, dtype=torch.float32, device=self.ode_layer.device)
                train_features = torch.stack(training_features)
                
                # Train for a few epochs
                try:
                    # Create target features with small variations based on time
                    # (This is a simple approach - in a real system you might have actual data points)
                    target_features = []
                    for i, normalized_year in enumerate(training_years):
                        # Create a variation of the feature based on time
                        time_factor = normalized_year * 0.2  # Max 20% variation
                        variation = torch.randn_like(training_features[i]) * time_factor
                        target = training_features[i] + variation
                        target_features.append(target)
                    
                    target_features = torch.stack(target_features)
                    
                    # Train the model
                    self.ode_layer.train_temporal_model(
                        training_features,
                        train_years,
                        target_features,
                        num_epochs=50,
                        learning_rate=0.001
                    )
                except Exception as e:
                    self.logger.error(f"Error training ODE model: {str(e)}")
                    traceback.print_exc()
                    # Continue with untrained model
        
        try:
            # Set model to eval mode for inference
            self.ode_layer.eval()
            
            # Integrate ODE to get temporal embeddings
            with torch.no_grad():
                trajectory = self.ode_layer.integrate(node_features, t_span)
            
            # Convert to numpy and store
            trajectory_np = trajectory.detach().cpu().numpy()
            
            # Store temporal embeddings
            temporal_embeddings = {}
            for i, node_id in enumerate(node_ids):
                temporal_embeddings[node_id] = {
                    'static': node_embeddings[node_id],
                    'temporal': trajectory_np[:, i, :],
                    'years': [year_mapping(t.item()) for t in t_span]
                }
            
            self.node_embeddings = temporal_embeddings
            self.logger.info(f"Generated temporal embeddings for {len(node_ids)} nodes")
            
            # Clean up to free GPU memory
            del trajectory
            del node_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return temporal_embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating temporal embeddings: {str(e)}")
            traceback.print_exc()
            return {}
    
    def get_node_embedding_at_time(self, node_id: str, year: int) -> Optional[np.ndarray]:
        """
        Get node embedding at a specific point in time
        
        Args:
            node_id: ID of the node
            year: Year to get embedding for
            
        Returns:
            Embedding vector at the specified time
        """
        if node_id not in self.node_embeddings:
            self.logger.warning(f"No temporal embedding found for node {node_id}")
            return None
            
        embedding_data = self.node_embeddings[node_id]
        years = embedding_data['years']
        
        # Find closest year in the trajectory
        if year <= years[0]:
            return embedding_data['temporal'][0]
        elif year >= years[-1]:
            return embedding_data['temporal'][-1]
        else:
            # Interpolate between nearest years
            for i in range(len(years) - 1):
                if years[i] <= year <= years[i+1]:
                    # Linear interpolation
                    alpha = (year - years[i]) / (years[i+1] - years[i])
                    return (1 - alpha) * embedding_data['temporal'][i] + alpha * embedding_data['temporal'][i+1]
        
        # Fallback to static embedding
        return embedding_data['static']
    
    def compute_temporal_similarity(self, node_id1: str, node_id2: str, year: int) -> float:
        """
        Compute similarity between nodes at a specific point in time
        
        Args:
            node_id1: First node ID
            node_id2: Second node ID
            year: Year to compute similarity for
            
        Returns:
            Similarity score (cosine similarity)
        """
        # Get embeddings at specified time
        emb1 = self.get_node_embedding_at_time(node_id1, year)
        emb2 = self.get_node_embedding_at_time(node_id2, year)
        
        if emb1 is None or emb2 is None:
            return 0.0
            
        # Compute cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
        
        return 0.0
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[HyperEdge]:
        """Get hyperedge by ID"""
        return self.hyperedges.get(edge_id)
    
    def get_node_edges(self, node_id: str) -> List[HyperEdge]:
        """Get all hyperedges connected to a node"""
        edge_ids = self.node_to_edges.get(node_id, set())
        return [self.hyperedges[edge_id] for edge_id in edge_ids if edge_id in self.hyperedges]
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type"""
        node_ids = self.node_types.get(node_type, set())
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def get_edges_by_type(self, edge_type: str) -> List[HyperEdge]:
        """Get all hyperedges of a specific type"""
        edge_ids = self.edge_types.get(edge_type, set())
        return [self.hyperedges[edge_id] for edge_id in edge_ids if edge_id in self.hyperedges]
    
    def to_networkx(self) -> Tuple[nx.Graph, nx.DiGraph]:
        """Convert hypergraph to NetworkX graph representations"""
        self.logger.info("Converting hypergraph to NetworkX graphs")
        
        # Create graphs
        graph = nx.Graph()
        digraph = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node_attrs in self.nodes.items():
            # Copy attributes
            attrs = node_attrs.copy()
            
            if "temporal_info" in attrs:
                temporal_info = attrs["temporal_info"]
                # Add each temporal attribute directly to the node
                for temp_key, temp_value in temporal_info.items():
                    attrs[temp_key] = temp_value
            # Convert non-serializable attributes
            for key, value in list(attrs.items()):
                if isinstance(value, (set, frozenset)):
                    attrs[key] = list(value)
                elif isinstance(value, (np.ndarray)):
                    attrs[key] = value.tolist()
                elif value is None:
                    # Fix for GraphML serialization issues with None values
                    attrs[key] = ''
            
            # Add to both graphs
            graph.add_node(node_id, **attrs)
            digraph.add_node(node_id, **attrs)
        
        # Process hyperedges
        for edge_id, edge in self.hyperedges.items():
            # For binary edges (connecting exactly 2 nodes), add direct edges
            if len(edge.nodes) == 2:
                nodes = list(edge.nodes)
                
                # Determine direction for directed graph
                source, target = nodes
                
                # Check edge type for direction hints
                if edge.edge_type.lower() in ["acted_in", "directed", "wrote", "composed_for", "shot"]:
                    # These suggest person -> movie direction
                    if self.nodes.get(source, {}).get("node_type") == "movie":
                        source, target = target, source
                elif edge.edge_type.lower() in ["has_genre", "has_theme", "has_mood"]:
                    # These suggest movie -> attribute direction
                    if self.nodes.get(source, {}).get("node_type") != "movie":
                        source, target = target, source
                elif edge.edge_type.lower() == "rated":
                    # User -> movie direction for ratings
                    if self.nodes.get(source, {}).get("node_type") != "user":
                        source, target = target, source
                
                # Edge attributes
                edge_attrs = {
                    "edge_type": edge.edge_type,
                    "edge_id": edge_id,
                    **edge.attributes
                }
                
                # Add temporal information if available
                if edge.temporal_info:
                    # Convert temporal info to serializable format
                    serializable_temporal = {}
                    for k, v in edge.temporal_info.items():
                        if v is None:
                            serializable_temporal[k] = ''
                        elif isinstance(v, (dict, list, set)):
                            serializable_temporal[k] = json.dumps(v)
                        else:
                            serializable_temporal[k] = v
                    edge_attrs["temporal_info"] = serializable_temporal
                
                # Add node participation weights
                if hasattr(edge, 'weights') and edge.weights:
                    source_weight = edge.weights.get(source, 1.0)
                    target_weight = edge.weights.get(target, 1.0)
                    edge_attrs["source_weight"] = float(source_weight)
                    edge_attrs["target_weight"] = float(target_weight)
                    edge_attrs["edge_weight"] = float((source_weight + target_weight) / 2)
                
                # Convert any None values to empty strings to avoid GraphML serialization issues
                for k, v in list(edge_attrs.items()):
                    if v is None:
                        edge_attrs[k] = ''
                
                # Add to both graphs
                graph.add_edge(nodes[0], nodes[1], **edge_attrs)
                digraph.add_edge(source, target, **edge_attrs)
            
            # For hyperedges with more than 2 nodes, create a central "hyperedge node"
            # and connect all nodes to it
            elif len(edge.nodes) > 2:
                hyperedge_node_id = f"hyperedge_{edge_id}"
                
                # Add hyperedge node
                hyperedge_attrs = {
                    "node_type": "hyperedge",
                    "edge_type": edge.edge_type,
                    "original_edge_id": edge_id
                }
                
                # Add edge attributes
                for k, v in edge.attributes.items():
                    if isinstance(v, (dict, list, set)):
                        hyperedge_attrs[k] = json.dumps(v)
                    elif v is None:
                        hyperedge_attrs[k] = ''
                    else:
                        hyperedge_attrs[k] = v
                
                # Add temporal information if available
                if edge.temporal_info:
                    for k, v in edge.temporal_info.items():
                        key = f"temporal_{k}"
                        if isinstance(v, (dict, list, set)):
                            hyperedge_attrs[key] = json.dumps(v)
                        elif v is None:
                            hyperedge_attrs[key] = ''
                        else:
                            hyperedge_attrs[key] = v
                
                graph.add_node(hyperedge_node_id, **hyperedge_attrs)
                digraph.add_node(hyperedge_node_id, **hyperedge_attrs)
                
                # Connect all nodes to hyperedge node
                for node_id in edge.nodes:
                    # Get node participation weight
                    if hasattr(edge, 'weights') and edge.weights:
                        weight = edge.weights.get(node_id, 1.0)
                    else:
                        weight = 1.0
                    
                    edge_attrs = {
                        "edge_type": edge.edge_type,
                        "edge_weight": float(weight),
                        "part_of_hyperedge": edge_id
                    }
                    
                    graph.add_edge(node_id, hyperedge_node_id, **edge_attrs)
                    
                    # For directed graph, use edge type to determine direction
                    if edge.edge_type.lower() in ["acted_in", "directed", "wrote", "composed_for", "shot"]:
                        # Person -> movie/hyperedge direction
                        if self.nodes.get(node_id, {}).get("node_type") in ["actor", "director", "writer", "music_director", "cinematographer"]:
                            digraph.add_edge(node_id, hyperedge_node_id, **edge_attrs)
                        else:
                            digraph.add_edge(hyperedge_node_id, node_id, **edge_attrs)
                    else:
                        # Default: node -> hyperedge direction
                        digraph.add_edge(node_id, hyperedge_node_id, **edge_attrs)
        
        self.logger.info(f"Converted hypergraph to NetworkX graphs: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph, digraph
    
    def save(self, file_path: str):
        """Save hypergraph to file"""
        self.logger.info(f"Saving hypergraph to {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convert to serializable format
        data = {
            "name": self.name,
            "nodes": self.nodes,
            "hyperedges": {edge_id: edge.to_dict() for edge_id, edge in self.hyperedges.items()},
            "stats": self.stats
        }
        
        # Add temporal embeddings if available
        if self.node_embeddings:
            # Convert numpy arrays to lists
            serializable_embeddings = {}
            for node_id, emb_data in self.node_embeddings.items():
                serializable_embeddings[node_id] = {
                    'static': emb_data['static'].tolist() if isinstance(emb_data['static'], np.ndarray) else emb_data['static'],
                    'temporal': emb_data['temporal'].tolist() if isinstance(emb_data['temporal'], np.ndarray) else emb_data['temporal'],
                    'years': emb_data['years']
                }
            data["node_embeddings"] = serializable_embeddings
        
        # Save ODE layer state if available and trained
        if self.ode_layer is not None:
            ode_state_path = file_path.replace(".pkl", "_ode_state.pt")
            try:
                torch.save(self.ode_layer.state_dict(), ode_state_path)
                data["ode_state_path"] = ode_state_path
                self.logger.info(f"Saved ODE model state to {ode_state_path}")
            except Exception as e:
                self.logger.error(f"Error saving ODE model state: {str(e)}")
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved hypergraph with {len(self.nodes)} nodes and {len(self.hyperedges)} hyperedges")
    
    @classmethod
    def load(cls, file_path: str, device=None) -> 'TemporalHypergraph':
        """Load hypergraph from file"""
        logger = logging.getLogger("hypergraph")
        logger.info(f"Loading hypergraph from {file_path}")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Create hypergraph
        hypergraph = cls(name=data["name"], device=device)
        
        # Load nodes
        for node_id, node_attrs in data["nodes"].items():
            node_type = node_attrs.get("node_type", "unknown")
            temporal_info = None
            
            if "temporal_info" in node_attrs:
                temporal_info = node_attrs["temporal_info"]
                # Remove temporal_info to avoid duplication
                node_attrs_copy = node_attrs.copy()
                del node_attrs_copy["temporal_info"]
            else:
                node_attrs_copy = node_attrs
            
            hypergraph.add_node(
                node_id=node_id,
                node_type=node_type,
                attributes=node_attrs_copy,
                temporal_info=temporal_info
            )
        
        # Load edges
        for edge_id, edge_data in data["hyperedges"].items():
            # Check if edge data is a dict or HyperEdge object
            if isinstance(edge_data, dict):
                # Extract weights if available
                weights = edge_data.get("weights")
                
                hypergraph.add_hyperedge(
                    edge_id=edge_id,
                    edge_type=edge_data["edge_type"],
                    nodes=edge_data["nodes"],
                    attributes=edge_data["attributes"],
                    temporal_info=edge_data["temporal_info"],
                    weights=weights
                )
            else:
                # Legacy format - convert to new format
                hypergraph.add_hyperedge(
                    edge_id=edge_id,
                    edge_type=edge_data.edge_type,
                    nodes=list(edge_data.nodes),
                    attributes=edge_data.attributes,
                    temporal_info=edge_data.temporal_info
                )
        
        # Load temporal embeddings if available
        if "node_embeddings" in data:
            hypergraph.node_embeddings = {}
            for node_id, emb_data in data["node_embeddings"].items():
                hypergraph.node_embeddings[node_id] = {
                    'static': np.array(emb_data['static']),
                    'temporal': np.array(emb_data['temporal']),
                    'years': emb_data['years']
                }
            
            # Determine ODE dimension from loaded embeddings
            if hypergraph.node_embeddings:
                first_node = next(iter(hypergraph.node_embeddings.values()))
                if 'temporal' in first_node and isinstance(first_node['temporal'], np.ndarray):
                    hypergraph.ode_dim = first_node['temporal'].shape[1]
        
        # Load ODE model if state path is available
        if "ode_state_path" in data and os.path.exists(data["ode_state_path"]):
            try:
                hypergraph.initialize_ode(device=device)
                hypergraph.ode_layer.load_state_dict(torch.load(
                    data["ode_state_path"],
                    map_location=device
                ))
                hypergraph.ode_layer.eval()  # Set to evaluation mode
                logger.info(f"Loaded ODE model state from {data['ode_state_path']}")
            except Exception as e:
                logger.error(f"Error loading ODE model state: {str(e)}")
        
        # Update statistics
        hypergraph.stats = data["stats"]
        
        logger.info(f"Loaded hypergraph with {len(hypergraph.nodes)} nodes and {len(hypergraph.hyperedges)} hyperedges")
        return hypergraph
    
    def compute_statistics(self) -> Dict:
        """Compute detailed statistics for the hypergraph"""
        self.logger.info("Computing hypergraph statistics")
        
        stats = {
            "node_count": len(self.nodes),
            "edge_count": len(self.hyperedges),
            "node_type_distribution": {
                node_type: len(nodes) for node_type, nodes in self.node_types.items()
            },
            "edge_type_distribution": {
                edge_type: len(edges) for edge_type, edges in self.edge_types.items()
            },
            "temporal_distribution": {
                "nodes_by_year": {
                    year: len(nodes) for year, nodes in self.temporal_nodes.items()
                },
                "edges_by_year": {
                    year: len(edges) for year, edges in self.temporal_edges.items()
                }
            },
            "average_degree": 0,
            "average_edge_size": 0
        }
        
        # Calculate average degree (number of edges per node)
        if self.nodes:
            total_degree = sum(len(edges) for edges in self.node_to_edges.values())
            stats["average_degree"] = total_degree / len(self.nodes)
        
        # Calculate average edge size (number of nodes per edge)
        if self.hyperedges:
            total_size = sum(len(edge.nodes) for edge in self.hyperedges.values())
            stats["average_edge_size"] = total_size / len(self.hyperedges)
        
        # More detailed statistics
        # Compute node connectivity distribution
        degree_dist = {}
        for node_id, edges in self.node_to_edges.items():
            degree = len(edges)
            degree_dist[degree] = degree_dist.get(degree, 0) + 1
        stats["node_degree_distribution"] = degree_dist
        
        # Compute hyperedge size distribution
        edge_size_dist = {}
        for edge in self.hyperedges.values():
            size = len(edge.nodes)
            edge_size_dist[size] = edge_size_dist.get(size, 0) + 1
        stats["edge_size_distribution"] = edge_size_dist
        
        # Compute density (for hypergraphs)
        # Density = |E| / (|V| choose k) where k is avg edge size
        if self.nodes and stats["average_edge_size"] > 0:
            k = min(len(self.nodes), int(stats["average_edge_size"]))
            if k > 1:
                from math import comb
                possible_edges = comb(len(self.nodes), k)
                if possible_edges > 0:
                    stats["hypergraph_density"] = len(self.hyperedges) / possible_edges
        
        self.stats.update(stats)
        self.logger.info("Computed hypergraph statistics")
        return stats
    
class TrainableGraphTransformer(nn.Module):
    """
    Trainable Graph Transformer Network for enhanced structural embeddings
    with contrastive learning capability
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int = 4, 
                num_layers: int = 2, dropout: float = 0.1, edge_dim: Optional[int] = None,
                use_layer_norm: bool = True, activation: str = 'gelu'):
        """
        Initialize Graph Transformer Network
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            edge_dim: Edge feature dimension (optional)
            use_layer_norm: Whether to use layer normalization
            activation: Activation function ('relu', 'gelu', or 'silu')
        """
        super(TrainableGraphTransformer, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        if self.hidden_dim % num_heads != 0:
            self.hidden_dim = (self.hidden_dim // num_heads) * num_heads
            self.logger.warning(f"Adjusted hidden_dim to {self.hidden_dim} to be divisible by {num_heads} heads")
        # Initial feature transform
        self.feature_transform = nn.Linear(in_dim, hidden_dim)
        
        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList()
        
        for _ in range(num_layers):
            # Transformer convolution layer
            self.transformer_layers.append(
                TransformerConv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    beta=True  # Enable edge attention
                )
            )
            
            # Layer normalization
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output projection
        self.output_transform = nn.Linear(hidden_dim, out_dim)
        
        # Final normalization and activation
        self.final_layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else None
        
        # Initialize weights
        self._init_weights()
        
        self.logger = logging.getLogger("transformer")
        self.logger.info(f"Initialized Trainable Graph Transformer with {num_layers} layers, {num_heads} heads")
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch assignment for multiple graphs (optional)
            
        Returns:
            Node embeddings [num_nodes, out_dim]
        """
        # Initial feature transform
        h = self.feature_transform(x)
        
        # Apply transformer layers with residual connections
        for i, (transformer, dropout) in enumerate(zip(self.transformer_layers, self.dropouts)):
            # Apply transformer convolution
            transformed = transformer(h, edge_index, edge_attr)
            
            # Residual connection
            h = h + dropout(transformed)
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                h = self.layer_norms[i](h)
            
            # Apply activation except after last layer
            if i < self.num_layers - 1:
                h = self.activation(h)
        
        # Final projection
        out = self.output_transform(h)
        
        # Final layer normalization if enabled
        if self.final_layer_norm is not None:
            out = self.final_layer_norm(out)
        
        return out
    
    def encode_graph(self, data):
        """
        Encode entire graph
        
        Args:
            data: PyTorch Geometric Data object with x, edge_index, etc.
            
        Returns:
            Node embeddings
        """
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        return self.forward(data.x, data.edge_index, edge_attr, batch)
    
    def train_with_contrastive_loss(self, data_loader, num_epochs=100, 
                                  learning_rate=0.001, weight_decay=1e-5,
                                  device=None):
        """
        Train graph transformer using contrastive learning
        
        Args:
            data_loader: DataLoader providing graph data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Compute device (CPU/GPU)
            
        Returns:
            Training losses
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.to(device)
        
        # Set model to training mode
        self.train()
        
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        losses = []
        
        self.logger.info(f"Training graph transformer with contrastive loss for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - get node embeddings
                embeddings = self.encode_graph(batch)
                
                # Sample node pairs for contrastive learning
                # Positive pairs: nodes connected by an edge
                # Negative pairs: nodes not connected
                
                # Get positive pairs from edge_index
                pos_edge_index = batch.edge_index
                
                # Generate negative pairs (nodes not connected by an edge)
                # This is a simple approach - in practice you might want more sophisticated negative sampling
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=embeddings.size(0),
                    num_neg_samples=pos_edge_index.size(1)
                )
                
                # Compute contrastive loss
                
                # Positive scores - dot product of connected nodes
                pos_scores = torch.sum(
                    embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]], 
                    dim=1
                )
                
                # Negative scores - dot product of non-connected nodes
                neg_scores = torch.sum(
                    embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]], 
                    dim=1
                )
                
                # InfoNCE loss
                pos_exp = torch.exp(pos_scores)
                neg_exp = torch.sum(torch.exp(neg_scores))
                
                loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Set model back to evaluation mode
        self.eval()
        
        self.logger.info(f"Graph transformer training completed, final loss: {losses[-1]:.6f}")
        
        return losses

class EnhancedHeterogeneousGraphTransformer(nn.Module):
    """
    Enhanced Heterogeneous Graph Transformer for multi-type entities and relations
    with improved attention mechanism and training capability
    """
    
    def __init__(self, node_types: List[str], edge_types: List[Tuple[str, str, str]], 
                 input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, use_layer_norm: bool = True):
        """
        Initialize Enhanced HGT model
        
        Args:
            node_types: List of node types
            edge_types: List of (src_type, edge_type, dst_type) tuples
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super(EnhancedHeterogeneousGraphTransformer, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        
        # Input embeddings for each node type
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Linear(input_dim, hidden_dim)
            for node_type in self.node_types
        })
        
        # Stack of HGT layers
        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList()
        
        for _ in range(num_layers):
            # HGT convolution layer
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(self.node_types, edge_types),
                    heads=num_heads
                )
            )
            
            # Layer normalization for each node type
            if use_layer_norm:
                self.layer_norms.append(nn.ModuleDict({
                    node_type: nn.LayerNorm(hidden_dim)
                    for node_type in self.node_types
                }))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output projection for each node type
        self.output_projections = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, output_dim)
            for node_type in self.node_types
        })
        
        # Final layer normalization for each node type
        self.final_layer_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(output_dim)
            for node_type in self.node_types
        }) if use_layer_norm else None
        
        # Initialize weights
        self._init_weights()
        
        self.logger = logging.getLogger("transformer")
        self.logger.info(f"Initialized Enhanced Heterogeneous Graph Transformer with {len(node_types)} node types")
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_dict, edge_indices_dict):
        """
        Forward pass
        
        Args:
            x_dict: Dictionary of node features by type
            edge_indices_dict: Dictionary of edge indices by type
            
        Returns:
            Dictionary of node embeddings by type
        """
        # Apply input projections
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_embeddings:
                h_dict[node_type] = self.node_embeddings[node_type](x)
        
        # Apply HGT layers with residual connections
        for i, (hgt_layer, dropout) in enumerate(zip(self.hgt_layers, self.dropouts)):
            # Skip layer if no edge indices exist (edge_indices_dict is empty)
            if not edge_indices_dict:
                continue
                
            # Apply HGT convolution
            h_updated = hgt_layer(h_dict, edge_indices_dict)
            for node_type in h_updated:
                h_updated[node_type] = dropout(h_updated[node_type])
            
            # Apply residual connection, dropout, and layer norm for each node type
            for node_type in h_dict.keys():
                if node_type in h_updated:
                    # Residual connection
                    h_dict[node_type] = h_dict[node_type] + dropout(h_updated[node_type])
                    
                    # Layer normalization if enabled
                    if self.use_layer_norm and node_type in self.layer_norms[i]:
                        h_dict[node_type] = self.layer_norms[i][node_type](h_dict[node_type])
                    
                    # Apply ReLU except after last layer
                    if i < self.num_layers - 1:
                        h_dict[node_type] = F.relu(h_dict[node_type])
        
        # Apply output projections
        out_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.output_projections:
                out = self.output_projections[node_type](h)
                
                # Final layer normalization if enabled
                if self.use_layer_norm and node_type in self.final_layer_norms:
                    out = self.final_layer_norms[node_type](out)
                
                out_dict[node_type] = out
            else:
                out_dict[node_type] = h
        
        return out_dict
    
    def train_with_link_prediction(self, data, num_epochs=100, learning_rate=0.001, 
                                 weight_decay=1e-5, device=None):
        """
        Train heterogeneous graph transformer using link prediction
        
        Args:
            data: HeteroData object containing graph data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Compute device (CPU/GPU)
            
        Returns:
            Training losses
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.to(device)
        
        # Set model to training mode
        self.train()
        
        # Convert HeteroData to device
        data = data.to(device)
        
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Split edges for training
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            edge_types=self.edge_types,
            rev_edge_types=[(dst, edge_type + '_rev', src) for src, edge_type, dst in self.edge_types]
        )
        train_data, val_data, test_data = transform(data)
        
        # Training loop
        losses = []
        
        self.logger.info(f"Training heterogeneous graph transformer for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - get node embeddings
            embeddings_dict = self.forward(
                {node_type: train_data[node_type].x for node_type in self.node_types if hasattr(train_data, node_type)},
                {edge_type: train_data[edge_type].edge_index 
                for edge_type in train_data.edge_types 
                if hasattr(train_data, edge_type) and hasattr(train_data[edge_type], 'edge_index')}
            )
            
            # Initialize loss
            loss = 0.0
            edge_count = 0
            
            # Compute link prediction loss for each edge type
            for src, edge_type, dst in self.edge_types:
                # Skip if source or destination nodes don't have embeddings
                if src not in embeddings_dict or dst not in embeddings_dict:
                    continue
                
                # Get positive edges
                pos_edge_key = (src, edge_type, dst)
                if pos_edge_key not in train_data.edge_types:
                    continue
                    
                pos_edge_index = train_data[pos_edge_key].edge_index
                
                # Get negative edges
                neg_edge_key = (src, f"{edge_type}_neg", dst)
                if neg_edge_key not in train_data.edge_types:
                    continue
                    
                neg_edge_index = train_data[neg_edge_key].edge_index
                
                # Get embeddings
                src_embeddings = embeddings_dict[src]
                dst_embeddings = embeddings_dict[dst]
                
                # Positive scores
                pos_scores = torch.sum(
                    src_embeddings[pos_edge_index[0]] * dst_embeddings[pos_edge_index[1]], 
                    dim=1
                )
                
                # Negative scores
                neg_scores = torch.sum(
                    src_embeddings[neg_edge_index[0]] * dst_embeddings[neg_edge_index[1]], 
                    dim=1
                )
                
                # Binary cross entropy loss
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_scores, 
                    torch.ones_like(pos_scores),
                    reduction='mean'
                )
                
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_scores, 
                    torch.zeros_like(neg_scores),
                    reduction='mean'
                )
                
                edge_loss = (pos_loss + neg_loss) / 2
                loss += edge_loss
                edge_count += 1
            
            # Average loss over edge types
            if edge_count > 0:
                loss = loss / edge_count
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Record loss
                losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        # Set model back to evaluation mode
        self.eval()
        
        self.logger.info(f"Heterogeneous graph transformer training completed")
        
        return losses

class QuantumInspiredAttention(nn.Module):
    """
    Quantum-inspired attention mechanism for graph neural networks with
    learnable parameters
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize quantum-inspired attention
        
        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(QuantumInspiredAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Ensure dimension is divisible by num_heads
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        
        # Linear transformations for quantum-inspired attention
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        
        # Output projection
        self.output_transform = nn.Linear(dim, dim)
        
        # Parameter for attention temperature (learnable)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        
        # Complex phase parameter (learnable)
        self.phase = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(dim)
        
        # Initialize parameters
        self._init_weights()
        
        self.logger = logging.getLogger("quantum")
        self.logger.info(f"Initialized Quantum-Inspired Attention with {num_heads} heads")
    
    def _init_weights(self):
        """Initialize weights with small values"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize temperature to 1 for stable attention
        nn.init.ones_(self.temperature)
        
        # Initialize phase to 0 for classical behavior at first
        nn.init.zeros_(self.phase)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, seq_len, dim]
            mask: Attention mask [batch_size, seq_len] (optional)
            
        Returns:
            Attended features [batch_size, seq_len, dim]
        """
        # Apply layer normalization to input
        x_norm = self.layer_norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        q = self.q_transform(x_norm)
        k = self.k_transform(x_norm)
        v = self.v_transform(x_norm)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Convert to complex numbers with learnable phase
        # In quantum mechanics, states are represented as complex numbers
        q_complex = q * torch.exp(1j * self.phase)
        k_complex = k * torch.exp(-1j * self.phase)  # Conjugated for proper inner product
        
        # Compute attention scores (complex inner product)
        # Normalize by sqrt(d_k) as in the original Transformer
        attention_scores = torch.matmul(q_complex, k_complex.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))
        
        # Take absolute square to get probabilities (quantum measurement)
        attention_probs = torch.abs(attention_scores) ** 2
        
        # Apply temperature scaling
        attention_probs = attention_probs / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            
            # Set masked positions to -inf before softmax
            attention_probs = attention_probs.masked_fill(
                expanded_mask == 0, 
                -1e9
            )
        
        # Apply softmax to get normalized probabilities
        attention_probs = F.softmax(attention_probs, dim=-1)
        
        # Apply dropout to attention probabilities
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_probs, v)
        
        # Reshape and project back to original dimension
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        output = self.output_transform(attended_values)
        
        # Apply dropout
        output = self.output_dropout(output)
        
        # Residual connection
        output = x + output
        
        return output

class AdvancedEmbeddingGenerator:
    """
    Enhanced embedding generator with trainable graph neural networks
    and quantum-inspired techniques
    """
    
    def __init__(self, api_client: EnhancedAPIClient, embeddings_dir: str = "embeddings",
                 hidden_dim: int = 128, output_dim: int = 64, 
                 use_quantum: bool = True, device=None):
        """
        Initialize advanced embedding generator
        
        Args:
            api_client: API client for text embeddings
            embeddings_dir: Directory to store embeddings
            hidden_dim: Hidden dimension for graph neural networks
            output_dim: Final output dimension
            use_quantum: Whether to use quantum-inspired techniques
            device: Device to use for computation
        """
        self.api_client = api_client
        self.embeddings_dir = embeddings_dir
        self.embedding_model = "text-embedding-3-small"
        self.semantic_dim = 1536  # Dimensionality of text-embedding-3-small
        
        # Advanced embedding parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_quantum = use_quantum
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural models
        self.transformer = None
        self.hetero_transformer = None
        self.tensor_network = None if not use_quantum else QuantumInspiredTensorNetwork(
            dim=self.semantic_dim,
            rank=min(32, self.semantic_dim // 4),
            device=self.device
        )
        
        # Quantum attention for embedding fusion
        self.quantum_attention = QuantumInspiredAttention(
            dim=output_dim,
            num_heads=4,
            dropout=0.1
        ).to(self.device)
        
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Create subdirectories for different embedding types
        os.makedirs(os.path.join(embeddings_dir, "semantic"), exist_ok=True)
        os.makedirs(os.path.join(embeddings_dir, "structural"), exist_ok=True)
        os.makedirs(os.path.join(embeddings_dir, "combined"), exist_ok=True)
        os.makedirs(os.path.join(embeddings_dir, "temporal"), exist_ok=True)
        
        self.logger = logging.getLogger("embeddings")
        self.logger.info(f"Initialized Advanced Embedding Generator (device: {self.device})")
    
    def _get_embedding_cache_path(self, entity_type: str, entity_id: str, embedding_type: str = "semantic") -> str:
        """Get file path for cached embedding"""
        sanitized_id = entity_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.embeddings_dir, embedding_type, f"{entity_type}_{sanitized_id}.npy")
    
    def _save_embedding(self, entity_type: str, entity_id: str, embedding: np.ndarray, embedding_type: str = "semantic"):
        """Save embedding to cache"""
        file_path = self._get_embedding_cache_path(entity_type, entity_id, embedding_type)
        np.save(file_path, embedding)
        self.logger.debug(f"Saved {embedding_type} embedding for {entity_type} {entity_id}")
    
    def _load_embedding(self, entity_type: str, entity_id: str, embedding_type: str = "semantic") -> Optional[np.ndarray]:
        """Load embedding from cache if available"""
        file_path = self._get_embedding_cache_path(entity_type, entity_id, embedding_type)
        
        if os.path.exists(file_path):
            try:
                embedding = np.load(file_path)
                self.logger.debug(f"Loaded cached {embedding_type} embedding for {entity_type} {entity_id}")
                return embedding
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        return None
    
    def _create_entity_description(self, entity_data: Dict, entity_type: str) -> str:
        """Create rich text description for entity embedding generation"""
        if entity_type.lower() in ["movie", "film"]:
            return self._create_movie_description(entity_data)
        elif entity_type.lower() in ["actor", "director", "writer", "music_director", "person"]:
            return self._create_person_description(entity_data)
        elif entity_type.lower() == "user":
            return self._create_user_description(entity_data)
        elif entity_type.lower() == "genre":
            return self._create_genre_description(entity_data)
        else:
            # Generic fallback for other entity types
            name = entity_data.get("name", entity_data.get("title", ""))
            description = f"{entity_type}: {name}. "
            
            # Add other attributes
            for key, value in entity_data.items():
                if key not in ["name", "title", "_metadata"] and value:
                    if isinstance(value, (list, tuple)):
                        value_str = ", ".join(str(v) for v in value[:5])
                        description += f"{key.replace('_', ' ')}: {value_str}. "
                    elif isinstance(value, (str, int, float)):
                        description += f"{key.replace('_', ' ')}: {value}. "
            
            return description
    
    def _create_movie_description(self, movie_data: Dict) -> str:
        """Create rich text description for movie embedding generation"""
        title = movie_data.get("title", "")
        year = movie_data.get("year", "")
        director = movie_data.get("director", "")
        plot = movie_data.get("plot", "")
        
        # Get primary genres and themes
        genres = movie_data.get("genres", [])
        if isinstance(genres, str):
            try:
                genres = json.loads(genres)
            except:
                genres = [g.strip() for g in genres.split(",") if g.strip()]
        genres_str = ", ".join(genres[:5]) if genres else ""
        
        themes = movie_data.get("themes", [])
        if isinstance(themes, str):
            try:
                themes = json.loads(themes)
            except:
                themes = [t.strip() for t in themes.split(",") if t.strip()]
        themes_str = ", ".join(themes[:5]) if themes else ""
        
        # Get main cast
        cast = movie_data.get("cast", [])
        if isinstance(cast, list):
            cast_str = ", ".join([actor.get("name", "") for actor in cast[:5]]) if cast else ""
        else:
            cast_str = ""
        
        # Get cultural impact and reception
        cultural_impact = movie_data.get("cultural_impact", "")
        reception = movie_data.get("critical_reception", "")
        
        # Combine into rich description
        description = f"Title: {title}. "
        
        if year:
            description += f"Year: {year}. "
        
        if director:
            description += f"Directed by {director}. "
        
        if genres_str:
            description += f"Genres: {genres_str}. "
        
        if cast_str:
            description += f"Starring: {cast_str}. "
        
        if plot:
            description += f"Plot: {plot} "
        
        if themes_str:
            description += f"Themes: {themes_str}. "
        
        if cultural_impact:
            description += f"Cultural impact: {cultural_impact} "
        
        if reception:
            description += f"Reception: {reception}"
        
        return description
    
    def _create_person_description(self, person_data: Dict) -> str:
        """Create rich text description for person embedding generation"""
        name = person_data.get("name", "")
        role = person_data.get("role", "")
        bio = person_data.get("biography", "")
        
        # Known for works
        known_for = person_data.get("known_for", [])
        if isinstance(known_for, str):
            try:
                known_for = json.loads(known_for)
            except:
                known_for = [k.strip() for k in known_for.split(",") if k.strip()]
        known_for_str = ", ".join(known_for[:5]) if known_for else ""
        
        # Style and characteristics
        style = person_data.get("signature_style", "")
        
        # Career highlights
        highlights = person_data.get("career_highlights", [])
        if isinstance(highlights, str):
            try:
                highlights = json.loads(highlights)
            except:
                highlights = [h.strip() for h in highlights.split(",") if h.strip()]
        highlights_str = ", ".join(highlights[:3]) if highlights else ""
        
        # Influence
        influence = person_data.get("influence", "")
        
        # Combine into rich description
        description = f"Name: {name}. "
        
        if role:
            description += f"Role: {role}. "
        
        if bio:
            description += f"Biography: {bio} "
        
        if known_for_str:
            description += f"Known for: {known_for_str}. "
        
        if style:
            description += f"Style: {style}. "
        
        if highlights_str:
            description += f"Career highlights: {highlights_str}. "
        
        if influence:
            description += f"Influence: {influence}"
        
        return description
    
    def _create_user_description(self, user_data: Dict) -> str:
        """Create rich text description for user embedding generation"""
        user_id = user_data.get("user_id", "")
        gender = user_data.get("gender", "")
        job = user_data.get("job", "")
        state = user_data.get("state", "")
        languages = user_data.get("languages", [])
        
        if isinstance(languages, str):
            try:
                languages = json.loads(languages)
            except:
                languages = [lang.strip() for lang in languages.split(",") if lang.strip()]
                
        lang_str = ", ".join(languages) if languages else ""
        
        # Get persona information if available
        persona = user_data.get("persona", {})
        pref_profile = persona.get("preference_profile", {})
        behavior = persona.get("behavior_insights", {})
        
        # Get ratings information
        ratings = user_data.get("ratings", [])
        liked_movies = [movie.get("title", "") for movie in ratings 
                       if movie.get("rating") == "1" or movie.get("rating") == 1]
        disliked_movies = [movie.get("title", "") for movie in ratings 
                          if movie.get("rating") == "-1" or movie.get("rating") == -1]
        
        # Combine into rich description
        description = f"User ID: {user_id}. "
        
        if gender:
            description += f"Gender: {gender}. "
        
        if job:
            description += f"Occupation: {job}. "
        
        if state:
            description += f"State: {state}. "
        
        if lang_str:
            description += f"Languages: {lang_str}. "
        
        # Add favorite genres from persona
        fav_genres = pref_profile.get("favorite_genres", [])
        if fav_genres:
            description += f"Favorite genres: {', '.join(fav_genres[:5])}. "
        
        # Add disliked genres from persona
        disliked_genres = pref_profile.get("disliked_genres", [])
        if disliked_genres:
            description += f"Disliked genres: {', '.join(disliked_genres[:3])}. "
        
        # Add rating patterns
        rating_patterns = behavior.get("rating_patterns", "")
        if rating_patterns:
            description += f"Rating patterns: {rating_patterns}. "
        
        # Add example liked movies
        if liked_movies:
            description += f"Liked movies: {', '.join(liked_movies[:5])}. "
        
        # Add example disliked movies
        if disliked_movies:
            description += f"Disliked movies: {', '.join(disliked_movies[:3])}. "
        
        return description
    
    def _create_genre_description(self, genre_data: Dict) -> str:
        """Create rich text description for genre embedding generation"""
        genre_name = genre_data.get("genre", "")
        definition = genre_data.get("definition", "")
        
        # Key characteristics
        characteristics = genre_data.get("key_characteristics", [])
        if isinstance(characteristics, str):
            try:
                characteristics = json.loads(characteristics)
            except:
                characteristics = [c.strip() for c in characteristics.split(",") if c.strip()]
        char_str = ", ".join(characteristics[:5]) if characteristics else ""
        
        # Common tropes
        tropes = genre_data.get("common_tropes", [])
        if isinstance(tropes, str):
            try:
                tropes = json.loads(tropes)
            except:
                tropes = [t.strip() for t in tropes.split(",") if t.strip()]
        tropes_str = ", ".join(tropes[:5]) if tropes else ""
        
        # Cultural impact
        cultural_impact = genre_data.get("cultural_impact", "")
        
        # Recurring themes
        themes = genre_data.get("recurring_themes", [])
        if isinstance(themes, str):
            try:
                themes = json.loads(themes)
            except:
                themes = [t.strip() for t in themes.split(",") if t.strip()]
        themes_str = ", ".join(themes[:5]) if themes else ""
        
        # Combine into rich description
        description = f"Genre: {genre_name}. "
        
        if definition:
            description += f"Definition: {definition}. "
        
        if char_str:
            description += f"Key characteristics: {char_str}. "
        
        if tropes_str:
            description += f"Common tropes: {tropes_str}. "
        
        if cultural_impact:
            description += f"Cultural impact: {cultural_impact}. "
        
        if themes_str:
            description += f"Recurring themes: {themes_str}. "
        
        return description
    
    async def generate_semantic_embedding(self, entity_data: Dict, entity_type: str, 
                                      entity_id: str, force_refresh: bool = False) -> np.ndarray:
        """Generate semantic (text-based) embedding for an entity"""
        # Check cache unless forced refresh
        if not force_refresh:
            cached_embedding = self._load_embedding(entity_type, entity_id, "semantic")
            if cached_embedding is not None:
                return cached_embedding
        
        # Create rich description
        description = self._create_entity_description(entity_data, entity_type)
        
        self.logger.info(f"Generating semantic embedding for {entity_type} {entity_id}")
        self.logger.debug(f"Description: {description[:100]}...")
        
        # Call API to get embedding
        try:
            response = await self.api_client.embedding(
                model=self.embedding_model,
                input_text=description
            )
            
            if "error" in response:
                self.logger.error(f"Error generating embedding: {response['error']}")
                # Return zero embedding as fallback
                return np.zeros(self.semantic_dim)
            
            embedding = np.array(response["embeddings"])
            
            # Validate embedding
            if not np.all(np.isfinite(embedding)):
                self.logger.warning(f"Embedding contains non-finite values, fixing...")
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Cache embedding
            self._save_embedding(entity_type, entity_id, embedding, "semantic")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Exception generating embedding: {str(e)}")
            traceback.print_exc()
            # Return zero embedding as fallback
            return np.zeros(self.semantic_dim)
    
    async def generate_semantic_embeddings_efficiently(self, entities: List[Tuple[Dict, str, str]], 
                                                 batch_size: int = 200) -> Dict[str, np.ndarray]:
        """Generate semantic embeddings efficiently in optimized batches"""
        self.logger.info(f"Efficiently generating semantic embeddings for {len(entities)} entities")
        
        results = {}
        batch_entity_ids = []
        batch_descriptions = []
        batch_entity_data = []
        batches = []
        
        # Prepare batches
        for i, (entity_data, entity_type, entity_id) in enumerate(entities):
            # Check cache first
            cached_embedding = self._load_embedding(entity_type, entity_id, "semantic")
            if cached_embedding is not None:
                results[entity_id] = cached_embedding
                continue
                
            # Create description
            description = self._create_entity_description(entity_data, entity_type)
            
            # Add to current batch
            batch_entity_ids.append(entity_id)
            batch_descriptions.append(description)
            batch_entity_data.append((entity_data, entity_type, entity_id))
            
            # When batch is full or at the end, add it to batches
            if len(batch_descriptions) == batch_size or i == len(entities) - 1:
                if batch_descriptions:  # Only add non-empty batches
                    batches.append((batch_entity_ids.copy(), batch_descriptions.copy(), batch_entity_data.copy()))
                    batch_entity_ids = []
                    batch_descriptions = []
                    batch_entity_data = []
        
        # Process batches with tqdm for progress tracking
        for batch_idx, (entity_ids, descriptions, entity_data_list) in enumerate(async_tqdm(batches, desc="Processing embedding batches")):
            try:
                if not descriptions:
                    continue
                    
                # Get embeddings for batch
                response = await self.api_client.embedding(
                    model=self.embedding_model,
                    input_text=descriptions
                )
                
                if "error" in response:
                    self.logger.error(f"Error generating batch embeddings: {response['error']}")
                    continue
                    
                # Process response
                embeddings = response["embeddings"]
                
                # Match embeddings to entities
                for i, (entity_id, embedding) in enumerate(zip(entity_ids, embeddings)):
                    # Cache embedding
                    entity_data, entity_type, _ = entity_data_list[i]
                    
                    embedding_array = np.array(embedding)
                    
                    # Validate embedding
                    if not np.all(np.isfinite(embedding_array)):
                        self.logger.warning(f"Embedding for {entity_id} contains non-finite values, fixing...")
                        embedding_array = np.nan_to_num(embedding_array, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    results[entity_id] = embedding_array
                    self._save_embedding(entity_type, entity_id, embedding_array, "semantic")
                    
            except Exception as e:
                self.logger.error(f"Exception in batch {batch_idx}: {str(e)}")
                traceback.print_exc()
        
        self.logger.info(f"Efficiently generated {len(results)} semantic embeddings")
        return results
    
    def initialize_graph_transformer(self, input_dim: int, nodes_by_type: Dict[str, List[str]], 
                                   heterogeneous: bool = True):
        """
        Initialize appropriate graph transformer based on graph structure
        
        Args:
            input_dim: Input dimension (from semantic embeddings)
            nodes_by_type: Dictionary of node_type -> list of node_ids
            heterogeneous: Whether to use heterogeneous model or homogeneous
        """
        try:
            device = self.device
            
            if heterogeneous:
                # Create node types list
                node_types = list(nodes_by_type.keys())
                
                # Create edge types (will be updated with actual graph connections)
                edge_types = []
                for src_type in node_types:
                    for dst_type in node_types:
                        # Add general edge type between these types
                        edge_types.append((src_type, f"{src_type}_to_{dst_type}", dst_type))
                
                # Initialize heterogeneous transformer
                self.hetero_transformer = EnhancedHeterogeneousGraphTransformer(
                    node_types=node_types,
                    edge_types=edge_types,
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    num_heads=4,
                    num_layers=2,
                    dropout=0.1,
                    use_layer_norm=True
                ).to(device)
                
                self.logger.info(f"Initialized heterogeneous graph transformer with {len(node_types)} node types")
                
            else:
                # Initialize homogeneous transformer
                self.transformer = TrainableGraphTransformer(
                    in_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.output_dim,
                    num_heads=4,
                    num_layers=2,
                    dropout=0.1,
                    use_layer_norm=True
                ).to(device)
                
                self.logger.info(f"Initialized homogeneous graph transformer")
        except Exception as e:
            self.logger.error(f"Error initializing graph transformer: {str(e)}")
            traceback.print_exc()
    
    def generate_structural_embeddings(self, graph: nx.Graph, semantic_embeddings: Dict[str, np.ndarray], 
                                     nodes_by_type: Dict[str, List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Generate structural embeddings using graph neural networks
        
        Args:
            graph: NetworkX graph with nodes and edges
            semantic_embeddings: Dictionary of node_id -> semantic embedding
            nodes_by_type: Optional dictionary of node_type -> list of node_ids
            
        Returns:
            Dictionary of node_id -> structural embedding
        """
        self.logger.info(f"Generating structural embeddings for {len(graph.nodes())} nodes")
        
        # Prepare node features (from semantic embeddings)
        print(type(nodes_by_type))
        heterogeneous = nodes_by_type is not None
        
        try:
            # Import PyTorch Geometric
            import torch_geometric
            from torch_geometric.data import Data, HeteroData
            from torch_geometric.utils import from_networkx
            
            # If semantic embeddings have different dimensions, reduce to smallest
            embedding_dims = [emb.shape[0] for emb in semantic_embeddings.values() if hasattr(emb, 'shape')]
            
            if not embedding_dims:
                self.logger.error("No valid embeddings found")
                return semantic_embeddings
                
            min_dim = min(embedding_dims)
            
            device = self.device
            
            if heterogeneous:
                # Initialize heterogeneous transformer model if needed
                self.initialize_graph_transformer(min_dim, nodes_by_type, heterogeneous=True)
                
                if self.hetero_transformer is None:
                    # Fallback to homogeneous model if initialization failed
                    self.logger.warning("Heterogeneous transformer initialization failed, falling back to homogeneous model")
                    return self.generate_structural_embeddings(graph, semantic_embeddings)
                
                # Create heterogeneous graph data
                data = HeteroData()
                
                # Add node features by type
                node_indices = {}  # Map from node_id to index in type-specific tensors
                
                for node_type, nodes in nodes_by_type.items():
                    # Filter to nodes that have embeddings
                    nodes_with_emb = [n for n in nodes if n in semantic_embeddings]
                    
                    if not nodes_with_emb:
                        continue
                    
                    # Store mapping from node_id to index
                    node_indices[node_type] = {node_id: i for i, node_id in enumerate(nodes_with_emb)}
                    
                    # Create features tensor
                    features = []
                    for node_id in nodes_with_emb:
                        # Get embedding and ensure proper dimensions
                        embedding = semantic_embeddings[node_id]
                        
                        if len(embedding) > min_dim:
                            embedding = embedding[:min_dim]
                        elif len(embedding) < min_dim:
                            embedding = np.pad(embedding, (0, min_dim - len(embedding)), mode='constant')
                        
                        features.append(torch.tensor(embedding, dtype=torch.float))
                    
                    if features:
                        # Stack and move to device
                        node_features = torch.stack(features).to(device)
                        
                        # Add to data object
                        data[node_type].x = node_features
                        data[node_type].node_ids = nodes_with_emb
                
                # Add edges by type
                edge_types_found = set()
                
                for u, v, edge_data in graph.edges(data=True):
                    # Get node types
                    u_type = graph.nodes[u].get("node_type", "unknown")
                    v_type = graph.nodes[v].get("node_type", "unknown")
                    
                    # Skip if node types not in our data
                    if u_type not in node_indices or v_type not in node_indices:
                        continue
                        
                    # Skip if nodes not in indices
                    if u not in node_indices[u_type] or v not in node_indices[v_type]:
                        continue
                    
                    # Determine edge type from edge_data
                    edge_type = edge_data.get("edge_type", f"{u_type}_to_{v_type}")
                    
                    # Create edge type key
                    edge_key = (u_type, edge_type, v_type)
                    edge_types_found.add(edge_key)
                    
                    # Get node indices
                    u_idx = node_indices[u_type][u]
                    v_idx = node_indices[v_type][v]
                    
                    # Initialize edge index if needed
                    if edge_key not in data.edge_types:
                        data[edge_key].edge_index = torch.tensor([[],[]], dtype=torch.long).to(device)
                    
                    # Add edge to existing index
                    current_edge_index = data[edge_key].edge_index
                    new_edge = torch.tensor([[u_idx], [v_idx]], dtype=torch.long).to(device)
                    data[edge_key].edge_index = torch.cat([current_edge_index, new_edge], dim=1)
                
                # Process graphs with no edges (create dummy edges if needed)
                if not edge_types_found:
                    self.logger.warning("No valid edges found in heterogeneous graph")
                    # Create dummy edges within node types to avoid empty edge errors
                    for node_type, nodes in node_indices.items():
                        if len(nodes) > 1:
                            # Create self-loops for first node
                            edge_key = (node_type, f"{node_type}_to_{node_type}", node_type)
                            data[edge_key].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
                            edge_types_found.add(edge_key)
                
                # Update edge_types in transformer if needed
                if self.hetero_transformer.edge_types != list(edge_types_found):
                    self.logger.info(f"Updating transformer edge types to match graph: {len(edge_types_found)} types")
                    # Re-initialize with correct edge types
                    self.hetero_transformer = EnhancedHeterogeneousGraphTransformer(
                        node_types=list(node_indices.keys()),
                        edge_types=list(edge_types_found),
                        input_dim=min_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=self.output_dim,
                        num_heads=4,
                        num_layers=2,
                        dropout=0.1,
                        use_layer_norm=True
                    ).to(device)
                
                # Generate embeddings using HGT model
                with torch.no_grad():
                    # Check if we have any node types with features
                    node_types_with_features = [nt for nt in data.node_types if hasattr(data[nt], 'x')]
                    
                    if not node_types_with_features:
                        self.logger.error("No node types with features found")
                        return semantic_embeddings
                    
                    # Prepare feature dictionary for model input
                    features_dict = {nt: data[nt].x for nt in node_types_with_features}
                    
                    # Check edge indices for empty tensors and fix
                    edge_dict = {}
                    for edge_key in data.edge_types:
                        if hasattr(data[edge_key], 'edge_index') and data[edge_key].edge_index.numel() > 0:
                            # Ensure edge index has correct shape [2, num_edges]
                            if data[edge_key].edge_index.shape[0] != 2:
                                data[edge_key].edge_index = data[edge_key].edge_index.t()
                            edge_dict[edge_key] = data[edge_key].edge_index
                    
                    # Forward pass
                    embeddings_dict = self.hetero_transformer(features_dict, edge_dict)
                
                # Convert back to dictionary by node ID
                structural_embeddings = {}
                
                for node_type in node_types_with_features:
                    if node_type in embeddings_dict:
                        node_ids = data[node_type].node_ids
                        node_embeds = embeddings_dict[node_type].cpu().numpy()
                        
                        for i, node_id in enumerate(node_ids):
                            try:
                                structural_embeddings[node_id] = node_embeds[i]
                            except IndexError:
                                self.logger.warning(f"Index error for node {node_id}, type {node_type}: {i} >= {len(node_embeds)}")
                                # Use semantic embedding as fallback
                                if node_id in semantic_embeddings:
                                    sem_emb = semantic_embeddings[node_id]
                                    if len(sem_emb) > self.output_dim:
                                        sem_emb = sem_emb[:self.output_dim]
                                    elif len(sem_emb) < self.output_dim:
                                        sem_emb = np.pad(sem_emb, (0, self.output_dim - len(sem_emb)), mode='constant')
                                    structural_embeddings[node_id] = sem_emb
                
                # For nodes without structural embeddings, use processed semantic embeddings
                for node_id, embedding in semantic_embeddings.items():
                    if node_id not in structural_embeddings:
                        # Resize semantic embedding to match output dimension
                        if len(embedding) > self.output_dim:
                            embedding = embedding[:self.output_dim]
                        elif len(embedding) < self.output_dim:
                            embedding = np.pad(embedding, (0, self.output_dim - len(embedding)), mode='constant')
                        structural_embeddings[node_id] = embedding
            else:
                # Homogeneous graph processing
                
                # Initialize transformer model if needed
                self.initialize_graph_transformer(min_dim, heterogeneous=False)
                
                if self.transformer is None:
                    self.logger.error("Failed to initialize transformer model")
                    return semantic_embeddings
                
                # Create feature mapping for all nodes
                node_lookup = {n: i for i, n in enumerate(graph.nodes())}
                idx_to_node = {i: n for n, i in node_lookup.items()}
                
                # Create node feature matrix
                x = np.zeros((len(graph.nodes()), min_dim))
                
                # Fill in features for nodes with embeddings
                for node_id, embedding in semantic_embeddings.items():
                    if node_id in node_lookup:
                        if len(embedding) >= min_dim:
                            x[node_lookup[node_id]] = embedding[:min_dim]
                        else:
                            x[node_lookup[node_id]] = np.pad(embedding, (0, min_dim - len(embedding)), mode='constant')
                
                # Convert to torch tensor
                x = torch.tensor(x, dtype=torch.float32).to(device)
                
                # Create edge index
                edge_index = []
                for u, v in graph.edges():
                    edge_index.append([node_lookup[u], node_lookup[v]])
                    edge_index.append([node_lookup[v], node_lookup[u]])  # Add reverse edge for undirected graph
                
                # Handle graphs with no edges by adding self-loops
                if not edge_index and len(node_lookup) > 0:
                    # Add self-loops for first node
                    first_node_idx = 0
                    edge_index.append([first_node_idx, first_node_idx])
                
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
                
                # Create PyG data object
                data = Data(x=x, edge_index=edge_index)
                
                # Generate embeddings using transformer
                with torch.no_grad():
                    out = self.transformer(data.x, data.edge_index)
                
                # Convert back to dictionary by node ID
                out_np = out.cpu().numpy()
                structural_embeddings = {idx_to_node[i]: out_np[i] for i in range(len(out_np))}
            
            # Cache structural embeddings
            for node_id, embedding in structural_embeddings.items():
                try:
                    node_type = graph.nodes[node_id].get("node_type", "unknown")
                    self._save_embedding(node_type, node_id, embedding, "structural")
                except Exception as e:
                    self.logger.warning(f"Error saving structural embedding for {node_id}: {str(e)}")
            
            self.logger.info(f"Generated structural embeddings for {len(structural_embeddings)} nodes")
            return structural_embeddings
            
        except ImportError:
            self.logger.error("PyTorch Geometric not available, falling back to semantic embeddings")
            return semantic_embeddings
        except Exception as e:
            self.logger.error(f"Error generating structural embeddings: {str(e)}")
            traceback.print_exc()
            # Fallback to semantic embeddings
            return semantic_embeddings
    
    def combine_embeddings(self, semantic_embeddings: Dict[str, np.ndarray], 
                         structural_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Combine semantic and structural embeddings using advanced fusion
        
        Args:
            semantic_embeddings: Dictionary of node_id -> semantic embedding
            structural_embeddings: Dictionary of node_id -> structural embedding
            
        Returns:
            Dictionary of node_id -> combined embedding
        """
        self.logger.info(f"Combining embeddings for {len(semantic_embeddings)} nodes")
        
        # Only process nodes that have both embeddings
        common_nodes = set(semantic_embeddings.keys()) & set(structural_embeddings.keys())
        
        # Prepare scaled and resized embeddings for fusion
        scaled_semantic = {}
        scaled_structural = {}
        
        for node_id in common_nodes:
            # Get and resize semantic embedding
            sem_emb = semantic_embeddings[node_id]
            if len(sem_emb) > self.output_dim:
                # Use PCA or truncation for dimension reduction
                sem_emb = sem_emb[:self.output_dim]
            elif len(sem_emb) < self.output_dim:
                sem_emb = np.pad(sem_emb, (0, self.output_dim - len(sem_emb)), mode='constant')
            
            # Scale semantic embedding
            sem_norm = np.linalg.norm(sem_emb)
            if sem_norm > 0:
                sem_emb = sem_emb / sem_norm
            scaled_semantic[node_id] = sem_emb
            
            # Get and resize structural embedding
            str_emb = structural_embeddings[node_id]
            if len(str_emb) > self.output_dim:
                str_emb = str_emb[:self.output_dim]
            elif len(str_emb) < self.output_dim:
                str_emb = np.pad(str_emb, (0, self.output_dim - len(str_emb)), mode='constant')
            
            # Scale structural embedding
            str_norm = np.linalg.norm(str_emb)
            if str_norm > 0:
                str_emb = str_emb / str_norm
            scaled_structural[node_id] = str_emb
        
        # For quantum-inspired approach, we use quantum attention
        if self.use_quantum and self.quantum_attention is not None:
            try:
                # Prepare batch of embeddings for quantum attention fusion
                semantic_batch = torch.tensor(
                    np.stack([scaled_semantic[node] for node in common_nodes]), 
                    dtype=torch.float32,
                    device=self.device
                )
                
                structural_batch = torch.tensor(
                    np.stack([scaled_structural[node] for node in common_nodes]), 
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Concatenate embeddings for attention context
                combined_input = torch.cat([
                    semantic_batch.unsqueeze(1),   # [nodes, 1, dim]
                    structural_batch.unsqueeze(1)  # [nodes, 1, dim]
                ], dim=1)  # Result: [nodes, 2, dim]
                
                # Apply quantum attention fusion
                with torch.no_grad():
                    # Apply attention to combine embeddings
                    fused_embeddings = self.quantum_attention(combined_input)
                    
                    # Extract output and normalize
                    combined_embeddings_tensor = fused_embeddings.mean(dim=1)  # Average over sequence dimension
                    combined_embeddings_tensor = F.normalize(combined_embeddings_tensor, p=2, dim=1)
                
                # Convert to numpy and create dictionary
                combined_embeddings_np = combined_embeddings_tensor.cpu().numpy()
                combined_embeddings = {
                    node_id: combined_embeddings_np[i] 
                    for i, node_id in enumerate(common_nodes)
                }
                
                # Cache combined embeddings
                for node_id, embedding in combined_embeddings.items():
                    node_type = "unknown"  # We don't have graph here to determine type
                    self._save_embedding(node_type, node_id, embedding, "combined")
                
                self.logger.info(f"Combined embeddings using quantum attention for {len(combined_embeddings)} nodes")
                return combined_embeddings
                
            except Exception as e:
                self.logger.error(f"Error in quantum combining: {str(e)}")
                traceback.print_exc()
                # Fall back to standard method
        
        # Standard approach: weighted average with learned weight
        try:
            # Initialize weight parameter (favor semantic slightly)
            semantic_weight = 0.6
            
            combined_embeddings = {}
            
            for node_id in common_nodes:
                semantic = scaled_semantic[node_id]
                structural = scaled_structural[node_id]
                
                # Weighted combination
                combined = semantic_weight * semantic + (1 - semantic_weight) * structural
                
                # Normalize
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined = combined / norm
                
                combined_embeddings[node_id] = combined
            
            # Cache combined embeddings
            for node_id, embedding in combined_embeddings.items():
                node_type = "unknown"  # We don't have graph here to determine type
                self._save_embedding(node_type, node_id, embedding, "combined")
            
            self.logger.info(f"Combined embeddings using standard method for {len(combined_embeddings)} nodes")
            return combined_embeddings
            
        except Exception as e:
            self.logger.error(f"Error combining embeddings: {str(e)}")
            traceback.print_exc()
            
            # Fallback to semantic embeddings
            return {node: semantic_embeddings[node] for node in common_nodes}


# Enhanced Community Detection and Path-Based Reasoning

class PathReasoning:
    """Path-based reasoning for knowledge graph with enhanced explanation capabilities"""
    
    def __init__(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray] = None):
        """
        Initialize path reasoning
        
        Args:
            graph: NetworkX graph
            embeddings: Optional node embeddings for similarity-based reasoning
        """
        self.graph = graph
        self.embeddings = embeddings
        self.logger = logging.getLogger("graph_builder")
        self.logger.info("Initialized Enhanced Path Reasoning module")
        
        # Cache to store previously computed paths
        self.path_cache = {}
        
    def find_paths(self, start_node: str, end_node: str, max_length: int = 3) -> List[List[str]]:
        """
        Find all paths between two nodes up to a maximum length
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        # Check cache first
        cache_key = (start_node, end_node, max_length)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if start_node not in self.graph or end_node not in self.graph:
            return []
        
        try:
            paths = []
            
            # For short paths, use all simple paths
            if max_length <= 3:
                paths = list(nx.all_simple_paths(self.graph, start_node, end_node, cutoff=max_length))
            else:
                # For longer paths, use bidirectional search
                # This is more efficient for large graphs
                
                # Forward search from start node
                forward_paths = {start_node: [[start_node]]}
                
                # Backward search from end node
                backward_paths = {end_node: [[end_node]]}
                
                # Perform bidirectional BFS
                for length in range(1, max_length // 2 + 2):  # +2 to ensure we explore enough levels
                    # Expand forward paths
                    new_forward = {}
                    for node, node_paths in forward_paths.items():
                        for neighbor in self.graph.neighbors(node):
                            # Avoid cycles
                            if any(neighbor in path for path in node_paths):
                                continue
                                
                            # Create new paths with this neighbor
                            new_paths = [path + [neighbor] for path in node_paths]
                            
                            # Check if we can connect to backward paths
                            if neighbor in backward_paths:
                                # Connect forward and backward paths
                                for f_path in new_paths:
                                    for b_path in backward_paths[neighbor]:
                                        # Combine paths (reverse backward path)
                                        combined_path = f_path[:-1] + b_path[::-1]
                                        if len(combined_path) <= max_length + 1:  # +1 because we count nodes, not edges
                                            paths.append(combined_path)
                            
                            # Continue BFS
                            if len(new_paths[0]) < max_length + 1:  # +1 because we count nodes, not edges
                                if neighbor not in new_forward:
                                    new_forward[neighbor] = []
                                new_forward[neighbor].extend(new_paths)
                    
                    # Expand backward paths
                    new_backward = {}
                    for node, node_paths in backward_paths.items():
                        for neighbor in self.graph.neighbors(node):
                            # Avoid cycles
                            if any(neighbor in path for path in node_paths):
                                continue
                                
                            # Create new paths with this neighbor
                            new_paths = [path + [neighbor] for path in node_paths]
                            
                            # Check if we can connect to forward paths
                            if neighbor in forward_paths:
                                # Connect forward and backward paths
                                for b_path in new_paths:
                                    for f_path in forward_paths[neighbor]:
                                        # Combine paths (reverse backward path)
                                        combined_path = f_path + b_path[1::][::-1]
                                        if len(combined_path) <= max_length + 1:  # +1 because we count nodes, not edges
                                            paths.append(combined_path)
                            
                            # Continue BFS
                            if len(new_paths[0]) < max_length + 1:  # +1 because we count nodes, not edges
                                if neighbor not in new_backward:
                                    new_backward[neighbor] = []
                                new_backward[neighbor].extend(new_paths)
                    
                    # Update path collections
                    forward_paths.update(new_forward)
                    backward_paths.update(new_backward)
                    
                    # Early stopping if we found paths
                    if paths and length >= max_length // 2:
                        break
                
                # If bidirectional search didn't find paths, fall back to simple paths
                if not paths:
                    # For fallback, use a smaller cutoff to avoid combinatorial explosion
                    fallback_cutoff = min(max_length, 4)
                    paths = list(nx.all_simple_paths(self.graph, start_node, end_node, cutoff=fallback_cutoff))
            
            # Sort paths by length (shortest first)
            paths.sort(key=len)
            
            # Limit to 10 paths to avoid overwhelming the user
            if len(paths) > 10:
                paths = paths[:10]
            
            # Cache results
            self.path_cache[cache_key] = paths
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding paths: {str(e)}")
            traceback.print_exc()
            return []
    
    def paths_to_text(self, paths: List[List[str]]) -> List[str]:
        """
        Convert paths to human-readable text
        
        Args:
            paths: List of paths, where each path is a list of node IDs
            
        Returns:
            List of path descriptions
        """
        descriptions = []
        
        for path in paths:
            # Skip empty paths
            if not path or len(path) < 2:
                continue
                
            # Build path description
            description = []
            
            for i in range(len(path) - 1):
                src_node = path[i]
                dst_node = path[i+1]
                
                # Get node names
                src_name = self.graph.nodes[src_node].get("name", self.graph.nodes[src_node].get("title", src_node))
                dst_name = self.graph.nodes[dst_node].get("name", self.graph.nodes[dst_node].get("title", dst_node))
                
                # Get edge attributes if exists
                if self.graph.has_edge(src_node, dst_node):
                    edge_data = self.graph.get_edge_data(src_node, dst_node)
                    edge_type = edge_data.get("edge_type", "connected to")
                    
                    # Format edge type to be more readable
                    edge_type = edge_type.lower().replace("_", " ")
                else:
                    edge_type = "connected to"
                
                # Add to description
                if i == 0:
                    description.append(f"{src_name} {edge_type} {dst_name}")
                else:
                    description.append(f"{edge_type} {dst_name}")
            
            # Join all steps
            path_text = " → ".join(description)
            
            # Include additional information like genres for movies
            if "movie" in self.graph.nodes[path[-1]].get("node_type", "").lower():
                movie = path[-1]
                genre_str = ""
                
                # Get genres from node attributes
                genres = self.graph.nodes[movie].get("genres", "[]")
                if isinstance(genres, str):
                    try:
                        parsed_genres = json.loads(genres)
                        if parsed_genres:
                            genre_str = f" (Genres: {', '.join(parsed_genres[:3])})"
                    except:
                        pass
                
                if genre_str:
                    path_text += genre_str
            
            descriptions.append(path_text)
        
        return descriptions
    
    def explain_recommendation(self, user_id: str, movie_id: str, max_paths: int = 3) -> List[str]:
        """
        Explain recommendation using path-based reasoning with enhanced explanations
        
        Args:
            user_id: User node ID
            movie_id: Movie node ID
            max_paths: Maximum number of paths to return
            
        Returns:
            List of explanation paths
        """
        explanations = []
        
        try:
            # First, try to find direct paths
            direct_paths = self.find_paths(user_id, movie_id, 3)
            
            if direct_paths:
                # Convert to text
                path_texts = self.paths_to_text(direct_paths[:max_paths])
                explanations.extend(path_texts)
            
            # If not enough paths, try indirect reasoning through preferences
            if len(explanations) < max_paths:
                # Get user's genre preferences
                genre_paths = []
                
                for neighbor in self.graph.neighbors(user_id):
                    # Check if this is a genre neighbor
                    if self.graph.nodes[neighbor].get("node_type") == "genre":
                        # Find paths from this genre to the movie
                        genre_to_movie = self.find_paths(neighbor, movie_id, 2)
                        
                        if genre_to_movie:
                            # Add user to genre to these paths
                            for path in genre_to_movie:
                                genre_paths.append([user_id] + path)
                
                # Convert to text
                path_texts = self.paths_to_text(genre_paths[:max_paths - len(explanations)])
                explanations.extend(path_texts)
            
            # If still not enough, try actor/director based reasoning
            if len(explanations) < max_paths:
                # Get movie's key people (actors, directors)
                movie_people = []
                
                # Find people connected to this movie
                for neighbor in self.graph.neighbors(movie_id):
                    node_type = self.graph.nodes[neighbor].get("node_type", "")
                    if node_type in ["actor", "director", "person"]:
                        movie_people.append(neighbor)
                
                # Find other movies that user liked with these people
                for person in movie_people:
                    # Find other movies with this person
                    for movie_neighbor in self.graph.neighbors(person):
                        if (movie_neighbor != movie_id and 
                            self.graph.nodes[movie_neighbor].get("node_type") == "movie"):
                            
                            # Check if user has connection to this movie
                            if self.graph.has_edge(user_id, movie_neighbor):
                                edge_data = self.graph.get_edge_data(user_id, movie_neighbor)
                                rating = edge_data.get("rating", 0)
                                
                                # If user liked this movie
                                if rating > 0:
                                    # We found a path: user -> other_movie -> person -> recommended_movie
                                    path = [user_id, movie_neighbor, person, movie_id]
                                    path_text = self.paths_to_text([path])[0]
                                    
                                    # Add to explanations if not already full
                                    if len(explanations) < max_paths:
                                        explanations.append(path_text)
                                        
                                    # Break if we have enough explanations
                                    if len(explanations) >= max_paths:
                                        break
                            
                        # Break if we have enough explanations
                        if len(explanations) >= max_paths:
                            break
                    
                    # Break if we have enough explanations
                    if len(explanations) >= max_paths:
                        break
            
            # If still not enough, try similarity-based reasoning
            if len(explanations) < max_paths and self.embeddings:
                # Get user's liked movies
                liked_movies = []
                
                for neighbor in self.graph.neighbors(user_id):
                    # Check if this is a movie the user liked
                    if (self.graph.nodes[neighbor].get("node_type") == "movie" and
                        self.graph.get_edge_data(user_id, neighbor).get("rating", 0) > 0):
                        liked_movies.append(neighbor)
                
                # For each liked movie, check similarity to recommended movie
                similar_pairs = []
                
                for liked_movie in liked_movies:
                    if liked_movie in self.embeddings and movie_id in self.embeddings:
                        similarity = cosine_similarity(
                            self.embeddings[liked_movie].reshape(1, -1),
                            self.embeddings[movie_id].reshape(1, -1)
                        )[0][0]
                        
                        similar_pairs.append((liked_movie, similarity))
                
                # Sort by similarity
                similar_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Add similarity explanations
                for liked_movie, similarity in similar_pairs[:max_paths - len(explanations)]:
                    liked_title = self.graph.nodes[liked_movie].get("title", liked_movie)
                    movie_title = self.graph.nodes[movie_id].get("title", movie_id)
                    
                    # Format similarity as percentage
                    sim_percent = int(similarity * 100)
                    
                    # Create a more insightful explanation
                    if similarity > 0.8:
                        explanation = f"You liked {liked_title}, which is very similar ({sim_percent}% match) to {movie_title}"
                    elif similarity > 0.6:
                        explanation = f"You liked {liked_title}, which has many elements in common ({sim_percent}% similarity) with {movie_title}"
                    elif similarity > 0.4:
                        explanation = f"You liked {liked_title}, which shares some characteristics ({sim_percent}% similarity) with {movie_title}"
                    else:
                        explanation = f"You liked {liked_title}, which has a few elements ({sim_percent}% similarity) similar to {movie_title}"
                    
                    explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            traceback.print_exc()
            return ["We think you'll enjoy this movie based on your preferences."]
    
    def find_common_interests(self, user_id1: str, user_id2: str) -> Dict[str, List]:
        """
        Find common interests between two users
        
        Args:
            user_id1: First user node ID
            user_id2: Second user node ID
            
        Returns:
            Dictionary of common interests by category
        """
        if user_id1 not in self.graph or user_id2 not in self.graph:
            return {}
        
        common_interests = {
            "movies": [],
            "genres": [],
            "actors": [],
            "directors": []
        }
        
        try:
            # Find common movies (both users rated positively)
            user1_movies = set()
            user2_movies = set()
            
            # Get movies for first user
            for neighbor in self.graph.neighbors(user_id1):
                if self.graph.nodes[neighbor].get("node_type") == "movie":
                    edge_data = self.graph.get_edge_data(user_id1, neighbor)
                    if edge_data.get("rating", 0) > 0:
                        user1_movies.add(neighbor)
            
            # Get movies for second user
            for neighbor in self.graph.neighbors(user_id2):
                if self.graph.nodes[neighbor].get("node_type") == "movie":
                    edge_data = self.graph.get_edge_data(user_id2, neighbor)
                    if edge_data.get("rating", 0) > 0:
                        user2_movies.add(neighbor)
            
            # Find common positively rated movies
            common_movies = user1_movies & user2_movies
            
            # Get movie titles for common movies
            for movie_id in common_movies:
                title = self.graph.nodes[movie_id].get("title", movie_id)
                common_interests["movies"].append({
                    "id": movie_id,
                    "title": title
                })
            
            # Find common genre preferences
            user1_genres = self._get_user_genre_preferences(user_id1)
            user2_genres = self._get_user_genre_preferences(user_id2)
            
            # Find common genres
            common_genres = user1_genres & user2_genres
            
            # Add common genres to result
            for genre_id in common_genres:
                genre_name = self.graph.nodes[genre_id].get("name", genre_id)
                common_interests["genres"].append({
                    "id": genre_id,
                    "name": genre_name
                })
            
            # Find common actor preferences
            user1_actors = self._get_user_entity_preferences(user_id1, "actor")
            user2_actors = self._get_user_entity_preferences(user_id2, "actor")
            
            # Find common actors
            common_actors = user1_actors & user2_actors
            
            # Add common actors to result
            for actor_id in common_actors:
                actor_name = self.graph.nodes[actor_id].get("name", actor_id)
                common_interests["actors"].append({
                    "id": actor_id,
                    "name": actor_name
                })
            
            # Find common director preferences
            user1_directors = self._get_user_entity_preferences(user_id1, "director")
            user2_directors = self._get_user_entity_preferences(user_id2, "director")
            
            # Find common directors
            common_directors = user1_directors & user2_directors
            
            # Add common directors to result
            for director_id in common_directors:
                director_name = self.graph.nodes[director_id].get("name", director_id)
                common_interests["directors"].append({
                    "id": director_id,
                    "name": director_name
                })
            
            return common_interests
            
        except Exception as e:
            self.logger.error(f"Error finding common interests: {str(e)}")
            traceback.print_exc()
            return {}
    
    def _get_user_genre_preferences(self, user_id: str) -> Set[str]:
        """Get set of genres preferred by user"""
        genres = set()
        
        # Direct connections to genre nodes
        for neighbor in self.graph.neighbors(user_id):
            if self.graph.nodes[neighbor].get("node_type") == "genre":
                genres.add(neighbor)
        
        # Inferred from positively rated movies
        for neighbor in self.graph.neighbors(user_id):
            if (self.graph.nodes[neighbor].get("node_type") == "movie" and
                self.graph.get_edge_data(user_id, neighbor).get("rating", 0) > 0):
                
                # Get genres for this movie
                for movie_neighbor in self.graph.neighbors(neighbor):
                    if self.graph.nodes[movie_neighbor].get("node_type") == "genre":
                        genres.add(movie_neighbor)
        
        return genres
    
    def _get_user_entity_preferences(self, user_id: str, entity_type: str) -> Set[str]:
        """Get set of entities (actors, directors) preferred by user"""
        entities = set()
        
        # Get preferred entities from positively rated movies
        for neighbor in self.graph.neighbors(user_id):
            if (self.graph.nodes[neighbor].get("node_type") == "movie" and
                self.graph.get_edge_data(user_id, neighbor).get("rating", 0) > 0):
                
                # Get entities for this movie
                for movie_neighbor in self.graph.neighbors(neighbor):
                    if self.graph.nodes[movie_neighbor].get("node_type") == entity_type:
                        entities.add(movie_neighbor)
        
        return entities

class EnhancedCommunityDetector:
    """Enhanced community detection for knowledge graphs with multiple algorithms"""
    
    def __init__(self):
        """Initialize community detector"""
        self.logger = logging.getLogger("community")
        self.logger.info("Initialized enhanced community detector")
        
        # Cache for community detections (avoid recomputing)
        self.community_cache = {}
    
    def detect_communities(self, graph: nx.Graph, method: str = "louvain", 
                          resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in graph using selected method
        
        Args:
            graph: NetworkX graph
            method: Community detection method ('louvain', 'hierarchical', 'label_prop')
            resolution: Resolution parameter for Louvain method
            
        Returns:
            Dictionary mapping node ID to community ID
        """
        # Check cache first
        cache_key = (method, resolution)
        if cache_key in self.community_cache:
            self.logger.info(f"Using cached community detection results for {method}")
            return self.community_cache[cache_key]
        
        self.logger.info(f"Detecting communities using {method} method")
        
        if method == "louvain":
            communities = self._detect_louvain(graph, resolution)
        elif method == "hierarchical":
            communities = self._detect_hierarchical(graph)
        elif method == "label_prop":
            communities = self._detect_label_propagation(graph)
        else:
            self.logger.warning(f"Unknown community detection method: {method}, falling back to Louvain")
            communities = self._detect_louvain(graph, resolution)
        
        # Cache result
        self.community_cache[cache_key] = communities
        
        return communities
    
    def _detect_louvain(self, graph: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities using Louvain method
        
        Args:
            graph: NetworkX graph
            resolution: Resolution parameter (higher for more communities)
            
        Returns:
            Dictionary mapping node ID to community ID
        """
        self.logger.info(f"Detecting communities using Louvain method (resolution={resolution})")
        
        # Filter out hyperedge nodes for better community detection
        filtered_graph = graph.copy()
        hyperedge_nodes = [node for node, data in graph.nodes(data=True) 
                         if data.get("node_type") == "hyperedge"]
        
        if hyperedge_nodes:
            self.logger.info(f"Filtering out {len(hyperedge_nodes)} hyperedge nodes for community detection")
            filtered_graph.remove_nodes_from(hyperedge_nodes)
        
        try:
            # Run Louvain community detection
            communities = community_louvain.best_partition(filtered_graph, resolution=resolution)
            
            # Add back hyperedge nodes by assigning them to the majority community of their neighbors
            for node in hyperedge_nodes:
                if node in graph:
                    neighbors = list(graph.neighbors(node))
                    neighbor_communities = [communities.get(n, -1) for n in neighbors if n in communities]
                    
                    if neighbor_communities:
                        # Assign to most common community among neighbors
                        counter = Counter(neighbor_communities)
                        communities[node] = counter.most_common(1)[0][0]
                    else:
                        # Assign to a new community if no neighbors have communities
                        communities[node] = max(communities.values(), default=-1) + 1
            
            # Renumber communities to be consecutive integers starting from 0
            unique_communities = sorted(set(communities.values()))
            community_map = {old_id: new_id for new_id, old_id in enumerate(unique_communities)}
            renumbered_communities = {node: community_map[comm_id] for node, comm_id in communities.items()}
            
            self.logger.info(f"Detected {len(set(renumbered_communities.values()))} communities")
            return renumbered_communities
            
        except Exception as e:
            self.logger.error(f"Error detecting communities using Louvain: {str(e)}")
            traceback.print_exc()
            # Fallback: assign all nodes to community 0
            return {node: 0 for node in graph.nodes()}
    
    def _detect_hierarchical(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Detect communities using hierarchical clustering
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node ID to community ID
        """
        self.logger.info("Detecting communities using hierarchical clustering")
        
        try:
            # Get node list for consistent ordering
            nodes = list(graph.nodes())
            
            # Create adjacency matrix
            adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)
            
            # Convert to distance matrix (1 - normalized adjacency)
            # Add small value to avoid division by zero
            degrees = np.array([graph.degree(n) for n in nodes])
            degrees = np.maximum(degrees, 0.1)  # Ensure no zeros
            
            # Create degree matrix
            degree_matrix = np.diag(degrees)
            degree_matrix_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
            
            # Compute normalized adjacency matrix
            norm_adjacency = degree_matrix_sqrt_inv @ adjacency_matrix @ degree_matrix_sqrt_inv
            
            # Handle any invalid values
            norm_adjacency = np.nan_to_num(norm_adjacency)
            
            # Convert to distance matrix
            distance_matrix = 1 - norm_adjacency
            
            # Determine optimal number of clusters (simplified)
            n_clusters = min(int(np.sqrt(len(nodes))), 50)  # Heuristic
            
            # Run hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="average"
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Create node to community mapping
            communities = {nodes[i]: int(cluster_labels[i]) for i in range(len(nodes))}
            
            self.logger.info(f"Detected {len(set(communities.values()))} communities")
            return communities
            
        except Exception as e:
            self.logger.error(f"Error detecting communities using hierarchical clustering: {str(e)}")
            traceback.print_exc()
            # Fallback: use Louvain method
            return self._detect_louvain(graph)
    
    def _detect_label_propagation(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Detect communities using label propagation
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node ID to community ID
        """
        self.logger.info("Detecting communities using label propagation")
        
        try:
            # Run label propagation
            communities = nx.algorithms.community.label_propagation_communities(graph)
            
            # Convert to node -> community mapping
            community_mapping = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_mapping[node] = i
            
            self.logger.info(f"Detected {len(set(community_mapping.values()))} communities")
            return community_mapping
            
        except Exception as e:
            self.logger.error(f"Error detecting communities using label propagation: {str(e)}")
            traceback.print_exc()
            # Fallback: use Louvain method
            return self._detect_louvain(graph)
    
    def detect_overlapping_communities(self, graph: nx.Graph, 
                                     embeddings: Dict[str, np.ndarray],
                                     n_communities: int = 10, 
                                     overlap_threshold: float = 0.3) -> Dict[str, List[Tuple[int, float]]]:
        """
        Detect overlapping communities using embeddings
        
        Args:
            graph: NetworkX graph
            embeddings: Dictionary of node_id -> embedding
            n_communities: Target number of communities
            overlap_threshold: Minimum similarity to be part of a community
            
        Returns:
            Dictionary mapping node ID to list of (community_id, membership_strength) tuples
        """
        self.logger.info(f"Detecting overlapping communities (n={n_communities})")
        
        try:
            # First get base communities using Louvain
            base_communities = self._detect_louvain(graph)
            
            # Get unique community IDs from base detection
            unique_communities = sorted(set(base_communities.values()))
            n_base_communities = len(unique_communities)
            
            # If we have fewer than requested communities, adjust
            n_communities = min(n_communities, n_base_communities)
            
            # Create community centroids using embeddings
            centroids = {}
            community_sizes = {}
            
            for comm_id in unique_communities:
                # Get nodes in this community
                nodes = [node for node, c_id in base_communities.items() if c_id == comm_id]
                community_sizes[comm_id] = len(nodes)
                
                # Calculate centroid if we have embeddings for these nodes
                comm_embeddings = [embeddings[node] for node in nodes if node in embeddings]
                if comm_embeddings:
                    centroids[comm_id] = np.mean(comm_embeddings, axis=0)
            
            # Calculate membership strength for each node to each community
            overlapping_communities = {}
            
            for node, node_comm in base_communities.items():
                if node not in embeddings:
                    # If no embedding, just assign to base community
                    overlapping_communities[node] = [(node_comm, 1.0)]
                    continue
                
                node_embedding = embeddings[node]
                
                # Calculate similarity to each community centroid
                similarities = []
                for comm_id, centroid in centroids.items():
                    sim = cosine_similarity(
                        node_embedding.reshape(1, -1), 
                        centroid.reshape(1, -1)
                    )[0][0]
                    
                    # Apply size-based adjustment (slight boost to smaller communities)
                    size_factor = 1.0
                    if community_sizes[comm_id] < 5:
                        size_factor = 1.1  # Small boost to tiny communities
                    
                    adjusted_sim = sim * size_factor
                    similarities.append((comm_id, adjusted_sim))
                
                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Keep communities above threshold
                significant_communities = [
                    (comm_id, sim) for comm_id, sim in similarities
                    if sim >= overlap_threshold
                ]
                
                # Ensure primary community is included
                if not significant_communities or significant_communities[0][0] != node_comm:
                    # Add primary community from base detection
                    if node_comm in centroids:
                        primary_sim = cosine_similarity(
                            node_embedding.reshape(1, -1), 
                            centroids[node_comm].reshape(1, -1)
                        )[0][0]
                    else:
                        primary_sim = 1.0  # Default if no centroid
                    
                    significant_communities.append((node_comm, primary_sim))
                    # Re-sort
                    significant_communities.sort(key=lambda x: x[1], reverse=True)
                
                # Normalize to sum to 1.0
                total_sim = sum(sim for _, sim in significant_communities)
                normalized_communities = [
                    (comm_id, sim / total_sim) for comm_id, sim in significant_communities
                ]
                
                overlapping_communities[node] = normalized_communities
            
            self.logger.info(f"Detected overlapping communities for {len(overlapping_communities)} nodes")
            return overlapping_communities
            
        except Exception as e:
            self.logger.error(f"Error detecting overlapping communities: {str(e)}")
            traceback.print_exc()
            # Fallback: convert base communities to overlapping format
            return {node: [(comm_id, 1.0)] for node, comm_id in self._detect_louvain(graph).items()}
    
    def detect_hierarchical_communities(self, graph: nx.Graph, 
                                       levels: int = 3) -> Dict[str, Dict[int, int]]:
        """
        Detect hierarchical communities at multiple resolutions
        
        Args:
            graph: NetworkX graph
            levels: Number of hierarchical levels
            
        Returns:
            Dictionary mapping node ID to dictionary of level -> community ID
        """
        self.logger.info(f"Detecting hierarchical communities with {levels} levels")
        
        try:
            # Generate different resolutions for Louvain method
            # More levels for finer granularity
            resolutions = []
            for level in range(levels):
                if level == 0:
                    # Highest level (fewest communities)
                    resolutions.append(0.3)
                elif level == levels - 1:
                    # Lowest level (most communities)
                    resolutions.append(2.0)
                else:
                    # Intermediate levels, spread evenly
                    factor = level / (levels - 1)
                    resolution = 0.3 + factor * 1.7  # From 0.3 to 2.0
                    resolutions.append(resolution)
            
            # Run community detection at each resolution
            community_levels = {}
            for level, resolution in enumerate(resolutions):
                community_levels[level] = self._detect_louvain(graph, resolution)
            
            # Restructure result for easier access
            hierarchical_communities = {}
            for node in graph.nodes():
                hierarchical_communities[node] = {
                    level: community_levels[level].get(node, -1) 
                    for level in range(levels)
                }
            
            # Calculate statistics
            level_stats = {}
            for level in range(levels):
                comms = community_levels[level]
                level_stats[level] = len(set(comms.values()))
            
            self.logger.info(f"Detected hierarchical communities with counts: {level_stats}")
            return hierarchical_communities
            
        except Exception as e:
            self.logger.error(f"Error detecting hierarchical communities: {str(e)}")
            traceback.print_exc()
            # Fallback: create single-level hierarchy
            base_communities = self._detect_louvain(graph)
            return {node: {0: comm_id} for node, comm_id in base_communities.items()}
    
    def enhance_communities_with_embeddings(self, graph: nx.Graph, 
                                          base_communities: Dict[str, int],
                                          embeddings: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Enhance community detection results using embeddings
        
        Args:
            graph: NetworkX graph
            base_communities: Dictionary mapping node ID to community ID
            embeddings: Dictionary of node_id -> embedding
            
        Returns:
            Enhanced community assignments
        """
        self.logger.info("Enhancing communities with embeddings")
        
        try:
            # Create community centroids
            unique_communities = sorted(set(base_communities.values()))
            centroids = {}
            
            for comm_id in unique_communities:
                # Get nodes in this community
                nodes = [node for node, c_id in base_communities.items() if c_id == comm_id]
                
                # Calculate centroid if we have embeddings for these nodes
                comm_embeddings = [embeddings[node] for node in nodes if node in embeddings]
                if comm_embeddings:
                    centroids[comm_id] = np.mean(comm_embeddings, axis=0)
            
            # Enhance communities by reassigning nodes to closest centroid
            enhanced_communities = base_communities.copy()
            
            # Number of changes made
            num_changes = 0
            
            for node in graph.nodes():
                if node not in embeddings:
                    continue
                
                node_embedding = embeddings[node]
                
                # Calculate similarity to each community centroid
                best_comm = None
                best_sim = -1
                
                for comm_id, centroid in centroids.items():
                    sim = cosine_similarity(
                        node_embedding.reshape(1, -1), 
                        centroid.reshape(1, -1)
                    )[0][0]
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_comm = comm_id
                
                # Only reassign if significantly better than current assignment
                current_comm = base_communities.get(node)
                if current_comm is not None and current_comm in centroids:
                    current_sim = cosine_similarity(
                        node_embedding.reshape(1, -1), 
                        centroids[current_comm].reshape(1, -1)
                    )[0][0]
                    
                    # Reassign if new community is significantly better
                    if best_sim > current_sim + 0.1:  # 10% improvement threshold
                        enhanced_communities[node] = best_comm
                        num_changes += 1
            
            self.logger.info(f"Enhanced communities with embeddings (changed {num_changes} nodes)")
            return enhanced_communities
            
        except Exception as e:
            self.logger.error(f"Error enhancing communities with embeddings: {str(e)}")
            traceback.print_exc()
            # Fallback: return base communities
            return base_communities
    
    def find_community_bridges(self, graph: nx.Graph, communities: Dict[str, int]) -> List[str]:
        """
        Find bridge nodes that connect different communities
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping node ID to community ID
            
        Returns:
            List of bridge node IDs sorted by importance
        """
        self.logger.info("Finding community bridge nodes")
        
        try:
            bridge_scores = {}
            
            for node in graph.nodes():
                # Skip nodes without community assignment
                if node not in communities:
                    continue
                
                node_comm = communities[node]
                neighbor_comms = set()
                
                # Check all neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor in communities:
                        neighbor_comms.add(communities[neighbor])
                
                # Remove self community
                if node_comm in neighbor_comms:
                    neighbor_comms.remove(node_comm)
                
                # Score is based on number of different communities connected
                # and weighted by node degree
                if neighbor_comms:
                    bridge_scores[node] = {
                        "node_id": node,
                        "communities_connected": len(neighbor_comms),
                        "degree": graph.degree(node),
                        "score": len(neighbor_comms) * graph.degree(node)
                    }
            
            # Sort by score
            bridge_nodes = sorted(
                bridge_scores.keys(),
                key=lambda n: bridge_scores[n]["score"],
                reverse=True
            )
            
            self.logger.info(f"Found {len(bridge_nodes)} community bridge nodes")
            return bridge_nodes
            
        except Exception as e:
            self.logger.error(f"Error finding community bridges: {str(e)}")
            traceback.print_exc()
            return []
    
    def characterize_communities(self, graph: nx.Graph, communities: Dict[str, int]) -> Dict[int, Dict]:
        """
        Generate characteristics for each community
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping node ID to community ID
            
        Returns:
            Dictionary mapping community ID to characteristics
        """
        self.logger.info("Characterizing communities")
        
        try:
            # Group nodes by community
            community_nodes = defaultdict(list)
            for node, comm_id in communities.items():
                community_nodes[comm_id].append(node)
            
            # Characterize each community
            community_chars = {}
            
            for comm_id, nodes in community_nodes.items():
                # Basic statistics
                node_count = len(nodes)
                node_types = Counter()
                top_genres = Counter()
                decade_distribution = Counter()
                
                # Collect node type statistics
                for node in nodes:
                    # Get node type
                    node_type = graph.nodes[node].get("node_type", "unknown")
                    node_types[node_type] += 1
                    
                    # For movies, collect genre and decade info
                    if node_type == "movie":
                        # Get genres
                        genres = graph.nodes[node].get("genres", "[]")
                        if isinstance(genres, str):
                            try:
                                genre_list = json.loads(genres)
                                for genre in genre_list:
                                    top_genres[genre] += 1
                            except:
                                pass
                        
                        # Get decade
                        year = graph.nodes[node].get("year")
                        if year:
                            if isinstance(year, str) and year.isdigit():
                                year = int(year)
                            if isinstance(year, int):
                                decade = (year // 10) * 10
                                decade_distribution[decade] += 1
                
                # Community graph metrics
                subgraph = graph.subgraph(nodes)
                avg_degree = sum(dict(subgraph.degree()).values()) / max(1, len(subgraph))
                
                # Density
                if len(nodes) > 1:
                    density = nx.density(subgraph)
                else:
                    density = 0
                
                # Compile characteristics
                characteristics = {
                    "node_count": node_count,
                    "node_type_distribution": dict(node_types),
                    "average_degree": avg_degree,
                    "density": density,
                    "top_genres": dict(top_genres.most_common(5)),
                    "decade_distribution": dict(decade_distribution),
                }
                
                community_chars[comm_id] = characteristics
            
            self.logger.info(f"Characterized {len(community_chars)} communities")
            return community_chars
            
        except Exception as e:
            self.logger.error(f"Error characterizing communities: {str(e)}")
            traceback.print_exc()
            return {}
        
class MultiObjectiveRecommender:
    """Multi-objective recommendation engine for Bollywood knowledge graph"""
    
    def __init__(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray] = None,
                path_reasoner: PathReasoning = None, ode_embeddings: Dict[str, Dict] = None):
        """
        Initialize multi-objective recommender
        
        Args:
            graph: NetworkX graph
            embeddings: Node embeddings dictionary
            path_reasoner: Path reasoning module for explanations
            ode_embeddings: Temporal embeddings from ODE
        """
        self.graph = graph
        self.embeddings = embeddings
        self.path_reasoner = path_reasoner
        self.ode_embeddings = ode_embeddings
        
        # Objective weights (can be personalized per user)
        self.default_weights = {
            "relevance": 0.4,  # Embedding similarity
            "novelty": 0.15,   # Different from already seen
            "diversity": 0.15, # Different from other recommendations
            "popularity": 0.1, # Based on connections in graph
            "recency": 0.1,    # More recent items get higher score
            "serendipity": 0.1 # Surprising yet relevant
        }
        
        self.logger = logging.getLogger("graph_builder")
        self.logger.info("Initialized Multi-Objective Recommender")
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """
        Extract user preferences from graph
        
        Args:
            user_id: User node ID
            
        Returns:
            Dictionary of preference information
        """
        if user_id not in self.graph:
            return {}
        
        preferences = {
            "liked_movies": [],
            "disliked_movies": [],
            "genres": [],
            "directors": [],
            "actors": [],
            "liked_embeddings": []
        }
        
        try:
            # Get user node
            user_node = self.graph.nodes[user_id]
            
            # Get rated movies
            for neighbor in self.graph.neighbors(user_id):
                if self.graph.nodes[neighbor].get("node_type") == "movie":
                    edge_data = self.graph.get_edge_data(user_id, neighbor)
                    rating = edge_data.get("rating", 0)
                    
                    if rating > 0:
                        preferences["liked_movies"].append(neighbor)
                        # Add embedding if available
                        if self.embeddings and neighbor in self.embeddings:
                            preferences["liked_embeddings"].append(self.embeddings[neighbor])
                    elif rating < 0:
                        preferences["disliked_movies"].append(neighbor)
            
            # Get favorite genres
            favorite_genres = user_node.get("favorite_genres", "[]")
            if isinstance(favorite_genres, str):
                try:
                    preferences["genres"] = json.loads(favorite_genres)
                except:
                    pass
            
            # Get recommended directors
            recommended_directors = user_node.get("recommended_directors", "[]")
            if isinstance(recommended_directors, str):
                try:
                    preferences["directors"] = json.loads(recommended_directors)
                except:
                    pass
            
            # If we have genre nodes connected to user, add them
            for neighbor in self.graph.neighbors(user_id):
                if self.graph.nodes[neighbor].get("node_type") == "genre":
                    edge_data = self.graph.get_edge_data(user_id, neighbor)
                    if edge_data.get("edge_type") == "PREFERS_GENRE":
                        genre_name = self.graph.nodes[neighbor].get("name", "")
                        if genre_name and genre_name not in preferences["genres"]:
                            preferences["genres"].append(genre_name)
            
            # If no explicit preferences, try to infer from liked movies
            if not preferences["genres"] and preferences["liked_movies"]:
                # Count genres from liked movies
                genre_counts = {}
                
                for movie_id in preferences["liked_movies"]:
                    # Get movie genres
                    movie_genres = self.graph.nodes[movie_id].get("genres", "[]")
                    if isinstance(movie_genres, str):
                        try:
                            genres = json.loads(movie_genres)
                            for genre in genres:
                                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        except:
                            pass
                
                # Get top genres
                if genre_counts:
                    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
                    preferences["genres"] = [genre for genre, count in top_genres[:3]]
            
            # Get preferred actors from liked movies
            if preferences["liked_movies"]:
                actor_counts = {}
                
                for movie_id in preferences["liked_movies"]:
                    # Check all neighbors of the movie
                    for neighbor in self.graph.neighbors(movie_id):
                        if self.graph.nodes[neighbor].get("node_type") == "actor":
                            actor_name = self.graph.nodes[neighbor].get("name", "")
                            if actor_name:
                                actor_counts[actor_name] = actor_counts.get(actor_name, 0) + 1
                
                # Get top actors
                if actor_counts:
                    top_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)
                    preferences["actors"] = [actor for actor, count in top_actors[:3]]
            
            # Get preferred directors from liked movies
            if preferences["liked_movies"] and not preferences["directors"]:
                director_counts = {}
                
                for movie_id in preferences["liked_movies"]:
                    # Check all neighbors of the movie
                    for neighbor in self.graph.neighbors(movie_id):
                        if self.graph.nodes[neighbor].get("node_type") == "director":
                            director_name = self.graph.nodes[neighbor].get("name", "")
                            if director_name:
                                director_counts[director_name] = director_counts.get(director_name, 0) + 1
                
                # Get top directors
                if director_counts:
                    top_directors = sorted(director_counts.items(), key=lambda x: x[1], reverse=True)
                    preferences["directors"] = [director for director, count in top_directors[:3]]
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            traceback.print_exc()
            return preferences
    
    def filter_candidate_movies(self, user_id: str, n: int = 100) -> List[str]:
        """
        Get candidate movies for recommendation
        
        Args:
            user_id: User node ID
            n: Maximum number of candidates to return
            
        Returns:
            List of candidate movie IDs
        """
        # Get user preferences
        preferences = self.get_user_preferences(user_id)
        
        # Get already rated movies
        rated_movies = set(preferences["liked_movies"] + preferences["disliked_movies"])
        
        # Get all movie nodes
        movie_nodes = [node for node, data in self.graph.nodes(data=True) 
                     if data.get("node_type") == "movie"]
        
        # Filter out already rated movies
        candidates = [movie for movie in movie_nodes if movie not in rated_movies]
        
        # If we have preferred genres, boost movies with those genres
        if preferences["genres"]:
            # Score each movie by genre match
            genre_scores = {}
            
            for movie_id in candidates:
                movie_genres = self.graph.nodes[movie_id].get("genres", "[]")
                if isinstance(movie_genres, str):
                    try:
                        genres = json.loads(movie_genres)
                        # Count matching genres
                        matching = sum(1 for g in genres if g in preferences["genres"])
                        if matching > 0:
                            genre_scores[movie_id] = matching / len(preferences["genres"])
                    except:
                        pass
            
            # Sort candidates by genre score
            if genre_scores:
                sorted_candidates = sorted(
                    candidates,
                    key=lambda x: genre_scores.get(x, 0),
                    reverse=True
                )
                
                # Take top candidates plus some random ones for diversity
                top_n = min(n // 2, len(sorted_candidates))
                top_candidates = sorted_candidates[:top_n]
                
                # Add some random candidates
                remaining = [c for c in candidates if c not in top_candidates]
                if remaining:
                    random_candidates = random.sample(
                        remaining,
                        min(n - top_n, len(remaining))
                    )
                    
                    return top_candidates + random_candidates
        
        # If no genre filtering or not enough candidates, return random selection
        if len(candidates) > n:
            return random.sample(candidates, n)
        
        return candidates
    
    def compute_relevance(self, user_id: str, movie_id: str, preferences: Dict = None) -> float:
        """
        Compute relevance score based on similarity to user preferences
        
        Args:
            user_id: User node ID
            movie_id: Movie node ID
            preferences: Optional pre-computed preferences
            
        Returns:
            Relevance score [0, 1]
        """
        if self.embeddings is None or user_id not in self.embeddings or movie_id not in self.embeddings:
            return 0.5  # Default score if no embeddings
        
        try:
            # Option 1: Direct embedding similarity between user and movie
            user_embedding = self.embeddings[user_id]
            movie_embedding = self.embeddings[movie_id]
            
            direct_similarity = cosine_similarity(
                user_embedding.reshape(1, -1),
                movie_embedding.reshape(1, -1)
            )[0][0]
            
            # Option 2: Similarity to liked movies
            if preferences is None:
                preferences = self.get_user_preferences(user_id)
            
            liked_similarities = []
            for liked_movie in preferences["liked_movies"]:
                if liked_movie in self.embeddings:
                    liked_embedding = self.embeddings[liked_movie]
                    sim = cosine_similarity(
                        liked_embedding.reshape(1, -1),
                        movie_embedding.reshape(1, -1)
                    )[0][0]
                    liked_similarities.append(sim)
            
            # Option 3: Genre-based similarity
            genre_score = 0.0
            if preferences["genres"]:
                movie_genres = self.graph.nodes[movie_id].get("genres", "[]")
                if isinstance(movie_genres, str):
                    try:
                        genres = json.loads(movie_genres)
                        matching = sum(1 for g in genres if g in preferences["genres"])
                        if matching > 0 and len(genres) > 0:
                            genre_score = matching / len(genres)
                    except:
                        pass
            
            # Weighted combination of all signals
            if liked_similarities:
                avg_liked_sim = np.mean(liked_similarities)
                return 0.4 * direct_similarity + 0.4 * avg_liked_sim + 0.2 * genre_score
            else:
                return 0.7 * direct_similarity + 0.3 * genre_score
            
        except Exception as e:
            self.logger.error(f"Error computing relevance: {str(e)}")
            traceback.print_exc()
            return 0.5
    
    def compute_novelty(self, movie_id: str, liked_movies: List[str]) -> float:
        """
        Compute novelty score (how different from already seen movies)
        
        Args:
            movie_id: Movie node ID
            liked_movies: List of movies already liked by user
            
        Returns:
            Novelty score [0, 1]
        """
        if not liked_movies or self.embeddings is None or movie_id not in self.embeddings:
            return 0.5  # Default score
        
        try:
            # Compute minimum distance to any liked movie
            movie_embedding = self.embeddings[movie_id]
            
            max_similarity = 0.0
            for liked_movie in liked_movies:
                if liked_movie in self.embeddings:
                    liked_embedding = self.embeddings[liked_movie]
                    sim = cosine_similarity(
                        liked_embedding.reshape(1, -1),
                        movie_embedding.reshape(1, -1)
                    )[0][0]
                    max_similarity = max(max_similarity, sim)
            
            # Novelty is inverse of maximum similarity
            return 1.0 - max_similarity
            
        except Exception as e:
            self.logger.error(f"Error computing novelty: {str(e)}")
            traceback.print_exc()
            return 0.5
    
    def compute_diversity(self, movie_id: str, recommended_movies: List[str]) -> float:
        """
        Compute diversity score (how different from other recommendations)
        
        Args:
            movie_id: Movie node ID
            recommended_movies: List of already recommended movies
            
        Returns:
            Diversity score [0, 1]
        """
        if not recommended_movies or self.embeddings is None or movie_id not in self.embeddings:
            return 1.0  # Maximum diversity if no other recommendations
        
        try:
            # Compute minimum distance to any recommended movie
            movie_embedding = self.embeddings[movie_id]
            
            max_similarity = 0.0
            for rec_movie in recommended_movies:
                if rec_movie in self.embeddings:
                    rec_embedding = self.embeddings[rec_movie]
                    sim = cosine_similarity(
                        rec_embedding.reshape(1, -1),
                        movie_embedding.reshape(1, -1)
                    )[0][0]
                    max_similarity = max(max_similarity, sim)
            
            # Diversity is inverse of maximum similarity
            return 1.0 - max_similarity
            
        except Exception as e:
            self.logger.error(f"Error computing diversity: {str(e)}")
            traceback.print_exc()
            return 1.0
    
    def compute_popularity(self, movie_id: str) -> float:
        """
        Compute popularity score
        
        Args:
            movie_id: Movie node ID
            
        Returns:
            Popularity score [0, 1]
        """
        if movie_id not in self.graph:
            return 0.0
        
        try:
            # Use degree as proxy for popularity, normalized by log scale
            degree = self.graph.degree(movie_id)
            
            # Get average movie degree for normalization
            movie_nodes = [n for n, d in self.graph.nodes(data=True) 
                         if d.get("node_type") == "movie"]
            movie_degrees = [self.graph.degree(n) for n in movie_nodes]
            avg_degree = np.mean(movie_degrees) if movie_degrees else 10
            
            # Normalize using logarithmic scale with average as reference
            popularity = 0.5 * (1 + np.tanh((degree - avg_degree) / (avg_degree / 2)))
            
            return popularity
            
        except Exception as e:
            self.logger.error(f"Error computing popularity: {str(e)}")
            traceback.print_exc()
            return 0.0
    
    def compute_recency(self, movie_id: str) -> float:
        """
        Compute recency score
        
        Args:
            movie_id: Movie node ID
            
        Returns:
            Recency score [0, 1]
        """
        if movie_id not in self.graph:
            return 0.5
        
        try:
            # Get year
            year = self.graph.nodes[movie_id].get("year", None)
            
            if year is None:
                return 0.5  # Default if no year
            
            # Convert to int if needed
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            
            if not isinstance(year, int):
                return 0.5
            
            # Compute recency (sigmoid function centered at 2000)
            current_year = 2025  # Use current year
            center_year = 2000   # Center of sigmoid
            scale = 15          # Scale factor for sigmoid steepness
            
            # Calculate normalized recency score
            recency = 1 / (1 + np.exp(-(year - center_year) / scale))
            
            return recency
            
        except Exception as e:
            self.logger.error(f"Error computing recency: {str(e)}")
            traceback.print_exc()
            return 0.5
    
    def compute_serendipity(self, user_id: str, movie_id: str, relevance: float, 
                           novelty: float, preferences: Dict = None) -> float:
        """
        Compute serendipity score (surprising yet relevant)
        
        Args:
            user_id: User node ID
            movie_id: Movie node ID
            relevance: Pre-computed relevance score
            novelty: Pre-computed novelty score
            preferences: Optional pre-computed preferences
            
        Returns:
            Serendipity score [0, 1]
        """
        try:
            # Serendipity balances relevance and novelty
            # We want items that are both somewhat relevant and somewhat novel
            
            # Get user's genre preferences
            if preferences is None:
                preferences = self.get_user_preferences(user_id)
            
            # Get movie genres
            movie_genres = self.graph.nodes[movie_id].get("genres", "[]")
            if isinstance(movie_genres, str):
                try:
                    movie_genres = json.loads(movie_genres)
                except:
                    movie_genres = []
            
            # Compute genre-based serendipity
            genre_serendipity = 0.5  # Default value
            
            if preferences["genres"] and movie_genres:
                # Count matching genres
                matching = sum(1 for g in movie_genres if g in preferences["genres"])
                
                # Unexpected genres
                unexpected = len(movie_genres) - matching
                
                # Serendipity is highest when there's a mix of expected and unexpected
                if matching > 0 and unexpected > 0:
                    # Balance between matching and unexpected
                    balance = 4 * (matching / len(movie_genres)) * (unexpected / len(movie_genres))
                    genre_serendipity = 0.5 + 0.5 * balance
                elif matching > 0:
                    # All expected genres - moderate serendipity
                    genre_serendipity = 0.3
                else:
                    # No expected genres - low serendipity (likely irrelevant)
                    genre_serendipity = 0.1
            
            # Combine relevance, novelty and genre serendipity
            # Serendipity is highest when item is moderately relevant and highly novel
            balanced_rel_nov = (relevance ** 0.7) * (novelty ** 0.3)
            
            # Final serendipity score
            serendipity = 0.7 * balanced_rel_nov + 0.3 * genre_serendipity
            
            return serendipity
            
        except Exception as e:
            self.logger.error(f"Error computing serendipity: {str(e)}")
            traceback.print_exc()
            return 0.5
    
    def generate_recommendations(self, user_id: str, n: int = 5, 
                               weights: Dict[str, float] = None) -> List[Dict]:
        """
        Generate recommendations based on multiple objectives
        
        Args:
            user_id: User node ID
            n: Number of recommendations to return
            weights: Optional custom weights for objectives
            
        Returns:
            List of recommendation dictionaries
        """
        self.logger.info(f"Generating {n} recommendations for user {user_id}")
        
        if user_id not in self.graph:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        # Use default weights if none provided
        weights = weights or self.default_weights
        
        try:
            # Get user preferences
            preferences = self.get_user_preferences(user_id)
            
            # Get candidate movies
            candidates = self.filter_candidate_movies(user_id, n * 10)
            
            if not candidates:
                self.logger.warning(f"No candidate movies for user {user_id}")
                return []
            
            # Generate recommendations incrementally to consider diversity
            recommendations = []
            recommended_ids = []
            
            while len(recommendations) < n and candidates:
                # Score all remaining candidates
                candidate_scores = []
                
                for movie_id in candidates:
                    # Compute scores for each objective
                    relevance = self.compute_relevance(user_id, movie_id, preferences)
                    novelty = self.compute_novelty(movie_id, preferences["liked_movies"])
                    diversity = self.compute_diversity(movie_id, recommended_ids)
                    popularity = self.compute_popularity(movie_id)
                    recency = self.compute_recency(movie_id)
                    serendipity = self.compute_serendipity(user_id, movie_id, relevance, novelty, preferences)
                    
                    # Combine using weighted sum
                    total_score = (
                        weights["relevance"] * relevance +
                        weights["novelty"] * novelty +
                        weights["diversity"] * diversity +
                        weights["popularity"] * popularity +
                        weights["recency"] * recency +
                        weights["serendipity"] * serendipity
                    )
                    
                    # Add scores to candidate list
                    candidate_scores.append({
                        "movie_id": movie_id,
                        "total_score": total_score,
                        "scores": {
                            "relevance": relevance,
                            "novelty": novelty,
                            "diversity": diversity,
                            "popularity": popularity,
                            "recency": recency,
                            "serendipity": serendipity
                        }
                    })
                
                # Sort by total score
                candidate_scores.sort(key=lambda x: x["total_score"], reverse=True)
                
                # Get top candidate
                top_candidate = candidate_scores[0]
                movie_id = top_candidate["movie_id"]
                
                # Add to recommendations
                movie_data = self.graph.nodes[movie_id]
                
                # Extract movie information
                title = movie_data.get("title", movie_id)
                year = movie_data.get("year", "")
                
                # Get genres
                genres = movie_data.get("genres", "[]")
                if isinstance(genres, str):
                    try:
                        genres = json.loads(genres)
                    except:
                        genres = []
                
                # Generate explanation if we have path reasoning
                explanation = []
                if self.path_reasoner:
                    explanation = self.path_reasoner.explain_recommendation(user_id, movie_id, max_paths=2)
                
                # Add recommendation
                recommendations.append({
                    "movie_id": movie_id,
                    "title": title,
                    "year": year,
                    "genres": genres,
                    "score": top_candidate["total_score"],
                    "objective_scores": top_candidate["scores"],
                    "explanation": explanation
                })
                
                # Add to recommended list for diversity calculation
                recommended_ids.append(movie_id)
                
                # Remove from candidates
                candidates.remove(movie_id)
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return []

class EnhancedBollywoodKGBuilder:
    """Enhanced knowledge graph builder with cutting-edge graph techniques"""
    
    def __init__(self, api_key: str, data_dir: str = "data", 
                cache_dir: str = "cache", output_dir: str = "final_enhanced_output",
                users_file: str = "users.csv", 
                ratings_file: str = "ratings.json",
                movie_mapping_file: str = "movie_id_mapping.json"):
        
        self.api_key = api_key
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.users_file = "/Users/amanvaibhavjha/Desktop/CRS/final_code/archive/users.csv"
        self.ratings_file = "/Users/amanvaibhavjha/Desktop/CRS/final_code/prepared_data/ratings_array.json"
        self.movie_mapping_file = "/Users/amanvaibhavjha/Desktop/CRS/final_code/prepared_data/movie_id_mapping.json"

        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize API client
        self.api_client = EnhancedAPIClient(api_key, cache_dir)

        # Initialize data collector
        self.data_collector = FlexibleDataCollector(self.api_client, data_dir)
        
        # Initialize embedding generator
        self.embedding_generator = AdvancedEmbeddingGenerator(
            self.api_client,
            os.path.join(output_dir, "embeddings"),
            use_quantum=True
        )

        # Initialize hypergraph
        self.hypergraph = TemporalHypergraph(name="bollywood_kg_enhanced")
        
        # Load movie TT mapping
        self.movie_tt_mapping = {}
        if os.path.exists(self.movie_mapping_file):
            try:
                self.movie_tt_mapping = load_tt_mapping(self.movie_mapping_file)
                print(f"Loaded {len(self.movie_tt_mapping)} movie mappings.")
            except Exception as e:
                print(f"Warning: Could not load movie mapping: {str(e)}")
        

        # Initialize user and rating data
        self.user_data = {}
        self.user_personas = {}
        self.genre_data = {}
        self.enhanced_genre_info = {}

        # Track processed entities
        self.processed_entities = set()
        
        # Initialize community detector
        self.community_detector = EnhancedCommunityDetector()
        
        # Initialize standard graphs
        self.graph = nx.Graph()
        self.digraph = nx.DiGraph()
        
        # Initialize embeddings
        self.embeddings = {}
        
        # Initialize communities
        self.communities = {}
        self.overlapping_communities = {}
        self.hierarchical_communities = {}
        
        # Initialize path reasoning (will be created after graph is built)
        self.path_reasoner = None
        
        # Initialize multi-objective recommender (will be created after graph is built)
        self.recommender = None
        
        self.logger = logging.getLogger("graph_builder")
        self.logger.info("Initialized Enhanced Bollywood KG Builder")

    async def load_user_data(self) -> Dict:
        "Load and process user data"

        self.logger.info(f"Loading user data from {self.users_file}")
        
        try:
            user_df = pd.read_csv(self.users_file)

            users = {}

            for _, row in user_df.iterrows():
                try:
                    user_id = row['_id']

                    # Process languages - handle different formats
                    if isinstance(row['languages'], str):
                        try:
                            languages = json.loads(row['languages'].replace("'", "\""))
                        except:
                            languages = [lang.strip(' "') for lang in row['languages'].strip('[]').split(',')]
                    else:
                        languages = []
                    
                    # Clean up languages (remove quotes, extra whitespace)
                    languages = [lang.strip(' "\'') for lang in languages]
                    
                    # Process date of birth and calculate age
                    dob = row.get('dob', '')
                    age = None
                    if dob and isinstance(dob, str):
                        try:
                            dob_date = datetime.strptime(dob, '%d-%m-%Y')
                            today = datetime.now()
                            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                        except Exception as e:
                            self.logger.warning(f"Error parsing DOB for user {user_id}: {str(e)}")
                    
                    # Create user data dictionary
                    users[user_id] = {
                        'user_id': user_id,
                        'gender': row.get('gender', ''),
                        'job': row.get('job', ''),
                        'state': row.get('state', ''),
                        'dob': dob,
                        'age': age,
                        'languages': languages
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error processing user data row: {str(e)}")
            
            self.logger.info(f"Loaded data for {len(users)} users")
            self.user_data = users
            return users
            
        except Exception as e:
            self.logger.error(f"Error loading user data: {str(e)}")
            traceback.print_exc()
            return {}
        

    async def load_ratings_data(self) -> Dict:
        """Load and process user ratings data from JSON file"""
        self.logger.info(f"Loading ratings data from {self.ratings_file}")
    
        try:
            # Read ratings JSON file
            with open(self.ratings_file, 'r') as f:
                ratings_data = json.load(f)

                # Process each user's ratings
            for user_entry in ratings_data:
                user_id = user_entry.get('_id', '').strip()
                if not user_id or user_id not in self.user_data:
                    self.logger.debug(f"User {user_id} not found in user_data, skipping")
                    continue
                
                rated_items = user_entry.get('rated', {})
                user_ratings = []
                
                # Process each movie rating
                for movie_id, rating_list in rated_items.items():
                    # Skip non-rating entries (like "submit", "submitexit", etc.)
                    if movie_id.lower().startswith("submit"):
                        continue
                    
                    # Ensure it's a non-empty list
                    if not isinstance(rating_list, list) or not rating_list:
                        continue

                    # rating_list[0] is typically "-1", "0", or "1"
                    rating_str = rating_list[0]
                    try:
                        rating_int = int(rating_str)
                    except ValueError:
                        # If it's not an integer, skip
                        continue
                    
                    # Check if rating is one of the valid -1, 0, 1
                    if rating_int not in [-1, 0, 1]:
                        continue

                    # Look up movie details from collected data or mapping
                    movie_data = self.data_collector.get_movie_data_by_tt_id(movie_id, self.movie_tt_mapping)
                    # Append rating details
                    user_ratings.append({
                        'movie_id': movie_id,
                        'title': movie_data.get('title', movie_id),
                        'year': movie_data.get('year', ''),
                        'genres': movie_data.get('genres', []),
                        'rating': rating_int
                    })
                
                # If we got any valid ratings, store them in user_data
                if user_ratings:
                    self.user_data[user_id]['ratings'] = user_ratings
                    self.logger.debug(f"Processed {len(user_ratings)} ratings for user {user_id}")
            
            # Count how many users have ratings
            users_with_ratings = sum(1 for user in self.user_data.values() if 'ratings' in user)
            self.logger.info(f"Processed ratings for {users_with_ratings} users")
            
            return self.user_data
        
        except Exception as e:
            self.logger.error(f"Error loading ratings data: {str(e)}")
            traceback.print_exc()
            return self.user_data
        
    async def _process_movies(self, movie_titles: List[str]):
        """Process movie entities into the hypergraph"""
        self.logger.info(f"Processing {len(movie_titles)} movies into hypergraph")
        
        for movie_title in async_tqdm(movie_titles, desc="Processing movies"):
            if movie_title in self.processed_entities:
                continue
            
            try:
                # Load movie data
                normalized_title = self.data_collector.normalize_entity_name(movie_title)
                movie_data = self.data_collector._load_data("movie", normalized_title)
                
                if not movie_data or "error" in movie_data:
                    self.logger.warning(f"Failed to load data for movie {movie_title}")
                    continue
                
                # Create node ID
                movie_id = f"movie_{self.data_collector.normalize_entity_name(movie_title)}"
                
                # Get tt_id if available
                tt_id = movie_data.get("imdb_id", movie_data.get("tt_id", ""))
                
                # Extract attributes
                attributes = {
                    "title": movie_data.get("title", movie_title),
                    "original_title": movie_data.get("original_title", ""),
                    "plot": movie_data.get("plot", ""),
                    "language": movie_data.get("language", "Hindi"),
                    "genres": json.dumps(movie_data.get("genres", [])),
                    "themes": json.dumps(movie_data.get("themes", [])),
                    "cultural_impact": movie_data.get("cultural_impact", ""),
                    "box_office": movie_data.get("box_office", ""),
                    "critical_reception": movie_data.get("critical_reception", ""),
                    "mood": json.dumps(movie_data.get("mood", [])),
                    "social_themes": json.dumps(movie_data.get("social_themes", [])),
                    "iconic_elements": json.dumps(movie_data.get("iconic_elements", [])),
                    "remake_info": movie_data.get("remake_info", ""),
                    "franchise": movie_data.get("franchise", ""),
                    "imdb_id": tt_id  # Store tt_id for easier lookups
                }
                
                # Extract temporal information
                year = movie_data.get("year")
                release_date = movie_data.get("release_date", "")
                
                temporal_info = {"year": year} if year else {}
                if release_date:
                    temporal_info["release_date"] = release_date
                
                # Add to hypergraph
                self.hypergraph.add_node(
                    node_id=movie_id,
                    node_type="movie",
                    attributes=attributes,
                    temporal_info=temporal_info
                )
                
                # Process cast information as hyperedges
                if "cast" in movie_data and movie_data["cast"]:
                    cast_edge_id = f"cast_{movie_id}"
                    cast_nodes = [movie_id]
                    
                    # Process cast members
                    for cast_member in movie_data["cast"]:
                        if not isinstance(cast_member, dict):
                            continue
                            
                        actor_name = cast_member.get("name")
                        if not actor_name:
                            continue
                        
                        actor_id = f"person_{self.data_collector.normalize_entity_name(actor_name)}"
                        cast_nodes.append(actor_id)
                        
                        # Add binary edge for specific actor-movie relationship
                        actor_edge_id = f"acted_{actor_id}_{movie_id}"
                        
                        character = cast_member.get("character", "")
                        importance = cast_member.get("importance", "supporting")
                        
                        # Create participation weights
                        weights = {
                            actor_id: 1.0,
                            movie_id: 1.0
                        }
                        
                        # Adjust weight based on importance
                        if importance == "lead":
                            weights[actor_id] = 1.5
                        elif importance == "supporting":
                            weights[actor_id] = 1.0
                        elif importance == "minor":
                            weights[actor_id] = 0.5
                        
                        self.hypergraph.add_hyperedge(
                            edge_id=actor_edge_id,
                            edge_type="ACTED_IN",
                            nodes=[actor_id, movie_id],
                            attributes={
                                "character": character,
                                "importance": importance
                            },
                            temporal_info=temporal_info,
                            weights=weights
                        )
                    
                    # Add cast hyperedge connecting all actors and the movie
                    if len(cast_nodes) > 2:
                        # Create weights with movie having highest importance
                        weights = {node_id: 1.0 for node_id in cast_nodes}
                        weights[movie_id] = 2.0  # Movie is most important in cast hyperedge
                        
                        self.hypergraph.add_hyperedge(
                            edge_id=cast_edge_id,
                            edge_type="CAST",
                            nodes=cast_nodes,
                            attributes={},
                            temporal_info=temporal_info,
                            weights=weights
                        )
                
                # Add director relationship
                director = movie_data.get("director")
                if director:
                    director_id = f"person_{self.data_collector.normalize_entity_name(director)}"
                    
                    self.hypergraph.add_hyperedge(
                        edge_id=f"directed_{director_id}_{movie_id}",
                        edge_type="DIRECTED",
                        nodes=[director_id, movie_id],
                        attributes={},
                        temporal_info=temporal_info
                    )
                
                # Add writer relationships
                for writer in movie_data.get("writers", []):
                    if not writer:
                        continue
                    
                    writer_id = f"person_{self.data_collector.normalize_entity_name(writer)}"
                    
                    self.hypergraph.add_hyperedge(
                        edge_id=f"wrote_{writer_id}_{movie_id}",
                        edge_type="WROTE",
                        nodes=[writer_id, movie_id],
                        attributes={},
                        temporal_info=temporal_info
                    )
                
                # Add music director relationship
                music_director = movie_data.get("music_director")
                if music_director:
                    music_director_id = f"person_{self.data_collector.normalize_entity_name(music_director)}"
                    
                    self.hypergraph.add_hyperedge(
                        edge_id=f"music_{music_director_id}_{movie_id}",
                        edge_type="COMPOSED_MUSIC",
                        nodes=[music_director_id, movie_id],
                        attributes={},
                        temporal_info=temporal_info
                    )
                
                # Add genre relationships
                genres = movie_data.get("genres", [])
                if isinstance(genres, str):
                    try:
                        genres = json.loads(genres)
                    except:
                        genres = [g.strip() for g in genres.split(",") if g.strip()]
                
                for genre in genres:
                    if not genre:
                        continue
                    
                    genre_id = f"genre_{self.data_collector.normalize_entity_name(genre)}"
                    
                    # Add genre node if it doesn't exist
                    if not self.hypergraph.get_node(genre_id):
                        self.hypergraph.add_node(
                            node_id=genre_id,
                            node_type="genre",
                            attributes={"name": genre}
                        )
                    
                    self.hypergraph.add_hyperedge(
                        edge_id=f"genre_{movie_id}_{genre_id}",
                        edge_type="HAS_GENRE",
                        nodes=[movie_id, genre_id],
                        attributes={},
                        temporal_info=temporal_info
                    )
                
                # Add theme relationships
                themes = movie_data.get("themes", [])
                if isinstance(themes, str):
                    try:
                        themes = json.loads(themes)
                    except:
                        themes = [t.strip() for t in themes.split(",") if t.strip()]
                
                for theme in themes:
                    if not theme:
                        continue
                    
                    theme_id = f"theme_{self.data_collector.normalize_entity_name(theme)}"
                    
                    # Add theme node if it doesn't exist
                    if not self.hypergraph.get_node(theme_id):
                        self.hypergraph.add_node(
                            node_id=theme_id,
                            node_type="theme",
                            attributes={"name": theme}
                        )
                    
                    self.hypergraph.add_hyperedge(
                        edge_id=f"theme_{movie_id}_{theme_id}",
                        edge_type="HAS_THEME",
                        nodes=[movie_id, theme_id],
                        attributes={},
                        temporal_info=temporal_info
                    )
                
                # If we have a tt_id, add it to the mapping
                if tt_id and tt_id not in self.movie_tt_mapping:
                    self.movie_tt_mapping[tt_id] = movie_id
                
                # Mark as processed
                self.processed_entities.add(movie_title)
                
            except Exception as e:
                self.logger.error(f"Error processing movie {movie_title}: {str(e)}")
                traceback.print_exc()
        
        self.logger.info(f"Processed {len(self.processed_entities)} movies into hypergraph")
    
    async def _process_users(self):
        """Process user entities into the hypergraph"""
        self.logger.info(f"Processing {len(self.user_data)} users into hypergraph")
        
        for user_id, user in async_tqdm(self.user_data.items(), desc="Processing users"):
            try:
                # Create node ID
                node_id = f"user_{self.data_collector.normalize_entity_name(user_id)}"
                
                # Extract attributes
                attributes = {
                    "name": user.get("user_id", user_id),
                    "gender": user.get("gender", ""),
                    "job": user.get("job", ""),
                    "state": user.get("state", ""),
                    "languages": json.dumps(user.get("languages", [])),
                    "age": user.get("age", ""),
                    "node_type": "user"  # Ensure node_type is set for proper categorization
                }
                
                # Add persona information if available
                if user_id in self.user_personas:
                    persona = self.user_personas[user_id]
                    
                    # Add demographic profile
                    demographic = persona.get("demographic_profile", {})
                    attributes["age_group"] = demographic.get("age_group", "")
                    attributes["likely_education_level"] = demographic.get("likely_education_level", "")
                    attributes["likely_income_level"] = demographic.get("likely_income_level", "")
                    
                    # Add preference profile
                    preferences = persona.get("preference_profile", {})
                    attributes["favorite_genres"] = json.dumps(preferences.get("favorite_genres", []))
                    attributes["disliked_genres"] = json.dumps(preferences.get("disliked_genres", []))
                    attributes["favorite_eras"] = json.dumps(preferences.get("favorite_eras", []))
                    attributes["preference_consistency"] = preferences.get("preference_consistency", 0.5)
                    attributes["experimental_tendency"] = preferences.get("experimental_tendency", 0.5)
                    attributes["mainstream_vs_arthouse"] = preferences.get("mainstream_vs_arthouse", 0.5)
                    
                    # Add behavior insights
                    behavior = persona.get("behavior_insights", {})
                    attributes["rating_patterns"] = behavior.get("rating_patterns", "")
                    attributes["likely_viewing_habits"] = behavior.get("likely_viewing_habits", "")
                    attributes["content_triggers"] = json.dumps(behavior.get("content_triggers", []))
                    attributes["content_attractors"] = json.dumps(behavior.get("content_attractors", []))
                    
                    # Add recommendation strategy
                    strategy = persona.get("recommendation_strategy", {})
                    attributes["recommended_genres"] = json.dumps(strategy.get("recommended_genres", []))
                    attributes["recommended_directors"] = json.dumps(strategy.get("recommended_directors", []))
                    attributes["recommended_eras"] = json.dumps(strategy.get("recommended_eras", []))
                    attributes["recommendation_diversity"] = strategy.get("recommendation_diversity", 0.5)
                    attributes["risk_tolerance"] = strategy.get("risk_tolerance", 0.5)
                
                # Add to hypergraph
                self.hypergraph.add_node(
                    node_id=node_id,
                    node_type="user",
                    attributes=attributes
                )
                
                # Process user ratings as edges
                if 'ratings' in user and user['ratings']:
                    for rating_item in user['ratings']:
                        movie_id = rating_item.get('movie_id', '')
                        title = rating_item.get('title', '')
                        rating_value = rating_item.get('rating', '')
                        
                        if not movie_id or rating_value == '':
                            continue
                        
                        # Create movie node ID
                        # First check if in mapping
                        if movie_id in self.movie_tt_mapping:
                            mapped_title = self.movie_tt_mapping[movie_id]
                            if mapped_title.startswith("movie_"):
                                movie_node_id = mapped_title
                            else:
                                movie_node_id = f"movie_{self.data_collector.normalize_entity_name(mapped_title)}"
                        else:
                            # If not in mapping, use title or ID
                            if title:
                                movie_node_id = f"movie_{self.data_collector.normalize_entity_name(title)}"
                            else:
                                movie_node_id = f"movie_{self.data_collector.normalize_entity_name(movie_id.replace(':', '_'))}"
                        
                        # Check if movie node exists (use less strict check)
                        movie_exists = False
                        
                        # Check existing movie nodes for matches
                        for existing_node_id in self.hypergraph.nodes:
                            if existing_node_id.startswith("movie_") and (
                                movie_node_id == existing_node_id or 
                                movie_id.lower() in existing_node_id.lower() or
                                (title and title.lower() in existing_node_id.lower())
                            ):
                                movie_node_id = existing_node_id
                                movie_exists = True
                                break
                        
                        # If movie doesn't exist, create a minimal node
                        if not movie_exists and not self.hypergraph.get_node(movie_node_id):
                            # Add minimal movie node
                            self.hypergraph.add_node(
                                node_id=movie_node_id,
                                node_type="movie",
                                attributes={
                                    "title": title or movie_id,
                                    "imdb_id": movie_id,
                                    "year": rating_item.get("year", ""),
                                    "genres": json.dumps(rating_item.get("genres", []))
                                }
                            )
                        
                        # Add rating edge
                        edge_id = f"rated_{node_id}_{movie_node_id}"
                        
                        # Convert rating value to appropriate format
                        try:
                            rating_num = int(rating_value)
                        except:
                            rating_num = 0  # Default to neutral if can't parse
                        
                        # Create participation weights
                        # Stronger weight for positive/negative ratings, weaker for neutral
                        if rating_num > 0:
                            weights = {node_id: 1.5, movie_node_id: 1.0}
                        elif rating_num < 0:
                            weights = {node_id: 1.2, movie_node_id: 1.0}
                        else:
                            weights = {node_id: 0.8, movie_node_id: 1.0}
                        
                        self.hypergraph.add_hyperedge(
                            edge_id=edge_id,
                            edge_type="RATED",
                            nodes=[node_id, movie_node_id],
                            attributes={
                                "rating": rating_num,
                                "rating_text": str(rating_value)
                            },
                            weights=weights
                        )
                
                # Mark as processed
                self.processed_entities.add(user_id)
                
            except Exception as e:
                self.logger.error(f"Error processing user {user_id}: {str(e)}")
                traceback.print_exc()
        
        self.logger.info(f"Processed {len(self.user_data)} users into hypergraph")
    
    async def build_enhanced_graph(self, movie_titles: List[str],
                                 process_users: bool = True,
                                 generate_embeddings: bool = True) -> Tuple[nx.Graph, nx.DiGraph]:
        """Build enhanced knowledge graph"""
        start_time = time.time()
        self.logger.info(f"Building enhanced knowledge graph from {len(movie_titles)} movies")
        
        try:
            # Load user data if processing users
            if process_users:
                await self.load_user_data()
                await self.load_ratings_data()
            
            # Process movies
            await self._process_movies(movie_titles)
            
            # Process users if requested
            if process_users:
                await self._process_users()
            
            # Generate standard graphs from hypergraph
            self.graph, self.digraph = self.hypergraph.to_networkx()
            
            # Generate embeddings if requested
            if generate_embeddings:
                # First generate semantic embeddings
                entities = []
                for node_id, data in self.graph.nodes(data=True):
                    node_type = data.get("node_type", "unknown")
                    entities.append((data, node_type, node_id))
                
                # Generate in efficient batches
                semantic_embeddings = await self.embedding_generator.generate_semantic_embeddings_efficiently(
                    entities, batch_size=200
                )
                
                # Generate structural embeddings
                # Group nodes by type for heterogeneous graph model
                nodes_by_type = defaultdict(list)
                for node_id, data in self.graph.nodes(data=True):
                    nodes_by_type[data.get("node_type", "unknown")].append(node_id)
                
                self.logger.info(f"Generating structural embeddings for {len(semantic_embeddings)} nodes")
                
                # For very large graphs, use a subset for GNN training
                if len(semantic_embeddings) > 10000:
                    self.logger.info(f"Graph is very large ({len(semantic_embeddings)} nodes), using subset for GNN")
                    subset_size = 10000
                    subset_entities = random.sample(list(semantic_embeddings.keys()), subset_size)
                    subset_semantics = {node: semantic_embeddings[node] for node in subset_entities}
                    
                    # Generate structural embeddings on subset
                    structural_embeddings = self.embedding_generator.generate_structural_embeddings(
                        self.graph, subset_semantics, nodes_by_type
                    )
                    
                    # For nodes not in subset, use semantic embeddings as fallback
                    for node in semantic_embeddings:
                        if node not in structural_embeddings:
                            structural_embeddings[node] = semantic_embeddings[node]
                else:
                    # Generate structural embeddings for all nodes
                    structural_embeddings = self.embedding_generator.generate_structural_embeddings(
                        self.graph, semantic_embeddings, nodes_by_type
                    )
                
                # Combine embeddings
                self.embeddings = self.embedding_generator.combine_embeddings(
                    semantic_embeddings, structural_embeddings
                )
                
                # Generate temporal embeddings using Graph ODE
                # Only for entities with temporal information (e.g., movies, people)
                temporal_entities = {}
                for node_id, data in self.graph.nodes(data=True):
                    if node_id in self.embeddings and "year" in data:
                        temporal_entities[node_id] = self.embeddings[node_id]
                print(len(self.embeddings))
                # Initialize ODE layer in hypergraph
                if temporal_entities:
                    self.logger.info(f"Generating temporal embeddings for {len(temporal_entities)} nodes")
                    self.hypergraph.generate_temporal_embeddings(
                        temporal_entities,
                        train_model=True
                    )
            
            # Detect communities
            self.logger.info("Detecting communities")
            self.communities = self.community_detector.detect_communities(self.graph, method="louvain")
            
            # Detect overlapping communities
            self.overlapping_communities = self.community_detector.detect_overlapping_communities(
                self.graph, self.embeddings, overlap_threshold=0.3
            )
            
            # Detect hierarchical communities
            self.hierarchical_communities = self.community_detector.detect_hierarchical_communities(
                self.graph, levels=3
            )
            
            # Enhance communities with embeddings
            self.enhanced_communities = self.community_detector.enhance_communities_with_embeddings(
                self.graph, self.communities, self.embeddings
            )
            
            # Find community bridges
            self.community_bridges = self.community_detector.find_community_bridges(
                self.graph, self.communities
            )
            
            # Characterize communities
            self.community_characteristics = self.community_detector.characterize_communities(
                self.graph, self.communities
            )
            
            # Add community information to graph nodes
            for node, comm_id in self.communities.items():
                if node in self.graph.nodes:
                    self.graph.nodes[node]["community_id"] = comm_id
                
                if node in self.digraph.nodes:
                    self.digraph.nodes[node]["community_id"] = comm_id
            
            # Add enhanced community information
            for node, comm_id in self.enhanced_communities.items():
                if node in self.graph.nodes:
                    self.graph.nodes[node]["enhanced_community_id"] = comm_id
                
                if node in self.digraph.nodes:
                    self.digraph.nodes[node]["enhanced_community_id"] = comm_id
            
            # Add overlapping community information
            for node, comms in self.overlapping_communities.items():
                if node in self.graph.nodes and comms:
                    # Store primary community and membership strength
                    primary_comm = comms[0]
                    self.graph.nodes[node]["primary_community"] = primary_comm[0]
                    self.graph.nodes[node]["primary_membership"] = primary_comm[1]
                    
                    # Store all communities as JSON
                    self.graph.nodes[node]["overlapping_communities"] = json.dumps(comms)
                
                # Do the same for digraph
                if node in self.digraph.nodes and comms:
                    primary_comm = comms[0]
                    self.digraph.nodes[node]["primary_community"] = primary_comm[0]
                    self.digraph.nodes[node]["primary_membership"] = primary_comm[1]
                    
                    # Store all communities as JSON
                    self.digraph.nodes[node]["overlapping_communities"] = json.dumps(comms)
            
            # Initialize path reasoning for explanations
            self.path_reasoner = PathReasoning(self.graph, self.embeddings)
            
            # Initialize multi-objective recommender
            self.recommender = MultiObjectiveRecommender(
                self.graph, 
                self.embeddings, 
                self.path_reasoner,
                self.hypergraph.node_embeddings if hasattr(self.hypergraph, 'node_embeddings') else None
            )
            
            # Compute hypergraph statistics
            self.hypergraph.compute_statistics()
            
            duration = time.time() - start_time
            self.logger.info(f"Built enhanced knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges in {duration:.2f} seconds")
            
            return self.graph, self.digraph
            
        except Exception as e:
            self.logger.error(f"Error building enhanced graph: {str(e)}")
            traceback.print_exc()
            
            # Return whatever we have so far
            if not self.graph:
                self.graph, self.digraph = self.hypergraph.to_networkx()
            
            return self.graph, self.digraph
    
    async def generate_recommendations_for_user(self, user_id: str, 
                                           num_recommendations: int = 5) -> List[Dict]:
        """Generate personalized movie recommendations for a specific user"""
        self.logger.info(f"Generating personalized recommendations for user {user_id}")
        
        # Ensure recommender is initialized
        if self.recommender is None:
            if self.graph and self.embeddings:
                self.path_reasoner = PathReasoning(self.graph, self.embeddings)
                self.recommender = MultiObjectiveRecommender(
                    self.graph, 
                    self.embeddings, 
                    self.path_reasoner,
                    self.hypergraph.node_embeddings if hasattr(self.hypergraph, 'node_embeddings') else None
                )
            else:
                self.logger.error("Cannot generate recommendations: graph or embeddings not available")
                return []
        
        # Convert user_id to node_id format if needed
        if not user_id.startswith("user_"):
            user_node_id = f"user_{self.data_collector.normalize_entity_name(user_id)}"
        else:
            user_node_id = user_id
        
        # Check if user exists in graph
        if user_node_id not in self.graph.nodes:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        # Use multi-objective recommender to generate recommendations
        recommendations = self.recommender.generate_recommendations(user_node_id, num_recommendations)
        return recommendations
    
    def save_graph(self, prefix: str = "bollywood_kg_enhanced"):
        """Save graph to files"""
        self.logger.info(f"Saving enhanced graph to {self.output_dir} with prefix {prefix}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save hypergraph
        hypergraph_path = os.path.join(self.output_dir, f"{prefix}_hypergraph.pkl")
        self.hypergraph.save(hypergraph_path)
        
        # Save NetworkX graphs as pickle files
        graph_path = os.path.join(self.output_dir, f"{prefix}_undirected.pkl")
        digraph_path = os.path.join(self.output_dir, f"{prefix}_directed.pkl")
        
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        with open(digraph_path, 'wb') as f:
            pickle.dump(self.digraph, f)
        
        # Save in GraphML format for neural model - with careful handling of None values
        graphml_path = os.path.join(self.output_dir, f"{prefix}_neural.graphml")
        
        # Save optimized graph for neural model
        try:
            # For GraphML, we need to clean certain attributes
            graph_for_ml = self.graph.copy()
            
            # Process nodes - convert non-compatible attributes and ensure no None values
            for node, attrs in graph_for_ml.nodes(data=True):
                for key, value in list(attrs.items()):
                    # Handle None values - convert to empty string
                    if value is None:
                        attrs[key] = ""
                        continue
                        
                    # Handle embeddings (convert to string)
                    if key == "embedding":
                        if isinstance(value, bytes):
                            attrs[key] = value.hex()
                        elif isinstance(value, np.ndarray):
                            attrs[key] = value.tobytes().hex()
                        else:
                            # Remove if not convertible
                            del attrs[key]
                    
                    # Handle JSON strings (keep as is - GraphML can handle strings)
                    elif isinstance(value, (list, dict, set)):
                        try:
                            attrs[key] = json.dumps(value)
                        except:
                            # Remove if not serializable
                            del attrs[key]
                    
                    # Handle non-string keys (GraphML requires string keys)
                    if not isinstance(key, str):
                        new_key = str(key)
                        attrs[new_key] = attrs.pop(key)
            
            # Process edges - convert non-compatible attributes
            for u, v, attrs in graph_for_ml.edges(data=True):
                for key, value in list(attrs.items()):
                    # Handle None values
                    if value is None:
                        attrs[key] = ""
                        continue
                        
                    # Handle non-primitive types
                    if isinstance(value, (list, dict, set)):
                        try:
                            attrs[key] = json.dumps(value)
                        except:
                            # Remove if not serializable
                            del attrs[key]
                    
                    # Handle non-string keys
                    if not isinstance(key, str):
                        new_key = str(key)
                        attrs[new_key] = attrs.pop(key)
            
            # Write GraphML - using nx.write_graphml with explicit encoders
            nx.write_graphml(
                graph_for_ml, 
                graphml_path,
                encoding='utf-8',
                prettyprint=True
            )
            self.logger.info(f"Saved GraphML file to {graphml_path}")
        except Exception as e:
            self.logger.error(f"Error saving GraphML: {str(e)}")
            traceback.print_exc()
            # Try alternative approach with simplified graph
            try:
                simplified_graph = nx.Graph()
                # Add nodes with minimal attributes
                for node, data in self.graph.nodes(data=True):
                    simplified_graph.add_node(
                        node,
                        node_type=str(data.get("node_type", "")),
                        name=str(data.get("name", data.get("title", "")))
                    )
                # Add edges with minimal attributes
                for u, v, data in self.graph.edges(data=True):
                    simplified_graph.add_edge(
                        u, v, 
                        edge_type=str(data.get("edge_type", ""))
                    )
                # Save simplified graph
                nx.write_graphml(simplified_graph, graphml_path)
                self.logger.info(f"Saved simplified GraphML file to {graphml_path}")
            except Exception as e2:
                self.logger.error(f"Error saving simplified GraphML: {str(e2)}")
        
        # Save node data separately as JSON for better compatibility
        nodes_path = os.path.join(self.output_dir, f"{prefix}_nodes.json")
        
        # Prepare node data (convert non-serializable attributes)
        nodes_data = {}
        for node, data in self.graph.nodes(data=True):
            # Create a serializable copy of the node data
            node_data = {}
            for key, value in data.items():
                if value is None:
                    node_data[key] = ""
                elif isinstance(value, (bytes, bytearray)):
                    node_data[key] = value.hex()
                elif isinstance(value, np.ndarray):
                    node_data[key] = value.tolist()
                elif isinstance(value, (dict, list, set)):
                    try:
                        # Try to convert any collection to JSON
                        node_data[key] = json.dumps(value)
                    except:
                        # If that fails, convert to string
                        node_data[key] = str(value)
                elif isinstance(value, (str, int, float, bool)):
                    node_data[key] = value
                else:
                    node_data[key] = str(value)
            nodes_data[str(node)] = node_data
        
        # Save nodes data
        with open(nodes_path, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2)
        
        # Save embeddings separately
        if self.embeddings:
            embeddings_path = os.path.join(self.output_dir, f"{prefix}_embeddings.pkl")
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        
        # Save communities
        if self.communities:
            communities_path = os.path.join(self.output_dir, f"{prefix}_communities.json")
            with open(communities_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "basic": {str(k): v for k, v in self.communities.items()},
                    "enhanced": {str(k): v for k, v in self.enhanced_communities.items()},
                    "overlapping": {str(k): v for k, v in self.overlapping_communities.items()},
                    "hierarchical": {str(k): {str(level): comm for level, comm in v.items()} 
                                for k, v in self.hierarchical_communities.items()},
                    "community_bridges": [str(node) for node in self.community_bridges[:50]],
                    "community_characteristics": {str(k): v for k, v in self.community_characteristics.items()}
                }, f, indent=2)
        
        # Save movie TT mapping
        mapping_path = os.path.join(self.output_dir, f"{prefix}_movie_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.movie_tt_mapping, f, indent=2)
        
        # Save hypergraph statistics
        stats_path = os.path.join(self.output_dir, f"{prefix}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            # Convert stats to serializable format
            stats = self.hypergraph.stats.copy()
            for key, value in list(stats.items()):
                if isinstance(value, (datetime, np.ndarray)):
                    stats[key] = str(value)
                elif isinstance(value, dict):
                    # Convert nested dictionaries with non-string keys
                    stats[key] = {str(k): v for k, v in value.items()}
            
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved all enhanced graph files to {self.output_dir}")
    
    def evaluate_graph_quality(self) -> Dict:
        """Evaluate the quality of the built graph"""
        self.logger.info("Evaluating graph quality")
        
        metrics = {}
        
        # Basic graph metrics
        metrics["num_nodes"] = self.graph.number_of_nodes()
        metrics["num_edges"] = self.graph.number_of_edges()
        metrics["density"] = nx.density(self.graph)
        
        try:
            # Use approximation for large graphs
            if self.graph.number_of_nodes() > 5000:
                # Sample 1000 nodes for clustering coefficient
                sample_nodes = random.sample(list(self.graph.nodes()), 1000)
                sample_graph = self.graph.subgraph(sample_nodes)
                metrics["average_clustering_sample"] = nx.average_clustering(sample_graph)
            else:
                metrics["average_clustering"] = nx.average_clustering(self.graph)
        except Exception as e:
            self.logger.warning(f"Error computing clustering coefficient: {str(e)}")
            metrics["average_clustering"] = None
        
        try:
            # For large graphs, estimate average shortest path on a sample
            if self.graph.number_of_nodes() > 5000:
                # Sample 1000 random nodes
                sample_nodes = random.sample(list(self.graph.nodes()), 1000)
                sample_graph = self.graph.subgraph(sample_nodes)
                
                # Get largest connected component
                largest_cc = max(nx.connected_components(sample_graph), key=len)
                cc_graph = sample_graph.subgraph(largest_cc)
                
                # Compute average shortest path on component using approximation
                # Sample 100 nodes for path calculations to speed up
                path_sample = random.sample(list(cc_graph.nodes()), min(100, len(cc_graph)))
                path_lengths = []
                
                for source in path_sample:
                    lengths = nx.single_source_shortest_path_length(cc_graph, source)
                    path_lengths.extend(lengths.values())
                
                if path_lengths:
                    metrics["average_shortest_path_sample"] = np.mean(path_lengths)
            else:
                # For small graphs, compute on largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                cc_graph = self.graph.subgraph(largest_cc)
                metrics["average_shortest_path"] = nx.average_shortest_path_length(cc_graph)
        except Exception as e:
            self.logger.warning(f"Error computing shortest paths: {str(e)}")
            metrics["average_shortest_path"] = None
        
        # Node type distribution
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        metrics["node_type_distribution"] = node_types
        
        # Edge type distribution
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        metrics["edge_type_distribution"] = edge_types
        
        # Community metrics
        if self.communities:
            community_sizes = {}
            for comm_id in set(self.communities.values()):
                size = list(self.communities.values()).count(comm_id)
                community_sizes[str(comm_id)] = size
            
            metrics["community_sizes"] = community_sizes
            metrics["num_communities"] = len(community_sizes)
            
            try:
                # Calculate modularity
                community_sets = {}
                for node, comm_id in self.communities.items():
                    if comm_id not in community_sets:
                        community_sets[comm_id] = set()
                    community_sets[comm_id].add(node)
                
                metrics["modularity"] = nx.algorithms.community.modularity(
                    self.graph, community_sets.values()
                )
            except Exception as e:
                self.logger.warning(f"Error computing modularity: {str(e)}")
                metrics["modularity"] = None
        
        # User metrics
        user_nodes = [node for node, data in self.graph.nodes(data=True) 
                    if data.get("node_type") == "user"]
        metrics["num_users"] = len(user_nodes)
        
        # Genre metrics
        genre_nodes = [node for node, data in self.graph.nodes(data=True) 
                     if data.get("node_type") == "genre"]
        metrics["num_genres"] = len(genre_nodes)
        
        # Rating metrics
        rating_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) 
                       if d.get("edge_type") == "RATED"]
        metrics["num_ratings"] = len(rating_edges)
        
        # Rating distribution
        rating_distribution = {"-1": 0, "0": 0, "1": 0}
        for _, _, data in rating_edges:
            rating = data.get("rating", 0)
            rating_str = str(rating)
            if rating_str in rating_distribution:
                rating_distribution[rating_str] += 1
        
        metrics["rating_distribution"] = rating_distribution
        
        # Temporal distribution
        temporal_distribution = {}
        for node, data in self.graph.nodes(data=True):
            if "year" in data:
                year = data["year"]
                if isinstance(year, str) and year.isdigit():
                    year = int(year)
                if isinstance(year, int):
                    temporal_distribution[str(year)] = temporal_distribution.get(str(year), 0) + 1
        
        metrics["temporal_distribution"] = temporal_distribution
        
        # Hypergraph metrics
        if hasattr(self.hypergraph, 'stats'):
            metrics["hypergraph"] = {
                str(k): v for k, v in self.hypergraph.stats.items()
                if not isinstance(v, dict)  # Handle nested dicts separately
            }
            
            # Handle nested dictionaries
            for k, v in self.hypergraph.stats.items():
                if isinstance(v, dict):
                    metrics["hypergraph"][str(k)] = {str(sk): sv for sk, sv in v.items()}
        
        # Embeddings metrics
        if self.embeddings:
            metrics["num_embeddings"] = len(self.embeddings)
            
            # Calculate average embedding similarity between nodes of same type
            if len(self.embeddings) > 1:
                # Sample up to 1000 pairs for efficiency
                node_pairs = []
                node_types = defaultdict(list)
                
                # Group nodes by type
                for node, data in self.graph.nodes(data=True):
                    if node in self.embeddings:
                        node_type = data.get("node_type", "unknown")
                        node_types[node_type].append(node)
                
                # Sample pairs from same type
                for node_type, nodes in node_types.items():
                    if len(nodes) >= 2:
                        # Sample up to 100 pairs from each type
                        num_pairs = min(100, len(nodes) * (len(nodes) - 1) // 2)
                        for _ in range(num_pairs):
                            n1, n2 = random.sample(nodes, 2)
                            node_pairs.append((n1, n2, node_type))
                
                # Calculate similarities
                similarity_by_type = defaultdict(list)
                for n1, n2, node_type in node_pairs:
                    if n1 in self.embeddings and n2 in self.embeddings:
                        sim = cosine_similarity(
                            self.embeddings[n1].reshape(1, -1),
                            self.embeddings[n2].reshape(1, -1)
                        )[0][0]
                        similarity_by_type[node_type].append(sim)
                
                # Calculate average similarity by type
                avg_similarity = {}
                for node_type, sims in similarity_by_type.items():
                    if sims:
                        avg_similarity[node_type] = float(np.mean(sims))
                
                metrics["avg_embedding_similarity_by_type"] = avg_similarity
        
        # Store evaluation metrics
        eval_path = os.path.join(self.output_dir, "graph_evaluation.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            # Convert to serializable format
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    serializable_metrics[key] = {str(k): v for k, v in value.items()}
                elif isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Evaluated graph quality and saved to {eval_path}")
        return metrics
    
    def visualize_graph(self, max_nodes: int = 1000, output_file: str = None):
        """Visualize graph structure (limited to max_nodes)"""
        self.logger.info(f"Visualizing graph (max {max_nodes} nodes)")
        
        try:
            # Create a smaller graph if needed
            if self.graph.number_of_nodes() > max_nodes:
                # Get most important nodes based on degree and node type
                # Prioritize movies, actors, users
                priority_types = ["movie", "actor", "user", "director", "genre"]
                
                # Sort nodes by priority type and degree
                node_priorities = []
                for node, data in self.graph.nodes(data=True):
                    node_type = data.get("node_type", "unknown")
                    degree = self.graph.degree(node)
                    
                    # Calculate priority score (higher is better)
                    # Type priority + log degree
                    type_priority = priority_types.index(node_type) if node_type in priority_types else 99
                    priority = type_priority * -1000 + np.log1p(degree)  # Negative so higher values = higher priority
                    
                    node_priorities.append((node, priority))
                
                # Sort by priority (higher values first)
                node_priorities.sort(key=lambda x: x[1], reverse=True)
                
                # Take top nodes
                top_nodes = [node for node, _ in node_priorities[:max_nodes]]
                
                # Create subgraph
                subgraph = self.graph.subgraph(top_nodes)
            else:
                subgraph = self.graph
            
            # Set up visualization
            plt.figure(figsize=(18, 18))
            
            # Use community information for colors if available
            if self.communities:
                # Create a colormap for communities
                import matplotlib.cm as cm
                from matplotlib.colors import ListedColormap
                
                # Get unique communities
                unique_communities = sorted(set(v for v in self.communities.values() if v is not None))
                n_communities = len(unique_communities)
                
                # Create a map from community ID to index
                comm_to_idx = {comm: i for i, comm in enumerate(unique_communities)}
                
                # Get node colors
                node_colors = []
                for node in subgraph.nodes():
                    if node in self.communities:
                        comm_idx = comm_to_idx.get(self.communities[node], 0)
                        node_colors.append(comm_idx)
                    else:
                        node_colors.append(0)
                
                # Create colormap (using tab20 for more distinct colors)
                # Using plt.colormaps instead of cm.get_cmap to avoid deprecation warning
                if hasattr(plt, 'colormaps'):
                    cmap = plt.colormaps['tab20']
                else:
                    # Fallback for older matplotlib versions
                    cmap = cm.get_cmap('tab20')
                
                # Create node size based on degree and type
                node_sizes = []
                for node in subgraph.nodes():
                    base_size = 20 + 5 * subgraph.degree(node)
                    
                    # Adjust size based on node type
                    node_type = subgraph.nodes[node].get("node_type", "")
                    if node_type == "movie":
                        base_size *= 1.5
                    elif node_type == "user":
                        base_size *= 1.2
                    elif node_type == "genre":
                        base_size *= 1.8
                    
                    node_sizes.append(base_size)
                
                # Draw graph with communities as colors
                pos = nx.spring_layout(subgraph, seed=42, k=0.15, iterations=100)
                nx.draw_networkx(
                    subgraph,
                    pos=pos,
                    node_color=node_colors,
                    cmap=cmap,
                    node_size=node_sizes,
                    font_size=8,
                    width=0.5,
                    alpha=0.8,
                    with_labels=False
                )
                
                # Add labels for important nodes
                important_nodes = {}
                for node in subgraph.nodes():
                    # Label nodes with high degree or specific types
                    node_type = subgraph.nodes[node].get("node_type", "unknown")
                    if (subgraph.degree(node) > 10 or 
                        node_type in ["genre"] or
                        (node_type == "movie" and subgraph.degree(node) > 5)):
                        
                        if node_type == "movie":
                            important_nodes[node] = subgraph.nodes[node].get("title", node)
                        elif node_type in ["person", "actor", "director"]:
                            important_nodes[node] = subgraph.nodes[node].get("name", node)
                        elif node_type == "genre":
                            important_nodes[node] = subgraph.nodes[node].get("name", node)
                        elif node_type == "user":
                            important_nodes[node] = subgraph.nodes[node].get("name", node)
                
                nx.draw_networkx_labels(
                    subgraph,
                    pos,
                    labels=important_nodes,
                    font_size=10,
                    font_weight='bold'
                )
                
                plt.title(f"Bollywood Knowledge Graph with Communities ({n_communities} communities)")
                
            else:
                # Color by node type
                color_map = {
                    "movie": "red",
                    "person": "blue",
                    "actor": "darkblue",
                    "director": "royalblue",
                    "genre": "green",
                    "theme": "purple",
                    "decade": "orange",
                    "award": "yellow",
                    "user": "pink",
                    "language": "cyan",
                    "state": "brown",
                    "occupation": "gray",
                    "hyperedge": "black"
                }
                
                node_colors = [color_map.get(subgraph.nodes[n].get("node_type", ""), "gray") 
                             for n in subgraph.nodes()]
                
                # Node sizes based on degree
                node_sizes = [20 + 10 * subgraph.degree(n) for n in subgraph.nodes()]
                
                # Draw graph
                pos = nx.spring_layout(subgraph, seed=42, iterations=100)
                nx.draw_networkx(
                    subgraph,
                    pos=pos,
                    node_color=node_colors,
                    node_size=node_sizes,
                    font_size=8,
                    width=0.5,
                    alpha=0.8,
                    with_labels=False
                )
                
                # Add legend
                plt.title(f"Bollywood Knowledge Graph (Node Types)")
                # Collect node types present in the subgraph
                present_node_types = {subgraph.nodes[n].get("node_type", "unknown") for n in subgraph.nodes()}
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, label=node_type)
                                for node_type, color in color_map.items() if node_type in present_node_types]
                plt.legend(handles=legend_elements)
            
            # Save to file if specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            else:
                output_file = os.path.join(self.output_dir, "graph_visualization.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            self.logger.info(f"Saved graph visualization to {output_file}")
            
            # Generate additional visualizations
            
            # 1. Community size distribution
            if self.communities:
                plt.figure(figsize=(12, 8))
                community_sizes = Counter(self.communities.values())
                sizes = sorted(community_sizes.items())
                
                plt.bar([str(comm) for comm, _ in sizes], [size for _, size in sizes])
                plt.title("Community Size Distribution")
                plt.xlabel("Community ID")
                plt.ylabel("Number of Nodes")
                plt.xticks(rotation=90)
                
                comm_dist_file = os.path.join(self.output_dir, "community_distribution.png")
                plt.savefig(comm_dist_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved community distribution visualization to {comm_dist_file}")
            
            # 2. Temporal distribution of movies
            movie_years = {}
            for node, data in self.graph.nodes(data=True):
                if data.get("node_type") == "movie" and "year" in data:
                    year = data["year"]
                    if isinstance(year, str) and year.isdigit():
                        year = int(year)
                    if isinstance(year, int):
                        movie_years[year] = movie_years.get(year, 0) + 1
            
            if movie_years:
                plt.figure(figsize=(15, 8))
                years = sorted(movie_years.keys())
                movies_count = [movie_years[year] for year in years]
                
                plt.bar(years, movies_count)
                plt.title("Temporal Distribution of Movies")
                plt.xlabel("Year")
                plt.ylabel("Number of Movies")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                temp_dist_file = os.path.join(self.output_dir, "temporal_distribution.png")
                plt.savefig(temp_dist_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved temporal distribution visualization to {temp_dist_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            traceback.print_exc()



#main function

async def run_enhanced_kg_builder(api_key: str,
                                movie_titles: List[str],
                                users_file: str,
                                ratings_file: str,
                                output_dir: str = "final_enhanced_output",
                                movie_mapping_file: str = "movie_id_mapping.json"):
    """
    Run the enhanced knowledge graph builder
    
    Args:
        api_key: OpenAI API key
        movie_titles: List of movies to include
        users_file: Path to users CSV file
        ratings_file: Path to ratings JSON file
        output_dir: Directory to save output
        movie_mapping_file: Path to movie ID mapping file
    
    Returns:
        Graph, digraph, hypergraph, and evaluation metrics
    """
    # Set up logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loggers = setup_logging(log_dir)
    
    logger = loggers["graph_builder"]
    logger.info(f"Starting enhanced graph builder with {len(movie_titles)} movies")
    
    # Initialize builder
    builder = EnhancedBollywoodKGBuilder(
        api_key=api_key,
        data_dir="data",  # Use existing data directory
        cache_dir="cache",  # Use existing cache directory
        output_dir=output_dir,  # Use new output directory
        users_file=users_file,
        ratings_file=ratings_file,
        movie_mapping_file=movie_mapping_file
    )
    
    # Build graph
    start_time = time.time()
    
    graph, digraph = await builder.build_enhanced_graph(
        movie_titles=movie_titles,
        process_users=True,
        generate_embeddings=True
    )
    
    build_time = time.time() - start_time
    
    logger.info(f"Enhanced graph building completed in {build_time:.2f} seconds")
    logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Save graph
    builder.save_graph(prefix="bollywood_kg_enhanced")
    
    # Evaluate graph quality
    metrics = builder.evaluate_graph_quality()
    
    # Create visualization
    builder.visualize_graph(max_nodes=500, output_file=os.path.join(output_dir, "enhanced_graph_viz.png"))
    
    # Generate sample recommendations
    recommendations = {}
    
    # Take a few random users
    user_nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "user"]
    if user_nodes:
        sample_users = random.sample(user_nodes, min(5, len(user_nodes)))
        
        for user_node in sample_users:
            # Extract user ID (remove "user_" prefix)
            if user_node.startswith("user_"):
                user_id = user_node[5:]
            else:
                user_id = user_node
                
            # Generate recommendations
            user_recs = await builder.generate_recommendations_for_user(user_id, num_recommendations=5)
            
            # Add to results
            recommendations[user_id] = user_recs
    
    # Save recommendations
    if recommendations:
        rec_path = os.path.join(output_dir, "sample_recommendations.json")
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, indent=2)
    
    logger.info(f"Enhanced knowledge graph builder completed. Results saved to {output_dir}")
    
    return graph, digraph, builder.hypergraph, metrics, recommendations

# Entry point for command-line usage
async def run_main():
    """Main entry point for command-line usage"""
    # Load configuration
    config_path = "config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "output_dir": "final_enhanced_output",
            "movie_list": [],
            "users_file": "users.csv",
            "ratings_file": "ratings.json",
            "movie_mapping_file": "movie_id_mapping.json"
        }
    
    # Check for API key
    api_key = config.get("api_key")
    if not api_key:
        print("Error: OpenAI API key not provided in config or environment")
        return
    
    # Get movie list
    movie_titles = config.get("movie_list", [])
    
    # If no movies specified, use a default list
    if not movie_titles:
        # Get all movie files from data directory
        movies_dir = "data/movies"
        if os.path.exists(movies_dir):
            movie_files = [f for f in os.listdir(movies_dir) if f.endswith('.json')]
            movie_titles = [os.path.splitext(f)[0].replace("_", " ") for f in movie_files]
            
            # Limit to a reasonable number if needed
            if len(movie_titles) > 100:
                # Sample to get a good distribution
                print(f"Found {len(movie_titles)} movies, sampling 100 for processing...")
                movie_titles = random.sample(movie_titles, 100)
        else:
            # Default list if no data directory
            movie_titles = [
                "3 Idiots",
                "Sholay",
                "Dilwale Dulhania Le Jayenge",
                "PK",
                "Lagaan",
                "Kabhi Khushi Kabhie Gham",
                "Bajrangi Bhaijaan",
                "Mother India",
                "Dangal",
                "Queen"
            ]
    
    # Run builder
    await run_enhanced_kg_builder(
        api_key=api_key,
        movie_titles=movie_titles,
        users_file="/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/users.csv",
        ratings_file="/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/ratings_array.json",
        output_dir="output_dir",
        movie_mapping_file="/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/movie_id_mapping.json"
    )

if __name__ == "__main__":
    asyncio.run(run_main())