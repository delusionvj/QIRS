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
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Iterator
from sklearn.metrics.pairwise import cosine_similarity
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
def setup_logging(log_dir="logs", level=logging.DEBUG):
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
        "genre_processing": logging.getLogger("genre_processing")
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


class TemporalHypergraph:
    "Temporal hypergraph representation for knowledge graph"

    def __init__(self, name: str = "bollywood_hypergraph"):
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
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved hypergraph with {len(self.nodes)} nodes and {len(self.hyperedges)} hyperedges")
    
    @classmethod
    def load(cls, file_path: str) -> 'TemporalHypergraph':
        """Load hypergraph from file"""
        logger = logging.getLogger("hypergraph")
        logger.info(f"Loading hypergraph from {file_path}")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Create hypergraph
        hypergraph = cls(name=data["name"])
        
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

# Simplified Embedding Generator - Textual Only
class TextualEmbeddingGenerator:
    """
    Simplified embedding generator that only uses textual embeddings
    """
    
    def __init__(self, api_client: EnhancedAPIClient, embeddings_dir: str = "embeddings"):
        """
        Initialize textual embedding generator
        
        Args:
            api_client: API client for text embeddings
            embeddings_dir: Directory to store embeddings
        """
        self.api_client = api_client
        self.embeddings_dir = embeddings_dir
        self.embedding_model = "text-embedding-3-small"
        self.semantic_dim = 1536  # Dimensionality of text-embedding-3-small
        
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Create subdirectories for embeddings
        os.makedirs(os.path.join(embeddings_dir, "textual"), exist_ok=True)
        
        self.logger = logging.getLogger("embeddings")
        self.logger.info(f"Initialized Textual Embedding Generator")
    
    def _get_embedding_cache_path(self, entity_type: str, entity_id: str) -> str:
        """Get file path for cached embedding"""
        sanitized_id = entity_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.embeddings_dir, "textual", f"{entity_type}_{sanitized_id}.npy")
    
    def _save_embedding(self, entity_type: str, entity_id: str, embedding: np.ndarray):
        """Save embedding to cache"""
        file_path = self._get_embedding_cache_path(entity_type, entity_id)
        np.save(file_path, embedding)
        self.logger.debug(f"Saved textual embedding for {entity_type} {entity_id}")
    
    def _load_embedding(self, entity_type: str, entity_id: str) -> Optional[np.ndarray]:
        """Load embedding from cache if available"""
        file_path = self._get_embedding_cache_path(entity_type, entity_id)
        
        if os.path.exists(file_path):
            try:
                embedding = np.load(file_path)
                self.logger.debug(f"Loaded cached textual embedding for {entity_type} {entity_id}")
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
    
    async def generate_textual_embedding(self, entity_data: Dict, entity_type: str, 
                                     entity_id: str, force_refresh: bool = False) -> np.ndarray:
        """Generate textual (text-based) embedding for an entity"""
        # Check cache unless forced refresh
        if not force_refresh:
            cached_embedding = self._load_embedding(entity_type, entity_id)
            if cached_embedding is not None:
                return cached_embedding
        
        # Create rich description
        description = self._create_entity_description(entity_data, entity_type)
        
        self.logger.info(f"Generating textual embedding for {entity_type} {entity_id}")
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
            self._save_embedding(entity_type, entity_id, embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Exception generating embedding: {str(e)}")
            traceback.print_exc()
            # Return zero embedding as fallback
            return np.zeros(self.semantic_dim)
    
    async def generate_textual_embeddings_efficiently(self, entities: List[Tuple[Dict, str, str]], 
                                                batch_size: int = 200) -> Dict[str, np.ndarray]:
        """Generate textual embeddings efficiently in optimized batches"""
        self.logger.info(f"Efficiently generating textual embeddings for {len(entities)} entities")
        
        results = {}
        batch_entity_ids = []
        batch_descriptions = []
        batch_entity_data = []
        batches = []
        
        # Prepare batches
        for i, (entity_data, entity_type, entity_id) in enumerate(entities):
            # Check cache first
            cached_embedding = self._load_embedding(entity_type, entity_id)
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
                    self._save_embedding(entity_type, entity_id, embedding_array)
                    
            except Exception as e:
                self.logger.error(f"Exception in batch {batch_idx}: {str(e)}")
                traceback.print_exc()
        
        self.logger.info(f"Efficiently generated {len(results)} textual embeddings")
        return results

class EnhancedCommunityDetector:
    """Community detection for knowledge graphs"""
    
    def __init__(self):
        """Initialize community detector"""
        self.logger = logging.getLogger("community")
        self.logger.info("Initialized community detector")
        
        # Cache for community detections (avoid recomputing)
        self.community_cache = {}
    
    def detect_communities(self, graph: nx.Graph, method: str = "louvain", 
                          resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in graph using selected method
        
        Args:
            graph: NetworkX graph
            method: Community detection method ('louvain')
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
        
        # Use the Louvain algorithm from the community library
        import community as community_louvain
        
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
            
            # Cache result
            self.community_cache[cache_key] = renumbered_communities
            
            return renumbered_communities
            
        except Exception as e:
            self.logger.error(f"Error detecting communities using Louvain: {str(e)}")
            traceback.print_exc()
            # Fallback: assign all nodes to community 0
            return {node: 0 for node in graph.nodes()}

class PathReasoning:
    """Path-based reasoning for knowledge graph"""
    
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
        self.logger.info("Initialized Path Reasoning module")
        
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
            # Use simple paths algorithm
            paths = list(nx.all_simple_paths(self.graph, start_node, end_node, cutoff=max_length))
            
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

class SimpleRecommender:
    """Simple recommendation engine using only textual embeddings"""
    
    def __init__(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray] = None,
                path_reasoner: PathReasoning = None):
        """
        Initialize recommender
        
        Args:
            graph: NetworkX graph
            embeddings: Textual embeddings dictionary
            path_reasoner: Path reasoning module for explanations
        """
        self.graph = graph
        self.embeddings = embeddings
        self.path_reasoner = path_reasoner
        
        self.logger = logging.getLogger("graph_builder")
        self.logger.info("Initialized Simple Recommender (textual only)")
    
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
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            traceback.print_exc()
            return preferences
    
    def compute_similarity(self, movie_id: str, user_embedding: np.ndarray) -> float:
        """
        Compute similarity between movie and user
        
        Args:
            movie_id: Movie node ID
            user_embedding: User embedding vector
            
        Returns:
            Similarity score [0, 1]
        """
        if self.embeddings is None or movie_id not in self.embeddings:
            return 0.5
        
        try:
            movie_embedding = self.embeddings[movie_id]
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                user_embedding.reshape(1, -1),
                movie_embedding.reshape(1, -1)
            )[0][0]
            
            return max(0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.5
    
    def generate_recommendations(self, user_id: str, n: int = 5) -> List[Dict]:
        """
        Generate recommendations based on user preferences and embeddings
        
        Args:
            user_id: User node ID
            n: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        self.logger.info(f"Generating {n} textual-based recommendations for user {user_id}")
        
        if user_id not in self.graph:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        try:
            # Get user preferences
            preferences = self.get_user_preferences(user_id)
            
            # Get already rated movies
            rated_movies = set(preferences["liked_movies"] + preferences["disliked_movies"])
            
            # Get user embedding
            user_embedding = None
            if self.embeddings and user_id in self.embeddings:
                user_embedding = self.embeddings[user_id]
            
            # If no user embedding, try to create centroid from liked movies
            if user_embedding is None and preferences["liked_embeddings"]:
                user_embedding = np.mean(preferences["liked_embeddings"], axis=0)
            
            # If still no embedding, we can't do similarity-based recommendations
            if user_embedding is None:
                self.logger.warning(f"No embedding available for user {user_id}")
                return []
            
            # Get all movie nodes
            movie_nodes = [node for node, data in self.graph.nodes(data=True) 
                         if data.get("node_type") == "movie"]
            
            # Filter out already rated movies
            candidates = [movie for movie in movie_nodes if movie not in rated_movies]
            
            # Score candidates by similarity to user
            scored_candidates = []
            
            for movie_id in candidates:
                similarity = self.compute_similarity(movie_id, user_embedding)
                
                # Add genre and other bonuses
                bonus = 0.0
                
                # Genre match bonus
                if preferences["genres"]:
                    movie_genres = self.graph.nodes[movie_id].get("genres", "[]")
                    if isinstance(movie_genres, str):
                        try:
                            genres = json.loads(movie_genres)
                            matching = sum(1 for g in genres if g in preferences["genres"])
                            if matching > 0 and len(genres) > 0:
                                bonus += 0.1 * (matching / len(preferences["genres"]))
                        except:
                            pass
                
                # Director match bonus
                if preferences["directors"]:
                    # Check if any directors match
                    for neighbor in self.graph.neighbors(movie_id):
                        if self.graph.nodes[neighbor].get("node_type") == "director":
                            director_name = self.graph.nodes[neighbor].get("name", "")
                            if director_name in preferences["directors"]:
                                bonus += 0.15
                                break
                
                # Actor match bonus
                if preferences["actors"]:
                    # Check if any actors match
                    actor_matches = 0
                    for neighbor in self.graph.neighbors(movie_id):
                        if self.graph.nodes[neighbor].get("node_type") == "actor":
                            actor_name = self.graph.nodes[neighbor].get("name", "")
                            if actor_name in preferences["actors"]:
                                actor_matches += 1
                    
                    if actor_matches > 0:
                        bonus += 0.1 * min(actor_matches, 3) / 3
                
                # Final score combines similarity and bonuses
                final_score = similarity + bonus
                
                scored_candidates.append((movie_id, final_score))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top n
            top_candidates = scored_candidates[:n]
            
            # Build recommendation objects
            recommendations = []
            
            for movie_id, score in top_candidates:
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
                    "score": score,
                    "explanation": explanation
                })
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return []


class BollywoodKGBuilder:
    """Knowledge graph builder with textual embeddings only"""
    
    def __init__(self, api_key: str, data_dir: str = "data", 
                cache_dir: str = "cache", output_dir: str = "text_output",
                users_file: str = "users.csv", 
                ratings_file: str = "ratings.json",
                movie_mapping_file: str = "movie_id_mapping.json"):
        
        self.api_key = api_key
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.users_file = "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/users.csv"
        self.ratings_file = "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/ratings_array.json"
        self.movie_mapping_file = "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/movie_id_mapping.json"

        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize API client
        self.api_client = EnhancedAPIClient(api_key, cache_dir)

        # Initialize data collector
        self.data_collector = FlexibleDataCollector(self.api_client, data_dir)
        
        # Initialize embedding generator - only textual embeddings
        self.embedding_generator = TextualEmbeddingGenerator(
            self.api_client,
            os.path.join(output_dir, "embeddings")
        )

        # Initialize hypergraph
        self.hypergraph = TemporalHypergraph(name="bollywood_kg_textual")
        
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
        
        # Initialize path reasoning (will be created after graph is built)
        self.path_reasoner = None
        
        # Initialize recommender (will be created after graph is built)
        self.recommender = None
        
        self.logger = logging.getLogger("graph_builder")
        self.logger.info("Initialized Bollywood KG Builder with textual embeddings only")

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
    
    async def build_graph(self, movie_titles: List[str],
                       process_users: bool = True,
                       generate_embeddings: bool = True) -> Tuple[nx.Graph, nx.DiGraph]:
        """Build knowledge graph with textual embeddings only"""
        start_time = time.time()
        self.logger.info(f"Building knowledge graph from {len(movie_titles)} movies with textual embeddings only")
        
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
                # Generate textual embeddings
                entities = []
                for node_id, data in self.graph.nodes(data=True):
                    node_type = data.get("node_type", "unknown")
                    entities.append((data, node_type, node_id))
                
                # Generate in efficient batches
                self.embeddings = await self.embedding_generator.generate_textual_embeddings_efficiently(
                    entities, batch_size=200
                )
                
                self.logger.info(f"Generated textual embeddings for {len(self.embeddings)} nodes")
            
            # Detect communities
            self.logger.info("Detecting communities")
            self.communities = self.community_detector.detect_communities(self.graph, method="louvain")
            
            # Add community information to graph nodes
            for node, comm_id in self.communities.items():
                if node in self.graph.nodes:
                    self.graph.nodes[node]["community_id"] = comm_id
                
                if node in self.digraph.nodes:
                    self.digraph.nodes[node]["community_id"] = comm_id
            
            # Initialize path reasoning for explanations
            self.path_reasoner = PathReasoning(self.graph, self.embeddings)
            
            # Initialize simple recommender
            self.recommender = SimpleRecommender(
                self.graph, 
                self.embeddings, 
                self.path_reasoner
            )
            
            # Compute hypergraph statistics
            self.hypergraph.compute_statistics()
            
            duration = time.time() - start_time
            self.logger.info(f"Built knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges in {duration:.2f} seconds")
            
            return self.graph, self.digraph
            
        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")
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
                self.recommender = SimpleRecommender(
                    self.graph, 
                    self.embeddings, 
                    self.path_reasoner
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
        
        # Use recommender to generate recommendations
        recommendations = self.recommender.generate_recommendations(user_node_id, num_recommendations)
        return recommendations
    
    def save_graph(self, prefix: str = "text_bollywood_kg"):
        """Save graph to files"""
        self.logger.info(f"Saving graph to {self.output_dir} with prefix {prefix}")
        
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
                        
                    # Handle embeddings (remove them from GraphML)
                    if key == "embedding":
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
                
                # Add nodes with essential attributes
                for node, data in self.graph.nodes(data=True):
                    # Include essential node attributes
                    node_attrs = {
                        "node_type": str(data.get("node_type", "")),
                        "name": str(data.get("name", data.get("title", ""))),
                    }
                    
                    # Include year if available
                    if "year" in data:
                        node_attrs["year"] = data["year"]
                        
                    # Include community ID if available
                    if "community_id" in data:
                        node_attrs["community_id"] = data["community_id"]
                        
                    simplified_graph.add_node(node, **node_attrs)
                
                # Add edges with standardized attributes
                for u, v, data in self.graph.edges(data=True):
                    edge_type = str(data.get("edge_type", "unknown"))
                    
                    # Create edge attributes ensuring consistent structure
                    edge_attrs = {
                        "edge_type": edge_type,
                        "weight": float(data.get("edge_weight", data.get("weight", 1.0))),
                    }
                    
                    # Handle ratings
                    if edge_type.upper() == "RATED" and "rating" in data:
                        # Preserve the exact rating value (-1, 0, 1)
                        edge_attrs["rating"] = float(data["rating"])
                    else:
                        # For non-rating edges, use 0.0
                        edge_attrs["rating"] = 0.0
                    
                    # Add timestamp if available or use default
                    edge_attrs["timestamp"] = float(data.get("timestamp", 0.0))
                    
                    # Add edge to graph
                    simplified_graph.add_edge(u, v, **edge_attrs)
                
        
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
                    "communities": {str(k): v for k, v in self.communities.items()}
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
        
        self.logger.info(f"Saved all graph files to {self.output_dir}")
    
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
                output_file = os.path.join(self.output_dir, f"text_graph_visualization.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            self.logger.info(f"Saved graph visualization to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            traceback.print_exc()


#main function
async def run_kg_builder(api_key: str,
                        movie_titles: List[str],
                        users_file: str,
                        ratings_file: str,
                        output_dir: str = "text_output",
                        movie_mapping_file: str = "movie_id_mapping.json"):
    """
    Run the knowledge graph builder with textual embeddings only
    
    Args:
        api_key: OpenAI API key
        movie_titles: List of movies to include
        users_file: Path to users CSV file
        ratings_file: Path to ratings JSON file
        output_dir: Directory to save output
        movie_mapping_file: Path to movie ID mapping file
    
    Returns:
        Graph, digraph, hypergraph, and recommendations
    """
    # Set up logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loggers = setup_logging(log_dir)
    
    logger = loggers["graph_builder"]
    logger.info(f"Starting graph builder with {len(movie_titles)} movies using textual embeddings only")
    
    # Initialize builder
    builder = BollywoodKGBuilder(
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
    
    graph, digraph = await builder.build_graph(
        movie_titles=movie_titles,
        process_users=True,
        generate_embeddings=True
    )
    
    build_time = time.time() - start_time
    
    logger.info(f"Graph building completed in {build_time:.2f} seconds")
    logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Save graph
    builder.save_graph(prefix="text_bollywood_kg")
    
    # Create visualization
    builder.visualize_graph(max_nodes=500, output_file=os.path.join(output_dir, "text_graph_viz.png"))
    
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
        rec_path = os.path.join(output_dir, "text_sample_recommendations.json")
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, indent=2)
    
    logger.info(f"Knowledge graph builder with textual embeddings completed. Results saved to {output_dir}")
    
    return graph, digraph, builder.hypergraph, recommendations

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
            "output_dir": "text_output",
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
    await run_kg_builder(
        api_key=api_key,
        movie_titles=movie_titles,
        users_file=config.get("users_file", "users.csv"),
        ratings_file=config.get("ratings_file", "ratings.json"),
        output_dir=config.get("output_dir", "text_output"),
        movie_mapping_file=config.get("movie_mapping_file", "movie_id_mapping.json")
    )

if __name__ == "__main__":
    asyncio.run(run_main())


#main function
async def run_kg_builder(api_key: str,
                        movie_titles: List[str],
                        users_file: str,
                        ratings_file: str,
                        output_dir: str = "text_output",
                        movie_mapping_file: str = "movie_id_mapping.json"):
    """
    Run the knowledge graph builder with textual embeddings only
    
    Args:
        api_key: OpenAI API key
        movie_titles: List of movies to include
        users_file: Path to users CSV file
        ratings_file: Path to ratings JSON file
        output_dir: Directory to save output
        movie_mapping_file: Path to movie ID mapping file
    
    Returns:
        Graph, digraph, hypergraph, and recommendations
    """
    # Set up logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loggers = setup_logging(log_dir)
    
    logger = loggers["graph_builder"]
    logger.info(f"Starting graph builder with {len(movie_titles)} movies using textual embeddings only")
    
    # Initialize builder
    builder = BollywoodKGBuilder(
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
    
    graph, digraph = await builder.build_graph(
        movie_titles=movie_titles,
        process_users=True,
        generate_embeddings=True
    )
    
    build_time = time.time() - start_time
    
    logger.info(f"Graph building completed in {build_time:.2f} seconds")
    logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Save graph
    builder.save_graph(prefix="text_bollywood_kg")
    
    # Create visualization
    builder.visualize_graph(max_nodes=500, output_file=os.path.join(output_dir, "text_graph_viz.png"))
    
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
        rec_path = os.path.join(output_dir, "text_sample_recommendations.json")
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, indent=2)
    
    logger.info(f"Knowledge graph builder with textual embeddings completed. Results saved to {output_dir}")
    
    return graph, digraph, builder.hypergraph, recommendations

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
            "output_dir": "text_output",
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
    await run_kg_builder(
        api_key=api_key,
        movie_titles=movie_titles,
        users_file="/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/users.csv",
        ratings_file="/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/ratings_array.json",
        output_dir="output_dir_text",
        movie_mapping_file="/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/data/Input/movie_id_mapping.json"
    )

if __name__ == "__main__":
    asyncio.run(run_main())