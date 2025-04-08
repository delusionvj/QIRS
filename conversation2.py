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
import requests
import traceback
from collections import defaultdict, Counter, deque
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import yaml

# For quantum graph components
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crs_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("conversational_agent")

# Configuration class
class Config:
    def __init__(self, config_path=None):
        self.config = {
            "openai_api_key": "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",
            "model": "gpt-4o-mini",
            "graph_path": "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/quantum_kg_output/quantum_unified_graph.pkl",
            "textual_embeddings_path": "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/quantum_kg_output/textual_embeddings.pkl",
            "neural_embeddings_path": "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/quantum_kg_output/quantum_combined_embeddings.pkl",
            "model_path": "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/quantum_kg_output/quantum_unified_model.pt",
            "community_profiles_path": "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/quantum_kg_output/quantum_community_profiles.json",
            "metapath_schemas_path": "/Users/amanvaibhavjha/Desktop/CRS/CRS_FINAL/quantum_kg_output/quantum_metapath_schemas.json",
            "max_context_length": 10,
            "confidence_threshold": 0.7,
            "follow_up_threshold": 0.5,
            "temperature": 0.7,
            "language": "hinglish",
            "data_dir": "data",
            "output_dir": "output_conv"
        }
        
        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        self.config.update(loaded_config)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Ensure API key is set
        if not self.config["openai_api_key"]:
            logger.warning("OpenAI API key not set. Please set it in the config file or environment variable.")
        
        # Create directories if they don't exist
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["output_dir"], exist_ok=True)
    
    def __getitem__(self, key):
        return self.config.get(key)
    
    def __setitem__(self, key, value):
        self.config[key] = value

# Class to handle OpenAI API requests
class OpenAIClient:
    def __init__(self, api_key, model="gpt-4o-mini", cache_dir=None):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.cache = {}
        self.logger = logging.getLogger("openai_client")
        
        # Create cache directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "openai_cache.json")
            # Load cache if exists
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'r') as f:
                        self.cache = json.load(f)
                except Exception as e:
                    self.logger.error(f"Error loading cache: {str(e)}")
    
    def _get_cache_key(self, messages, temperature):
        """Generate a cache key for a request"""
        # Convert messages to a string for hashing
        messages_str = json.dumps(messages, sort_keys=True)
        return f"{self.model}_{temperature}_{hash(messages_str)}"
    
    def _save_cache(self):
        """Save cache to file"""
        if self.cache_dir:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                self.logger.error(f"Error saving cache: {str(e)}")
    
    def chat_completion(self, messages, temperature=0.7, max_tokens=None, use_cache=True):
        """Make a chat completion API call to OpenAI with caching"""
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(messages, temperature)
            if cache_key in self.cache:
                self.logger.debug("Using cached response")
                return self.cache[cache_key]
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            self.logger.debug(f"Making API call to {self.model}")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                self.logger.error(f"API error ({response.status_code}): {response.text}")
                return {"error": response.text}
            
            result = response.json()
            response_content = result["choices"][0]["message"]["content"]
            response_data = {"content": response_content}
            print(response_data)
            # Cache result if enabled
            if use_cache:
                cache_key = self._get_cache_key(messages, temperature)
                self.cache[cache_key] = response_data
                self._save_cache()
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            return {"error": str(e)}

# Class to recognize entities from user messages
class EntityRecognizer:
    def __init__(self, openai_client, graph=None):
        self.openai_client = openai_client
        self.graph = graph  # Optional graph for entity validation
        self.logger = logging.getLogger("entity_recognizer")
    
    def _validate_entity(self, entity_type, entity_name):
        """Validate entity against the knowledge graph if available"""
        if not self.graph:
            return 1.0  # Default high confidence if no graph
        
        # Search for entity in graph
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "").lower()
            
            # Check if node type matches
            if node_type == entity_type.lower():
                # Check node name or title for match
                node_name = data.get("name", data.get("title", ""))
                
                # Exact match
                if entity_name.lower() == node_name.lower():
                    return 1.0
                
                # Partial match - calculate string similarity
                similarity = self._string_similarity(entity_name.lower(), node_name.lower())
                if similarity > 0.8:
                    return similarity
        
        return 0.5  # Medium confidence for entities not found in graph
    
    def _string_similarity(self, s1, s2):
        """Calculate simple string similarity based on overlap"""
        if not s1 or not s2:
            return 0.0
        
        # Tokenize
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def recognize_entities(self, text, conversation_history=None):
        """Recognize entities in text using GPT-4o-mini"""
        # Add conversation context if available
        context = ""
        if conversation_history and len(conversation_history) > 0:
            context = "Previous conversation:\n" + "\n".join([
                f"User: {turn['user']}" + (f"\nAssistant: {turn['system']}" if 'system' in turn else "")
                for turn in conversation_history[-2:]  # Last 2 turns for context
            ]) + "\n\n"
        
        prompt = [
            {"role": "system", "content": """You are an expert in recognizing Bollywood movie entities in Hindi/English/Hinglish conversations.
            
            Extract all entities related to:
            1. Movies (title)
            2. Actors/Actresses (name)
            3. Directors (name)
            4. Genres (name)
            5. Years/decades (year)
            6. User preferences (like, dislike, preferences)
            
            Return the results as a JSON object with this structure:
            {
                "movies": [{"id": "movie_title", "confidence": 0.9}],
                "actors": [{"id": "actor_name", "confidence": 0.9}],
                "directors": [{"id": "director_name", "confidence": 0.9}],
                "genres": [{"id": "genre_name", "confidence": 0.9}],
                "years": [{"id": "year", "confidence": 0.9}],
                "preferences": [{"type": "preference_type", "value": "preference_value", "sentiment": "positive/negative", "confidence": 0.9}]
            }
            
            If an entity type has no matches, return an empty array for that type.
            Include only high-confidence entities and be precise with entity names.
            For preferences, extract types like: genre_preference, actor_preference, era_preference, mood_preference, etc.
            """},
            {"role": "user", "content": context + "Extract entities from this text:\n\n" + text}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.3)
        
        if "error" in result:
            self.logger.error(f"Error recognizing entities: {result['error']}")
            return {}
        
        try:
            # Extract JSON from the response
            content = result["content"]
            
            # Find JSON block using regex
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code block markers
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
            
            # Clean up JSON string - replace single quotes with double quotes if needed
            json_str = json_str.replace("'", '"')
            
            entities = json.loads(json_str)
            
            # Validate entities if graph is available
            if self.graph:
                for entity_type in ["movies", "actors", "directors", "genres"]:
                    if entity_type in entities:
                        for entity in entities[entity_type]:
                            entity["confidence"] = max(
                                entity["confidence"],
                                self._validate_entity(entity_type[:-1], entity["id"])
                            )
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error parsing entity recognition result: {str(e)}")
            self.logger.error(f"Raw response: {result['content']}")
            return {}

# Class to recognize user intent
class IntentRecognizer:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.logger = logging.getLogger("intent_recognizer")
    
    def recognize_intent(self, text, conversation_history=None, entities=None):
        """Recognize user intent using GPT-4o-mini"""
        # Include conversation history for context if available
        context = ""
        if conversation_history and len(conversation_history) > 0:
            context = "Conversation history:\n" + "\n".join([
                f"User: {turn['user']}" + (f"\nAssistant: {turn['system']}" if 'system' in turn else "")
                for turn in conversation_history[-3:]  # Include last 3 turns
            ]) + "\n\n"
        
        # Add entity information if available
        entity_context = ""
        if entities:
            entity_context = "Recognized entities:\n"
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_names = [e["id"] for e in entity_list]
                    entity_context += f"- {entity_type}: {', '.join(entity_names)}\n"
        
        prompt = [
            {"role": "system", "content": """You are an expert in recognizing user intents in conversations about Bollywood movies.
            Determine the primary intent of the user message from the following categories:
            
            1. Greeting: User is saying hello or starting conversation
            2. Recommendation_Request: User is asking for movie recommendations
            3. Movie_Info: User is asking for information about a specific movie
            4. Actor_Info: User is asking for information about an actor
            5. Director_Info: User is asking for information about a director
            6. Genre_Info: User is asking for information about a genre
            7. Express_Preference: User is expressing preference (like/dislike)
            8. Comparison: User is asking for a comparison between movies/actors/etc.
            9. Clarification: User is asking for clarification about a previous message
            10. Follow_Up: User is responding to a follow-up question
            11. Farewell: User is ending the conversation
            12. Off_Topic: User message is not related to movies or recommendations
            
            Return the results as a JSON object with the following structure:
            {
                "primary_intent": "intent_name",
                "confidence": 0.9,
                "secondary_intent": "intent_name",
                "secondary_confidence": 0.5,
                "requires_followup": true/false,
                "followup_type": "preference/genre/actor/etc.",
                "ready_for_recommendation": true/false,
                "reasoning": "Brief explanation of your reasoning"
            }
            
            The "ready_for_recommendation" field should be true if:
            1. User is explicitly asking for recommendations, OR
            2. User has provided enough preference information to make good recommendations
            
            The "requires_followup" field should be true if:
            1. User intent is unclear, OR
            2. More information is needed to fulfill the user's request
            """},
            {"role": "user", "content": context + entity_context + "User message: " + text}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.3)
        
        if "error" in result:
            self.logger.error(f"Error recognizing intent: {result['error']}")
            return {
                "primary_intent": "Unknown",
                "confidence": 0.0,
                "requires_followup": True,
                "ready_for_recommendation": False,
                "reasoning": "Error in intent recognition"
            }
        
        try:
            # Extract JSON from the response
            content = result["content"]
            
            # Find JSON block using regex
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code block markers
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
            
            # Clean up JSON string - replace single quotes with double quotes if needed
            json_str = json_str.replace("'", '"')
            
            intent_data = json.loads(json_str)
            return intent_data
            
        except Exception as e:
            self.logger.error(f"Error parsing intent recognition result: {str(e)}")
            self.logger.error(f"Raw response: {result['content']}")
            return {
                "primary_intent": "Unknown",
                "confidence": 0.0,
                "requires_followup": True,
                "ready_for_recommendation": False,
                "reasoning": "Error parsing intent recognition result"
            }

# Class for reasoning with the knowledge graph
class GraphReasoner:
    def __init__(self, graph, meta_path_reasoner=None, community_detector=None, embeddings=None):
        self.graph = graph
        self.meta_path_reasoner = meta_path_reasoner
        self.community_detector = community_detector
        self.embeddings = embeddings
        self.logger = logging.getLogger("graph_reasoner")
        
        # Create empty placeholders if components are not provided
        if meta_path_reasoner is None:
            self.meta_path_reasoner = self._create_mock_meta_path_reasoner()
        
        if community_detector is None:
            self.community_detector = self._create_mock_community_detector()
    
    def _create_mock_meta_path_reasoner(self):
        """Create a simple version of meta-path reasoner if none is provided"""
        class SimplifiedMetaPathReasoner:
            def __init__(self, graph):
                self.graph = graph
                self.logger = logging.getLogger("simplified_meta_path")
            
            def find_paths(self, start_node, end_node, max_length=3, max_paths=5):
                """Find paths between nodes using simple BFS"""
                if start_node not in self.graph or end_node not in self.graph:
                    return []
                
                # Use NetworkX to find simple paths
                try:
                    all_paths = list(nx.all_simple_paths(self.graph, start_node, end_node, cutoff=max_length))
                    
                    # Sort by path length (shorter paths first)
                    all_paths.sort(key=len)
                    
                    # Convert to path data format
                    path_data = []
                    for path in all_paths[:max_paths]:
                        # Extract node types for meta-path
                        meta_path = []
                        for node in path:
                            if node in self.graph.nodes():
                                node_type = self.graph.nodes[node].get("node_type", "unknown")
                                meta_path.append(node_type)
                        
                        # Create path name from node types
                        path_name = "->".join(meta_path)
                        
                        path_data.append({
                            "path": path,
                            "meta_path": meta_path,
                            "path_name": path_name,
                            "score": 1.0 / len(path)  # Simple score based on path length
                        })
                    
                    return path_data
                    
                except Exception as e:
                    logger.error(f"Error finding paths: {str(e)}")
                    return []
            
            def generate_explanation(self, path_data, personalize=True):
                """Generate simple explanation for a path"""
                if not path_data or "path" not in path_data:
                    return "No explanation available."
                
                path = path_data["path"]
                if len(path) < 2:
                    return "Direct connection."
                
                # Create explanation based on node types and names
                explanation = []
                for i in range(len(path) - 1):
                    if path[i] in self.graph.nodes() and path[i+1] in self.graph.nodes():
                        u_type = self.graph.nodes[path[i]].get("node_type", "item")
                        v_type = self.graph.nodes[path[i+1]].get("node_type", "item")
                        
                        u_name = self.graph.nodes[path[i]].get("name", 
                                                          self.graph.nodes[path[i]].get("title", path[i]))
                        v_name = self.graph.nodes[path[i+1]].get("name", 
                                                          self.graph.nodes[path[i+1]].get("title", path[i+1]))
                        
                        if i == 0:
                            explanation.append(f"{u_name} ({u_type}) is connected to {v_name} ({v_type})")
                        else:
                            explanation.append(f"which is connected to {v_name} ({v_type})")
                
                return " ".join(explanation)
        
        return SimplifiedMetaPathReasoner(self.graph)
    
    def _create_mock_community_detector(self):
        """Create a simple version of community detector if none is provided"""
        class SimplifiedCommunityDetector:
            def __init__(self, graph):
                self.graph = graph
                self.communities = {}
                self.community_to_nodes = defaultdict(list)
                self.community_profiles = {}
                self.logger = logging.getLogger("simplified_community")
                
                # Run a simple community detection
                self._detect_communities()
            
            def _detect_communities(self):
                """Simple community detection based on connected components"""
                try:
                    # Try to use NetworkX's community detection
                    from networkx.algorithms import community
                    
                    # Use Louvain method if available
                    communities = community.louvain_communities(self.graph)
                    
                    for i, comm in enumerate(communities):
                        for node in comm:
                            self.communities[node] = i
                            self.community_to_nodes[i].append(node)
                            
                            # Add community ID to node attributes
                            if node in self.graph.nodes():
                                self.graph.nodes[node]["community_id"] = i
                    
                    # Generate simple profiles
                    for comm_id, nodes in self.community_to_nodes.items():
                        # Count node types
                        type_counts = Counter()
                        for node in nodes:
                            if node in self.graph.nodes():
                                node_type = self.graph.nodes[node].get("node_type", "unknown")
                                type_counts[node_type] += 1
                        
                        # Determine main type
                        main_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"
                        
                        self.community_profiles[comm_id] = {
                            "id": comm_id,
                            "size": len(nodes),
                            "main_type": main_type,
                            "node_type_distribution": dict(type_counts),
                            "attributes": {}
                        }
                
                except (ImportError, Exception) as e:
                    logger.warning(f"Error in community detection: {str(e)}")
                    
                    # Fallback to connected components
                    for i, component in enumerate(nx.connected_components(self.graph)):
                        for node in component:
                            self.communities[node] = i
                            self.community_to_nodes[i].append(node)
                    
                    # Generate simple profiles
                    for comm_id, nodes in self.community_to_nodes.items():
                        # Count node types
                        type_counts = Counter()
                        for node in nodes:
                            if node in self.graph.nodes():
                                node_type = self.graph.nodes[node].get("node_type", "unknown")
                                type_counts[node_type] += 1
                        
                        # Determine main type
                        main_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"
                        
                        self.community_profiles[comm_id] = {
                            "id": comm_id,
                            "size": len(nodes),
                            "main_type": main_type,
                            "node_type_distribution": dict(type_counts),
                            "attributes": {}
                        }
            
            def get_community_for_node(self, node_id):
                """Get community ID for a node"""
                return self.communities.get(node_id)
            
            def get_community_profile(self, comm_id):
                """Get profile for a community"""
                return self.community_profiles.get(comm_id, {})
            
            def get_community_description(self, comm_id):
                """Generate a description for a community"""
                if comm_id not in self.community_profiles:
                    return "Unknown community"
                
                profile = self.community_profiles[comm_id]
                
                # Basic description based on size and main type
                description = f"A community of {profile['size']} nodes, primarily {profile['main_type']}s"
                
                # Add node type distribution
                if "node_type_distribution" in profile:
                    type_str = ", ".join([f"{count} {type_name}s" 
                                        for type_name, count in profile["node_type_distribution"].items()
                                        if count > 0 and type_name != profile["main_type"]])
                    
                    if type_str:
                        description += f", along with {type_str}"
                
                return description + "."
        
        return SimplifiedCommunityDetector(self.graph)
    
    def find_relevant_nodes(self, entities, max_nodes=5):
        """Find relevant nodes in the graph based on recognized entities"""
        relevant_nodes = []
        
        # Process movies
        for movie in entities.get("movies", []):
            movie_id = movie["id"]
            candidates = []
            
            for node, node_data in self.graph.nodes(data=True):
                if node_data.get("node_type") == "movie":
                    title = node_data.get("title", "")
                    
                    # Exact match
                    if movie_id.lower() == title.lower():
                        candidates.append((node, 1.0))
                    # Partial match
                    elif movie_id.lower() in title.lower() or title.lower() in movie_id.lower():
                        similarity = self._string_similarity(movie_id, title)
                        if similarity > 0.5:  # Threshold for similarity
                            candidates.append((node, similarity))
            
            # Sort candidates by similarity and take the best match
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_node, similarity = candidates[0]
                
                relevant_nodes.append({
                    "id": best_node,
                    "type": "movie",
                    "name": self.graph.nodes[best_node].get("title", best_node),
                    "confidence": similarity * movie["confidence"]
                })
        
        # Process actors
        for actor in entities.get("actors", []):
            actor_id = actor["id"]
            candidates = []
            
            for node, node_data in self.graph.nodes(data=True):
                if node_data.get("node_type") == "actor":
                    name = node_data.get("name", "")
                    
                    # Exact match
                    if actor_id.lower() == name.lower():
                        candidates.append((node, 1.0))
                    # Partial match
                    elif actor_id.lower() in name.lower() or name.lower() in actor_id.lower():
                        similarity = self._string_similarity(actor_id, name)
                        if similarity > 0.5:  # Threshold for similarity
                            candidates.append((node, similarity))
            
            # Sort candidates by similarity and take the best match
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_node, similarity = candidates[0]
                
                relevant_nodes.append({
                    "id": best_node,
                    "type": "actor",
                    "name": self.graph.nodes[best_node].get("name", best_node),
                    "confidence": similarity * actor["confidence"]
                })
        
        # Process directors
        for director in entities.get("directors", []):
            director_id = director["id"]
            candidates = []
            
            for node, node_data in self.graph.nodes(data=True):
                if node_data.get("node_type") == "director":
                    name = node_data.get("name", "")
                    
                    # Exact match
                    if director_id.lower() == name.lower():
                        candidates.append((node, 1.0))
                    # Partial match
                    elif director_id.lower() in name.lower() or name.lower() in director_id.lower():
                        similarity = self._string_similarity(director_id, name)
                        if similarity > 0.5:  # Threshold for similarity
                            candidates.append((node, similarity))
            
            # Sort candidates by similarity and take the best match
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_node, similarity = candidates[0]
                
                relevant_nodes.append({
                    "id": best_node,
                    "type": "director",
                    "name": self.graph.nodes[best_node].get("name", best_node),
                    "confidence": similarity * director["confidence"]
                })
        
        # Process genres
        for genre in entities.get("genres", []):
            genre_id = genre["id"]
            candidates = []
            
            for node, node_data in self.graph.nodes(data=True):
                if node_data.get("node_type") == "genre":
                    name = node_data.get("name", "")
                    
                    # Exact match
                    if genre_id.lower() == name.lower():
                        candidates.append((node, 1.0))
                    # Partial match
                    elif genre_id.lower() in name.lower() or name.lower() in genre_id.lower():
                        similarity = self._string_similarity(genre_id, name)
                        if similarity > 0.5:  # Threshold for similarity
                            candidates.append((node, similarity))
            
            # Sort candidates by similarity and take the best match
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_node, similarity = candidates[0]
                
                relevant_nodes.append({
                    "id": best_node,
                    "type": "genre",
                    "name": self.graph.nodes[best_node].get("name", best_node),
                    "confidence": similarity * genre["confidence"]
                })
        
        # Sort all nodes by confidence and limit to max_nodes
        relevant_nodes.sort(key=lambda x: x["confidence"], reverse=True)
        return relevant_nodes[:max_nodes]
    
    def _string_similarity(self, s1, s2):
        """Calculate simple string similarity based on overlap"""
        if not s1 or not s2:
            return 0.0
        
        # Tokenize
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def extract_path_context(self, node_ids, context_depth=2, max_paths=3):
        """Extract context from paths between nodes"""
        path_contexts = []
        
        # Find paths between all pairs of nodes
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                source = node_ids[i]
                target = node_ids[j]
                
                # Find paths using meta-path reasoner
                paths = self.meta_path_reasoner.find_paths(
                    source, 
                    target, 
                    max_length=context_depth
                )
                
                # Add path explanations
                for path in paths[:max_paths]:
                    explanation = self.meta_path_reasoner.generate_explanation(
                        path,
                        personalize=False
                    )
                    
                    path_contexts.append({
                        "source": source,
                        "target": target,
                        "path": path["path"],
                        "explanation": explanation,
                        "meta_path": path["path_name"] if "path_name" in path else "",
                        "score": path["score"] if "score" in path else 0.5
                    })
        
        # Sort by score
        path_contexts.sort(key=lambda x: x["score"], reverse=True)
        return path_contexts
    
    def extract_community_context(self, node_ids, max_communities=3):
        """Extract context from communities of nodes"""
        community_contexts = []
        
        for node_id in node_ids:
            # Get community of the node
            community_id = self.community_detector.get_community_for_node(node_id)
            
            if community_id is not None:
                # Get community profile
                profile = self.community_detector.get_community_profile(community_id)
                
                # Get community description
                description = self.community_detector.get_community_description(community_id)
                
                community_contexts.append({
                    "node_id": node_id,
                    "community_id": community_id,
                    "description": description,
                    "size": profile.get("size", 0),
                    "main_type": profile.get("main_type", "unknown"),
                    "attributes": profile.get("attributes", {})
                })
        
        # Sort by size (larger communities first)
        community_contexts.sort(key=lambda x: x["size"], reverse=True)
        return community_contexts[:max_communities]
    
    def extract_neighborhood_context(self, node_ids, max_neighbors=5):
        """Extract context from node neighborhoods"""
        neighborhood_contexts = []
        
        for node_id in node_ids:
            if node_id in self.graph:
                # Get immediate neighbors
                neighbors = list(self.graph.neighbors(node_id))
                
                # Collect information about neighbors
                neighbor_info = []
                for neighbor in neighbors[:max_neighbors]:
                    if neighbor in self.graph.nodes():
                        node_data = self.graph.nodes[neighbor]
                        node_type = node_data.get("node_type", "unknown")
                        node_name = node_data.get("name", node_data.get("title", neighbor))
                        
                        # Get edge information
                        edge_data = self.graph.get_edge_data(node_id, neighbor)
                        edge_type = edge_data.get("edge_type", "connected_to") if edge_data else "connected_to"
                        
                        neighbor_info.append({
                            "id": neighbor,
                            "name": node_name,
                            "type": node_type,
                            "edge_type": edge_type
                        })
                
                # Group neighbors by type for better context
                grouped_neighbors = defaultdict(list)
                for info in neighbor_info:
                    grouped_neighbors[info["type"]].append(info)
                
                neighborhood_contexts.append({
                    "node_id": node_id,
                    "node_name": self.graph.nodes[node_id].get("name", 
                                                          self.graph.nodes[node_id].get("title", node_id)),
                    "node_type": self.graph.nodes[node_id].get("node_type", "unknown"),
                    "neighbor_groups": dict(grouped_neighbors),
                    "total_neighbors": len(neighbors)
                })
        
        return neighborhood_contexts
    
    def generate_reasoning_context(self, entities, user_node=None):
        """Generate reasoning context based on entities and user node"""
        # Find relevant nodes
        relevant_nodes = self.find_relevant_nodes(entities)
        
        # Extract node IDs
        node_ids = [node["id"] for node in relevant_nodes]
        
        # Add user node if available
        if user_node and user_node in self.graph:
            if user_node not in node_ids:
                node_ids.append(user_node)
                relevant_nodes.append({
                    "id": user_node,
                    "type": "user",
                    "name": self.graph.nodes[user_node].get("name", user_node),
                    "confidence": 1.0
                })
        
        # Get path context
        path_contexts = self.extract_path_context(node_ids)
        
        # Get community context
        community_contexts = self.extract_community_context(node_ids)
        
        # Get neighborhood context
        neighborhood_contexts = self.extract_neighborhood_context(node_ids)
        
        # Combine contexts
        reasoning_context = {
            "relevant_nodes": relevant_nodes,
            "path_contexts": path_contexts,
            "community_contexts": community_contexts,
            "neighborhood_contexts": neighborhood_contexts
        }
        
        return reasoning_context

# Class for generating follow-up questions
class FollowupQuestionGenerator:
    def __init__(self, openai_client, graph_reasoner):
        self.openai_client = openai_client
        self.graph_reasoner = graph_reasoner
        self.logger = logging.getLogger("followup_generator")
    
    def generate_followup_question(self, user_message, entities, reasoning_context, 
                                  conversation_history, intent=None, language="english"):
        """Generate follow-up question based on reasoning context"""
        # Determine follow-up type based on intent and context
        followup_type = self._determine_followup_type(intent, entities, reasoning_context)
        
        # Create prompt for GPT-4o-mini
        system_prompt = f"""You are a knowledgeable Bollywood movie recommendation assistant that asks thoughtful follow-up questions.
        Based on the conversation history, user message, and graph context provided, generate a natural and helpful follow-up question.
        
        You should focus on asking about {followup_type} to better understand the user's preferences.
        
        Your follow-up question should:
        1. Be specific and directly related to the conversation context
        2. Help clarify user preferences or gather more information
        3. Sound natural and conversational
        4. Not repeat information the user has already provided
        5. Focus on the most relevant aspects based on graph context
        
        The graph context contains information from a knowledge graph about Bollywood movies, actors, directors, genres, etc.
        Use this information to ask informed questions.
        """
        
        if language.lower() == "hinglish":
            system_prompt += "\n\nRespond in Hinglish (mix of Hindi and English) that sounds natural and conversational. Use Hindi for emotional expressions and common phrases, and English for technical terms."
        
        # Format the conversation history for the prompt
        conversation_context = ""
        if conversation_history:
            conversation_context = "Conversation history:\n"
            for i, turn in enumerate(conversation_history[-3:]):  # Include last 3 turns
                conversation_context += f"User: {turn['user']}\n"
                if 'system' in turn:
                    conversation_context += f"Assistant: {turn['system']}\n"
        
        # Format reasoning context for the prompt
        graph_context = "Graph Context:\n"
        
        # Add information about relevant nodes
        if "relevant_nodes" in reasoning_context and reasoning_context["relevant_nodes"]:
            graph_context += "Relevant nodes:\n"
            for node in reasoning_context["relevant_nodes"]:
                graph_context += f"- {node['name']} ({node['type']})\n"
        
        # Add path contexts
        if "path_contexts" in reasoning_context and reasoning_context["path_contexts"]:
            graph_context += "\nRelevant relationships:\n"
            for i, path in enumerate(reasoning_context["path_contexts"][:3]):  # Limit to 3 paths
                graph_context += f"- {path['explanation']}\n"
        
        # Add community contexts
        if "community_contexts" in reasoning_context and reasoning_context["community_contexts"]:
            graph_context += "\nCommunity information:\n"
            for community in reasoning_context["community_contexts"]:
                graph_context += f"- {community['description']}\n"
        
        # Add neighborhood contexts for specific entities
        if "neighborhood_contexts" in reasoning_context and reasoning_context["neighborhood_contexts"]:
            graph_context += "\nEntity details:\n"
            for context in reasoning_context["neighborhood_contexts"]:
                graph_context += f"- {context['node_name']} ({context['node_type']}) connects to: "
                
                # Format neighbor groups
                neighbor_info = []
                for type_name, neighbors in context.get("neighbor_groups", {}).items():
                    if neighbors:
                        names = [n["name"] for n in neighbors[:3]]  # Limit to 3 per type
                        if names:
                            neighbor_info.append(f"{len(neighbors)} {type_name}s including {', '.join(names)}")
                
                graph_context += ", ".join(neighbor_info) + "\n"
        
        # Create the prompt
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (f"User message: {user_message}\n\n" +
                                       conversation_context + "\n" +
                                       graph_context + "\n\n" +
                                       "Generate a single follow-up question to better understand the user's preferences or clarify their request.")}
        ]
        
        # Get response from OpenAI
        result = self.openai_client.chat_completion(prompt, temperature=0.7)
        
        if "error" in result:
            self.logger.error(f"Error generating follow-up question: {result['error']}")
            # Return a generic follow-up question
            if language.lower() == "hinglish":
                return "Aap Bollywood movies ke baare mein aur kya jaanna chahte hain?"
            else:
                return "What else would you like to know about Bollywood movies?"
        
        # Extract the follow-up question
        followup_question = result["content"].strip()
        
        # Remove any prefixes that might be in the response
        prefixes = ["Here's a follow-up question:", "Follow-up question:", "Assistant:", "Followup:"]
        for prefix in prefixes:
            if followup_question.startswith(prefix):
                followup_question = followup_question[len(prefix):].strip()
        
        return followup_question
    
    def _determine_followup_type(self, intent, entities, reasoning_context):
        """Determine what type of follow-up question to ask based on context"""
        # Default follow-up type
        followup_type = "general preferences"
        
        # Use intent to guide follow-up if available
        if intent:
            followup_type_from_intent = intent.get("followup_type")
            if followup_type_from_intent:
                return followup_type_from_intent
            
            primary_intent = intent.get("primary_intent", "").lower()
            
            if "recommendation" in primary_intent:
                # For recommendation requests, ask about genre or mood preferences
                if "genres" not in entities or not entities["genres"]:
                    return "genre preferences"
                else:
                    return "mood preferences"
            
            elif "movie_info" in primary_intent:
                # For movie info requests, ask about specific aspects
                if "movies" in entities and entities["movies"]:
                    return "specific movie aspects (plot, cast, reception)"
                else:
                    return "which specific movie"
            
            elif "actor_info" in primary_intent:
                # For actor info requests, ask about specific aspects
                if "actors" in entities and entities["actors"]:
                    return "specific actor aspects (famous roles, career)"
                else:
                    return "which specific actor"
            
            elif "express_preference" in primary_intent:
                # For preference expressions, dig deeper
                if "genres" in entities and entities["genres"]:
                    return "specific movies within this genre"
                elif "actors" in entities and entities["actors"]:
                    return "what they like about this actor"
                else:
                    return "what types of movies they generally enjoy"
        
        # If no intent or intent doesn't suggest a follow-up type,
        # use reasoning context to determine what's missing
        missing_info = self._identify_missing_information(entities, reasoning_context)
        
        if missing_info:
            return missing_info
        
        return followup_type
    
    def _identify_missing_information(self, entities, reasoning_context):
        """Identify what information is missing based on entities and reasoning context"""
        # Check if we have genre information
        if "genres" not in entities or not entities["genres"]:
            return "genre preferences"
        
        # Check if we have actor information
        if "actors" not in entities or not entities["actors"]:
            return "favorite actors"
        
        # Check if we have era/time information
        if "years" not in entities or not entities["years"]:
            return "preferred movie time periods"
        
        # Check if we have movie examples
        if "movies" not in entities or not entities["movies"]:
            return "examples of movies they like"
        
        # If we have all basic information, ask about more specific preferences
        if "preferences" not in entities or not entities["preferences"]:
            return "specific elements they enjoy in movies (music, cinematography, storyline)"
        
        # If we have everything, ask about mood or context
        return "current mood or viewing context"

# Class for generating recommendations
class RecommendationGenerator:
    def __init__(self, openai_client, graph_reasoner, graph=None, embeddings=None, quantum_model=None):
        self.openai_client = openai_client
        self.graph_reasoner = graph_reasoner
        self.graph = graph
        self.embeddings = embeddings
        self.quantum_model = quantum_model
        self.logger = logging.getLogger("recommendation_generator")
    
    def generate_recommendations(self, user_message, entities, reasoning_context, 
                               conversation_history, user_node=None, num_recommendations=3):
        """Generate movie recommendations based on user preferences and graph context"""
        # Collect preference information from entities and conversation history
        preferences = self._extract_preferences(entities, conversation_history)
        
        # Get candidate movies based on preferences
        candidates = self._get_candidate_movies(preferences, user_node)
        
        # Score candidates
        scored_candidates = self._score_candidates(candidates, preferences, reasoning_context, user_node)
        
        # Take top candidates
        top_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)[:num_recommendations]
        
        # Generate explanations for recommendations
        recommendations = self._generate_explanations(top_candidates, preferences, user_node)
        
        return recommendations
    
    def _extract_preferences(self, entities, conversation_history):
        """Extract preference information from entities and conversation history"""
        preferences = {
            "genres": [],
            "actors": [],
            "directors": [],
            "years": [],
            "positive_examples": [],
            "negative_examples": [],
            "other_preferences": {}
        }
        
        # Extract from current entities
        if "genres" in entities:
            preferences["genres"].extend([genre["id"] for genre in entities["genres"]])
        
        if "actors" in entities:
            preferences["actors"].extend([actor["id"] for actor in entities["actors"]])
        
        if "directors" in entities:
            preferences["directors"].extend([director["id"] for director in entities["directors"]])
        
        if "years" in entities:
            preferences["years"].extend([year["id"] for year in entities["years"]])
        
        if "movies" in entities:
            for movie in entities["movies"]:
                # Try to determine if it's a positive or negative example
                is_positive = True
                if "preferences" in entities:
                    for pref in entities["preferences"]:
                        if (pref.get("type") == "movie_preference" and 
                            pref.get("value") == movie["id"] and 
                            pref.get("sentiment") == "negative"):
                            is_positive = False
                            break
                
                if is_positive:
                    preferences["positive_examples"].append(movie["id"])
                else:
                    preferences["negative_examples"].append(movie["id"])
        
        if "preferences" in entities:
            for pref in entities["preferences"]:
                if pref.get("type") not in ["movie_preference", "genre_preference", 
                                         "actor_preference", "director_preference"]:
                    pref_type = pref.get("type", "other_preference")
                    pref_value = pref.get("value", "")
                    pref_sentiment = pref.get("sentiment", "positive")
                    
                    if pref_type not in preferences["other_preferences"]:
                        preferences["other_preferences"][pref_type] = []
                    
                    preferences["other_preferences"][pref_type].append({
                        "value": pref_value,
                        "sentiment": pref_sentiment
                    })
        
        # Extract from conversation history
        if conversation_history:
            for turn in conversation_history:
                if "entities" in turn:
                    turn_entities = turn["entities"]
                    
                    # Extract genres
                    if "genres" in turn_entities:
                        for genre in turn_entities["genres"]:
                            if genre["id"] not in preferences["genres"]:
                                preferences["genres"].append(genre["id"])
                    
                    # Extract actors
                    if "actors" in turn_entities:
                        for actor in turn_entities["actors"]:
                            if actor["id"] not in preferences["actors"]:
                                preferences["actors"].append(actor["id"])
                    
                    # Extract directors
                    if "directors" in turn_entities:
                        for director in turn_entities["directors"]:
                            if director["id"] not in preferences["directors"]:
                                preferences["directors"].append(director["id"])
                    
                    # Extract years
                    if "years" in turn_entities:
                        for year in turn_entities["years"]:
                            if year["id"] not in preferences["years"]:
                                preferences["years"].append(year["id"])
                    
                    # Extract movie examples
                    if "movies" in turn_entities:
                        for movie in turn_entities["movies"]:
                            # Check if it's a positive or negative example
                            is_positive = True
                            if "preferences" in turn_entities:
                                for pref in turn_entities["preferences"]:
                                    if (pref.get("type") == "movie_preference" and 
                                        pref.get("value") == movie["id"] and 
                                        pref.get("sentiment") == "negative"):
                                        is_positive = False
                                        break
                            
                            if is_positive and movie["id"] not in preferences["positive_examples"]:
                                preferences["positive_examples"].append(movie["id"])
                            elif not is_positive and movie["id"] not in preferences["negative_examples"]:
                                preferences["negative_examples"].append(movie["id"])
        
        # Remove duplicates
        preferences["genres"] = list(set(preferences["genres"]))
        preferences["actors"] = list(set(preferences["actors"]))
        preferences["directors"] = list(set(preferences["directors"]))
        preferences["years"] = list(set(preferences["years"]))
        preferences["positive_examples"] = list(set(preferences["positive_examples"]))
        preferences["negative_examples"] = list(set(preferences["negative_examples"]))
        
        return preferences
    
    def _get_candidate_movies(self, preferences, user_node=None, max_candidates=50):
        """Get candidate movies based on preferences"""
        candidates = []
        
        # Helper function to find nodes by attribute
        def find_nodes_by_attribute(attribute, value, node_type=None):
            matching_nodes = []
            for node, data in self.graph.nodes(data=True):
                if node_type and data.get("node_type") != node_type:
                    continue
                
                attr_value = data.get(attribute)
                if attr_value:
                    # Handle JSON strings
                    if isinstance(attr_value, str) and attr_value.startswith("["):
                        try:
                            attr_list = json.loads(attr_value)
                            if value in attr_list:
                                matching_nodes.append(node)
                        except:
                            # If not valid JSON, try string comparison
                            if value.lower() in attr_value.lower():
                                matching_nodes.append(node)
                    # Direct comparison
                    elif value.lower() in str(attr_value).lower():
                        matching_nodes.append(node)
            
            return matching_nodes
        
        # Start with previously rated movies from user node
        if user_node and user_node in self.graph:
            # Find movies connected to user
            for neighbor in self.graph.neighbors(user_node):
                if neighbor in self.graph.nodes() and self.graph.nodes[neighbor].get("node_type") == "movie":
                    # Check if positively rated
                    edge_data = self.graph.get_edge_data(user_node, neighbor)
                    if edge_data and "rating" in edge_data:
                        rating = edge_data["rating"]
                        
                        if float(rating) > 0:
                            # Add to positive examples if not already there
                            movie_title = self.graph.nodes[neighbor].get("title", neighbor)
                            if movie_title not in preferences["positive_examples"]:
                                preferences["positive_examples"].append(movie_title)
        
        # Gather candidates from positive examples (similar movies)
        for movie_title in preferences["positive_examples"]:
            # Find movie node
            movie_nodes = find_nodes_by_attribute("title", movie_title, "movie")
            
            if movie_nodes:
                # Add similar movies
                for movie_node in movie_nodes:
                    # Add the movie itself
                    if movie_node not in candidates:
                        candidates.append(movie_node)
                    
                    # Use meta-path reasoning to find similar movies
                    for node, data in self.graph.nodes(data=True):
                        if node != movie_node and data.get("node_type") == "movie":
                            # Find paths between these movies
                            paths = self.graph_reasoner.meta_path_reasoner.find_paths(
                                movie_node, node, max_length=3, max_paths=2
                            )
                            
                            if paths:
                                # Calculate similarity based on path score
                                best_path = paths[0]
                                similarity = best_path["score"] if "score" in best_path else 0.5
                                
                                if similarity > 0.3:  # Threshold for similarity
                                    if node not in candidates:
                                        candidates.append(node)
        
        # If we don't have enough candidates yet, add more based on preferences
        if len(candidates) < max_candidates:
            # Add movies from preferred genres
            for genre in preferences["genres"]:
                # Find genre nodes
                genre_nodes = find_nodes_by_attribute("name", genre, "genre")
                
                for genre_node in genre_nodes:
                    # Find movies connected to this genre
                    for node in self.graph.neighbors(genre_node):
                        if node in self.graph.nodes() and self.graph.nodes[node].get("node_type") == "movie":
                            if node not in candidates:
                                candidates.append(node)
            
            # Add movies from preferred actors
            for actor in preferences["actors"]:
                # Find actor nodes
                actor_nodes = find_nodes_by_attribute("name", actor, "actor")
                
                for actor_node in actor_nodes:
                    # Find movies connected to this actor
                    for node in self.graph.neighbors(actor_node):
                        if node in self.graph.nodes() and self.graph.nodes[node].get("node_type") == "movie":
                            if node not in candidates:
                                candidates.append(node)
            
            # Add movies from preferred directors
            for director in preferences["directors"]:
                # Find director nodes
                director_nodes = find_nodes_by_attribute("name", director, "director")
                
                for director_node in director_nodes:
                    # Find movies connected to this director
                    for node in self.graph.neighbors(director_node):
                        if node in self.graph.nodes() and self.graph.nodes[node].get("node_type") == "movie":
                            if node not in candidates:
                                candidates.append(node)
            
            # Add movies from preferred years
            for year in preferences["years"]:
                # Find movies from this year
                year_nodes = find_nodes_by_attribute("year", year, "movie")
                
                for node in year_nodes:
                    if node not in candidates:
                        candidates.append(node)
        
        # If still not enough candidates, add popular movies
        if len(candidates) < max_candidates:
            # Sort all movie nodes by degree (as a proxy for popularity)
            movie_nodes = [(node, self.graph.degree(node)) 
                         for node, data in self.graph.nodes(data=True) 
                         if data.get("node_type") == "movie"]
            
            movie_nodes.sort(key=lambda x: x[1], reverse=True)
            
            for node, _ in movie_nodes:
                if node not in candidates:
                    candidates.append(node)
                    
                    if len(candidates) >= max_candidates:
                        break
        
        # Remove negative examples
        final_candidates = []
        for movie_node in candidates:
            if movie_node in self.graph.nodes():
                movie_title = self.graph.nodes[movie_node].get("title", movie_node)
                
                if movie_title not in preferences["negative_examples"]:
                    final_candidates.append(movie_node)
        
        return final_candidates[:max_candidates]
    
    def _score_candidates(self, candidates, preferences, reasoning_context, user_node=None):
        """Score candidate movies based on preferences and context"""
        scored_candidates = []
        
        for movie_node in candidates:
            if movie_node not in self.graph.nodes():
                continue
                
            movie_data = self.graph.nodes[movie_node]
            
            # Initialize score components
            genre_score = 0.0
            actor_score = 0.0
            director_score = 0.0
            year_score = 0.0
            similarity_score = 0.0
            community_score = 0.0
            path_score = 0.0
            
            # Check genre match
            if preferences["genres"]:
                genres = movie_data.get("genres", "[]")
                
                if isinstance(genres, str):
                    try:
                        genre_list = json.loads(genres)
                    except:
                        genre_list = [g.strip() for g in genres.split(",")]
                else:
                    genre_list = []
                
                # Count matches
                matches = sum(1 for genre in preferences["genres"] 
                           if any(genre.lower() in g.lower() for g in genre_list))
                
                if matches > 0:
                    genre_score = min(1.0, matches / len(preferences["genres"]))
            
            # Check actor match
            if preferences["actors"]:
                # Find actors connected to this movie
                movie_actors = []
                
                for neighbor in self.graph.neighbors(movie_node):
                    if neighbor in self.graph.nodes() and self.graph.nodes[neighbor].get("node_type") == "actor":
                        actor_name = self.graph.nodes[neighbor].get("name", "")
                        movie_actors.append(actor_name)
                
                # Count matches
                matches = sum(1 for actor in preferences["actors"] 
                           if any(actor.lower() in a.lower() for a in movie_actors))
                
                if matches > 0:
                    actor_score = min(1.0, matches / len(preferences["actors"]))
            
            # Check director match
            if preferences["directors"]:
                # Find directors connected to this movie
                movie_directors = []
                
                for neighbor in self.graph.neighbors(movie_node):
                    if neighbor in self.graph.nodes() and self.graph.nodes[neighbor].get("node_type") == "director":
                        director_name = self.graph.nodes[neighbor].get("name", "")
                        movie_directors.append(director_name)
                
                # Count matches
                matches = sum(1 for director in preferences["directors"] 
                           if any(director.lower() in d.lower() for d in movie_directors))
                
                if matches > 0:
                    director_score = min(1.0, matches / len(preferences["directors"]))
            
            # Check year match
            if preferences["years"]:
                movie_year = movie_data.get("year", "")
                
                if movie_year:
                    # Count matches
                    matches = sum(1 for year in preferences["years"] 
                               if str(year) == str(movie_year))
                    
                    if matches > 0:
                        year_score = 1.0
                    else:
                        # Check if within a decade
                        try:
                            movie_year_int = int(movie_year)
                            
                            for year in preferences["years"]:
                                year_int = int(year)
                                if abs(movie_year_int - year_int) <= 10:
                                    year_score = 1.0 - (abs(movie_year_int - year_int) / 10)
                                    break
                        except:
                            pass
            
            # Check embedding similarity with positive examples
            if self.embeddings and preferences["positive_examples"]:
                # Find embeddings for positive examples
                example_embeddings = []
                
                for example in preferences["positive_examples"]:
                    # Find example node
                    for node, data in self.graph.nodes(data=True):
                        if data.get("node_type") == "movie" and data.get("title", "") == example:
                            if node in self.embeddings:
                                example_embeddings.append(self.embeddings[node])
                                break
                
                # Calculate similarity if we have embeddings
                if movie_node in self.embeddings and example_embeddings:
                    movie_embedding = self.embeddings[movie_node]
                    
                    # Calculate average similarity
                    similarities = []
                    for example_emb in example_embeddings:
                        similarity = cosine_similarity(
                            movie_embedding.reshape(1, -1),
                            example_emb.reshape(1, -1)
                        )[0][0]
                        
                        similarities.append(max(0, similarity))
                    
                    if similarities:
                        similarity_score = sum(similarities) / len(similarities)
            
            # Check community match
            if self.graph_reasoner.community_detector:
                movie_community = self.graph_reasoner.community_detector.get_community_for_node(movie_node)
                
                if movie_community is not None:
                    # Check if any positive example is in the same community
                    for example in preferences["positive_examples"]:
                        # Find example node
                        for node, data in self.graph.nodes(data=True):
                            if data.get("node_type") == "movie" and data.get("title", "") == example:
                                example_community = self.graph_reasoner.community_detector.get_community_for_node(node)
                                
                                if example_community == movie_community:
                                    community_score = 1.0
                                    break
                    
                    # Check if user node's community has overlap
                    if user_node:
                        user_community = self.graph_reasoner.community_detector.get_community_for_node(user_node)
                        
                        if user_community is not None:
                            # Get profiles
                            user_profile = self.graph_reasoner.community_detector.get_community_profile(user_community)
                            movie_profile = self.graph_reasoner.community_detector.get_community_profile(movie_community)
                            
                            # Check for genre overlap
                            if "attributes" in user_profile and "attributes" in movie_profile:
                                user_genres = user_profile["attributes"].get("favorite_genres", [])
                                movie_genres = movie_profile["attributes"].get("top_genres", [])
                                
                                overlap = set(user_genres) & set(movie_genres)
                                
                                if overlap:
                                    community_score = max(community_score, len(overlap) / max(1, len(user_genres)))
            
            # Check path score with positive examples
            if preferences["positive_examples"]:
                # Find paths between this movie and positive examples
                path_scores = []
                
                for example in preferences["positive_examples"]:
                    # Find example node
                    for node, data in self.graph.nodes(data=True):
                        if data.get("node_type") == "movie" and data.get("title", "") == example:
                            # Find paths
                            paths = self.graph_reasoner.meta_path_reasoner.find_paths(
                                node, movie_node, max_length=3, max_paths=2
                            )
                            
                            if paths:
                                # Use best path score
                                best_path = paths[0]
                                path_scores.append(best_path["score"] if "score" in best_path else 0.5)
                
                if path_scores:
                    path_score = sum(path_scores) / len(path_scores)
            
            # Calculate final score
            # Weight the components based on availability
            components = [
                (genre_score, 0.25 if preferences["genres"] else 0.0),
                (actor_score, 0.15 if preferences["actors"] else 0.0),
                (director_score, 0.1 if preferences["directors"] else 0.0),
                (year_score, 0.05 if preferences["years"] else 0.0),
                (similarity_score, 0.2 if self.embeddings and preferences["positive_examples"] else 0.0),
                (community_score, 0.15),
                (path_score, 0.1 if preferences["positive_examples"] else 0.0)
            ]
            
            # Normalize weights
            total_weight = sum(weight for _, weight in components)
            
            if total_weight > 0:
                final_score = sum(score * (weight / total_weight) for score, weight in components)
            else:
                # Default score based on popularity (degree)
                final_score = min(1.0, self.graph.degree(movie_node) / 100)
            
            # Add to scored candidates
            scored_candidates.append({
                "node": movie_node,
                "title": movie_data.get("title", movie_node),
                "year": movie_data.get("year", ""),
                "score": final_score,
                "scores": {
                    "genre": genre_score,
                    "actor": actor_score,
                    "director": director_score,
                    "year": year_score,
                    "similarity": similarity_score,
                    "community": community_score,
                    "path": path_score
                }
            })
        
        return scored_candidates
    
    def _generate_explanations(self, candidates, preferences, user_node=None):
        """Generate explanations for recommended movies"""
        recommendations = []
        
        for candidate in candidates:
            movie_node = candidate["node"]
            
            # Initialize explanation components
            explanation_components = []
            
            # Add genre-based explanation
            if candidate["scores"]["genre"] > 0.3:
                genres = []
                
                # Get movie genres
                movie_data = self.graph.nodes[movie_node]
                movie_genres = movie_data.get("genres", "[]")
                
                if isinstance(movie_genres, str):
                    try:
                        genre_list = json.loads(movie_genres)
                    except:
                        genre_list = [g.strip() for g in movie_genres.split(",")]
                else:
                    genre_list = []
                
                # Find matching genres
                matching_genres = [genre for genre in preferences["genres"] 
                                if any(genre.lower() in g.lower() for g in genre_list)]
                
                if matching_genres:
                    genre_str = ", ".join(matching_genres[:2])
                    explanation_components.append(f"matches your interest in {genre_str}")
            
            # Add actor-based explanation
            if candidate["scores"]["actor"] > 0.3:
                # Find actors connected to this movie
                movie_actors = []
                
                for neighbor in self.graph.neighbors(movie_node):
                    if neighbor in self.graph.nodes() and self.graph.nodes[neighbor].get("node_type") == "actor":
                        actor_name = self.graph.nodes[neighbor].get("name", "")
                        movie_actors.append(actor_name)
                
                # Find matching actors
                matching_actors = [actor for actor in preferences["actors"] 
                                if any(actor.lower() in a.lower() for a in movie_actors)]
                
                if matching_actors:
                    actor_str = ", ".join(matching_actors[:2])
                    explanation_components.append(f"stars {actor_str} who you like")
            
            # Add director-based explanation
            if candidate["scores"]["director"] > 0.3:
                # Find directors connected to this movie
                movie_directors = []
                
                for neighbor in self.graph.neighbors(movie_node):
                    if neighbor in self.graph.nodes() and self.graph.nodes[neighbor].get("node_type") == "director":
                        director_name = self.graph.nodes[neighbor].get("name", "")
                        movie_directors.append(director_name)
                
                # Find matching directors
                matching_directors = [director for director in preferences["directors"] 
                                   if any(director.lower() in d.lower() for d in movie_directors)]
                
                if matching_directors:
                    director_str = ", ".join(matching_directors[:1])
                    explanation_components.append(f"is directed by {director_str}")
            
            # Add path-based explanation
            if preferences["positive_examples"]:
                best_path = None
                best_example = None
                best_score = 0.0
                
                for example in preferences["positive_examples"]:
                    # Find example node
                    for node, data in self.graph.nodes(data=True):
                        if data.get("node_type") == "movie" and data.get("title", "") == example:
                            # Find paths
                            paths = self.graph_reasoner.meta_path_reasoner.find_paths(
                                node, movie_node, max_length=3, max_paths=2
                            )
                            
                            if paths:
                                path = paths[0]
                                score = path["score"] if "score" in path else 0.5
                                
                                if score > best_score:
                                    best_score = score
                                    best_path = path
                                    best_example = example
                
                if best_path and best_example and best_score > 0.3:
                    explanation = self.graph_reasoner.meta_path_reasoner.generate_explanation(best_path, personalize=True)
                    
                    # Clean up explanation - remove node IDs
                    explanation = re.sub(r'node_\w+', '', explanation)
                    
                    explanation_components.append(f"similar to {best_example} that you liked")
            
            # Add community-based explanation
            if candidate["scores"]["community"] > 0.3:
                movie_community = self.graph_reasoner.community_detector.get_community_for_node(movie_node)
                
                if movie_community is not None:
                    # Get community profile
                    profile = self.graph_reasoner.community_detector.get_community_profile(movie_community)
                    
                    if profile:
                        main_type = profile.get("main_type", "")
                        
                        if main_type == "movie":
                            # Extract key attributes
                            if "attributes" in profile:
                                attributes = profile["attributes"]
                                
                                # Check if we have top directors or actors
                                if "top_directors" in attributes and attributes["top_directors"]:
                                    director = attributes["top_directors"][0]
                                    explanation_components.append(f"from the same film style as {director}")
                                
                                elif "top_actors" in attributes and attributes["top_actors"]:
                                    actors = ", ".join(attributes["top_actors"][:2])
                                    explanation_components.append(f"features popular actors like {actors}")
                                
                                elif "dominant_era" in attributes:
                                    era = attributes["dominant_era"]
                                    explanation_components.append(f"from the {era} era of Bollywood")
            
            # Generate final explanation
            if explanation_components:
                explanation = " and ".join(explanation_components)
                explanation = explanation[0].upper() + explanation[1:]
            else:
                # Generic explanation
                explanation = "Recommended based on your preferences"
            
            # Add recommendation
            recommendations.append({
                "title": candidate["title"],
                "year": candidate["year"],
                "node_id": movie_node,
                "score": candidate["score"],
                "explanation": explanation
            })
        
        return recommendations

# Class for response generation
class ResponseGenerator:
    def __init__(self, openai_client, language="english"):
        self.openai_client = openai_client
        self.language = language
        self.logger = logging.getLogger("response_generator")
    
    def generate_response(self, user_message, intent, recommendations=None, 
                        followup_question=None, reasoning_context=None, 
                        conversation_history=None):
        """Generate a natural language response based on intent and recommendations"""
        # Determine response type
        if intent["primary_intent"] == "Greeting":
            return self._generate_greeting_response(user_message, conversation_history)
        
        elif intent["primary_intent"] == "Farewell":
            return self._generate_farewell_response(user_message, conversation_history)
        
        elif recommendations:
            return self._generate_recommendation_response(recommendations, reasoning_context, followup_question)
        
        elif followup_question:
            return self._generate_followup_response(followup_question, reasoning_context)
        
        else:
            return self._generate_general_response(user_message, intent, reasoning_context, conversation_history)
    
    def _generate_greeting_response(self, user_message, conversation_history):
        """Generate a greeting response"""
        prompt = [
            {"role": "system", "content": self._get_system_prompt() + """
            The user has just greeted you. Respond with a warm, friendly greeting introducing yourself as a Bollywood movie recommendation assistant.
            Briefly mention what you can help with (movie recommendations, information about actors, directors, etc.).
            
            Keep your response concise, friendly, and conversational.
            """},
            {"role": "user", "content": user_message}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.7)
        
        if "error" in result:
            self.logger.error(f"Error generating greeting response: {result['error']}")
            
            # Return a generic greeting
            if self.language.lower() == "hinglish":
                return "Namaste! Main aapka Bollywood movie recommendation assistant hoon. Aap kaunsi movies ke baare mein jaanna chahenge?"
            else:
                return "Hello! I'm your Bollywood movie recommendation assistant. What kind of movies would you like to know about?"
        
        return result["content"]
    
    def _generate_farewell_response(self, user_message, conversation_history):
        """Generate a farewell response"""
        prompt = [
            {"role": "system", "content": self._get_system_prompt() + """
            The user is ending the conversation. Respond with a warm, friendly farewell.
            If they've discussed movies during the conversation, you can briefly mention you hope they enjoy their movies or that you're here if they need more recommendations in the future.
            
            Keep your response concise and friendly.
            """},
            {"role": "user", "content": user_message}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.7)
        
        if "error" in result:
            self.logger.error(f"Error generating farewell response: {result['error']}")
            
            # Return a generic farewell
            if self.language.lower() == "hinglish":
                return "Alvida! Aapse baat karke accha laga. Kabhi bhi movie recommendations ke liye wapas aaiye. Dhanyavaad!"
            else:
                return "Goodbye! It was nice talking with you. Come back anytime for more movie recommendations. Thank you!"
        
        return result["content"]
    
    def _generate_recommendation_response(self, recommendations, reasoning_context, followup_question=None):
        """Generate a response with movie recommendations"""
        # Format recommendations for the prompt
        recommendations_text = ""
        for i, rec in enumerate(recommendations):
            recommendations_text += f"{i+1}. {rec['title']} ({rec['year']}) - {rec['explanation']}\n"
        
        # Format reasoning context
        context_text = ""
        if reasoning_context:
            # Add info about relevant nodes
            if "relevant_nodes" in reasoning_context and reasoning_context["relevant_nodes"]:
                context_text += "Based on: "
                node_strs = []
                for node in reasoning_context["relevant_nodes"]:
                    node_strs.append(f"{node['name']} ({node['type']})")
                context_text += ", ".join(node_strs) + "\n"
            
            # Add community info if available
            if "community_contexts" in reasoning_context and reasoning_context["community_contexts"]:
                for community in reasoning_context["community_contexts"][:1]:
                    context_text += f"Community insight: {community['description']}\n"
            
            # Add path info if available
            if "path_contexts" in reasoning_context and reasoning_context["path_contexts"]:
                for path in reasoning_context["path_contexts"][:1]:
                    if "explanation" in path:
                        context_text += f"Relationship: {path['explanation']}\n"
        
        prompt = [
            {"role": "system", "content": self._get_system_prompt() + f"""
            Create a recommendation response that recommends the following Bollywood movies to the user:
            
            {recommendations_text}
            
            Additional context:
            {context_text}
            
            Your response should:
            1. Start with a brief introduction acknowledging what the user is looking for
            2. Present each movie recommendation with its title, year, and a compelling reason to watch it
            3. Use information from the explanations provided
            4. Be enthusiastic but natural
            5. Use conversational language
            
            If a follow-up question is provided, end your response with this question.
            
            Keep the response concise, friendly, and focused on the recommendations.
            """},
            {"role": "user", "content": f"Please recommend me some Bollywood movies." + 
                                      (f" Follow up with this question: {followup_question}" if followup_question else "")}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.7)
        
        if "error" in result:
            self.logger.error(f"Error generating recommendation response: {result['error']}")
            
            # Return a generic recommendation response
            response = ""
            
            if self.language.lower() == "hinglish":
                response = "Yeh kuch movies hai jo aapko pasand aa sakti hain:\n\n"
            else:
                response = "Here are some movies you might enjoy:\n\n"
                
            # Add each recommendation
            for i, rec in enumerate(recommendations):
                if self.language.lower() == "hinglish":
                    response += f"{i+1}. {rec['title']} ({rec['year']}) - {rec['explanation']}\n"
                else:
                    response += f"{i+1}. {rec['title']} ({rec['year']}) - {rec['explanation']}\n"
            
            # Add follow-up question if provided
            if followup_question:
                response += f"\n{followup_question}"
            
            return response
        
        return result["content"]
    
    def _generate_followup_response(self, followup_question, reasoning_context):
        """Generate a response with a follow-up question"""
        # Format reasoning context
        context_text = ""
        if reasoning_context:
            # Add info about relevant nodes
            if "relevant_nodes" in reasoning_context and reasoning_context["relevant_nodes"]:
                context_text += "Based on: "
                node_strs = []
                for node in reasoning_context["relevant_nodes"]:
                    node_strs.append(f"{node['name']} ({node['type']})")
                context_text += ", ".join(node_strs) + "\n"
            
            # Add community info if available
            if "community_contexts" in reasoning_context and reasoning_context["community_contexts"]:
                for community in reasoning_context["community_contexts"][:1]:
                    context_text += f"Community insight: {community['description']}\n"
            
            # Add path info if available
            if "path_contexts" in reasoning_context and reasoning_context["path_contexts"]:
                for path in reasoning_context["path_contexts"][:1]:
                    if "explanation" in path:
                        context_text += f"Relationship: {path['explanation']}\n"
        
        prompt = [
            {"role": "system", "content": self._get_system_prompt() + f"""
            Create a response that asks the user a follow-up question to better understand their movie preferences.
            
            The follow-up question is:
            {followup_question}
            
            Additional context:
            {context_text}
            
            Your response should:
            1. Be brief and natural
            2. Acknowledge what you understand so far about their preferences (if any)
            3. Ask the follow-up question naturally, as part of the conversation
            4. Use conversational language
            
            Keep the response concise, friendly, and focused on the follow-up question.
            """},
            {"role": "user", "content": "I'm looking for movie recommendations."}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.7)
        
        if "error" in result:
            self.logger.error(f"Error generating followup response: {result['error']}")
            return followup_question
        
        return result["content"]
    
    def _generate_general_response(self, user_message, intent, reasoning_context, conversation_history):
        """Generate a general response based on intent and context"""
        # Format conversation history
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "Previous conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_text += f"User: {turn['user']}\n"
                if 'system' in turn:
                    history_text += f"Assistant: {turn['system']}\n"
        
        # Format reasoning context
        context_text = ""
        if reasoning_context:
            # Add info about relevant nodes
            if "relevant_nodes" in reasoning_context and reasoning_context["relevant_nodes"]:
                context_text += "Relevant information about: "
                node_strs = []
                for node in reasoning_context["relevant_nodes"]:
                    node_strs.append(f"{node['name']} ({node['type']})")
                context_text += ", ".join(node_strs) + "\n"
            
            # Add neighborhood contexts
            if "neighborhood_contexts" in reasoning_context and reasoning_context["neighborhood_contexts"]:
                for context in reasoning_context["neighborhood_contexts"][:2]:
                    context_text += f"About {context['node_name']} ({context['node_type']}):\n"
                    
                    # Add neighbor information grouped by type
                    for type_name, neighbors in context.get("neighbor_groups", {}).items():
                        if neighbors:
                            names = [n["name"] for n in neighbors[:3]]
                            if names:
                                context_text += f"- Connected to {len(neighbors)} {type_name}s including {', '.join(names)}\n"
                    
                    context_text += "\n"
            
            # Add path contexts
            if "path_contexts" in reasoning_context and reasoning_context["path_contexts"]:
                context_text += "Relationships:\n"
                for path in reasoning_context["path_contexts"][:2]:
                    if "explanation" in path:
                        context_text += f"- {path['explanation']}\n"
        
        prompt = [
            {"role": "system", "content": self._get_system_prompt() + f"""
            Create a response to the user's message based on their intent and the provided context.
            
            User's intent: {intent['primary_intent']}
            Intent reasoning: {intent.get('reasoning', 'Not provided')}
            
            {history_text}
            
            Knowledge graph context:
            {context_text}
            
            Your response should:
            1. Directly address the user's question or statement
            2. Use the provided context to give accurate information
            3. Be conversational and natural
            4. If you don't have specific information about something, acknowledge that rather than making it up
            
            Keep the response concise, informative, and helpful.
            """},
            {"role": "user", "content": user_message}
        ]
        
        result = self.openai_client.chat_completion(prompt, temperature=0.7)
        
        if "error" in result:
            self.logger.error(f"Error generating general response: {result['error']}")
            
            # Return a generic response
            if self.language.lower() == "hinglish":
                return "Maaf kijiye, mujhe aapka sawal samajh nahi aaya. Kya aap thoda specific ho sakte hain ki aap kaise Bollywood movies ke baare mein jaanna chahte hain?"
            else:
                return "I'm sorry, I didn't quite understand your question. Could you be more specific about what kind of Bollywood movies you're interested in?"
        
        return result["content"]
    
    def _get_system_prompt(self):
        """Get the base system prompt based on language preference"""
        if self.language.lower() == "hinglish":
            return """You are a knowledgeable and friendly Bollywood movie recommendation assistant.

You communicate in Hinglish (a mix of Hindi and English) that sounds natural and conversational. Use Hindi for emotional expressions and common phrases, and English for technical terms or movie titles.

Your responses should:
1. Be warm, friendly, and conversational
2. Show enthusiasm for Bollywood films
3. Provide informative and helpful content
4. Use natural-sounding Hinglish
5. Be concise and to the point
"""
        else:
            return """You are a knowledgeable and friendly Bollywood movie recommendation assistant.

Your responses should:
1. Be warm, friendly, and conversational
2. Show enthusiasm for Bollywood films
3. Provide informative and helpful content
4. Be concise and to the point
"""

# Main Conversational Agent Class
class BollywoodConversationalAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("bollywood_agent")
        
        # Initialize components
        self.openai_client = OpenAIClient(
            api_key=self.config["openai_api_key"],
            model=self.config["model"],
            cache_dir=os.path.join(self.config["data_dir"], "cache")
        )
        
        # Load knowledge graph and embeddings
        self.graph = self._load_graph()
        self.textual_embeddings = self._load_embeddings("textual")
        self.neural_embeddings = self._load_embeddings("neural")
        self.combined_embeddings = self._combine_embeddings()
        self.quantum_model = self._load_model()
        
        # Initialize reasoning components
        self.meta_path_reasoner = self._initialize_meta_path_reasoner()
        self.community_detector = self._initialize_community_detector()
        
        # Initialize NLP components
        self.graph_reasoner = GraphReasoner(
            self.graph,
            meta_path_reasoner=self.meta_path_reasoner,
            community_detector=self.community_detector,
            embeddings=self.combined_embeddings
        )
        
        self.entity_recognizer = EntityRecognizer(
            self.openai_client,
            graph=self.graph
        )
        
        self.intent_recognizer = IntentRecognizer(
            self.openai_client
        )
        
        self.followup_generator = FollowupQuestionGenerator(
            self.openai_client,
            self.graph_reasoner
        )
        
        self.recommendation_generator = RecommendationGenerator(
            self.openai_client,
            self.graph_reasoner,
            graph=self.graph,
            embeddings=self.combined_embeddings,
            quantum_model=self.quantum_model
        )
        
        self.response_generator = ResponseGenerator(
            self.openai_client,
            language=self.config["language"]
        )
        
        # Conversation state
        self.conversations = {}
        
        self.logger.info("Bollywood Conversational Recommendation Agent initialized")
    
    def _load_graph(self):
        """Load knowledge graph from file"""
        graph_path = self.config["graph_path"]
        
        if not os.path.exists(graph_path):
            self.logger.error(f"Graph file not found: {graph_path}")
            return nx.Graph()
        
        try:
            self.logger.info(f"Loading graph from {graph_path}")
            
            if graph_path.endswith(".pkl") or graph_path.endswith(".pickle"):
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
            elif graph_path.endswith(".graphml"):
                graph = nx.read_graphml(graph_path)
            else:
                raise ValueError(f"Unsupported graph file format: {graph_path}")
            
            self.logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}")
            return nx.Graph()
    
    def _load_embeddings(self, embedding_type):
        """Load embeddings from file"""
        if embedding_type == "textual":
            path = self.config["textual_embeddings_path"]
        else:
            path = self.config["neural_embeddings_path"]
        
        if not os.path.exists(path):
            self.logger.warning(f"{embedding_type.capitalize()} embeddings file not found: {path}")
            return {}
        
        try:
            self.logger.info(f"Loading {embedding_type} embeddings from {path}")
            
            with open(path, 'rb') as f:
                embeddings = pickle.load(f)
            
            self.logger.info(f"Loaded {len(embeddings)} {embedding_type} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error loading {embedding_type} embeddings: {str(e)}")
            return {}
    
    def _combine_embeddings(self):
        """Combine textual and neural embeddings"""
        # If we only have one embedding type, use that
        if self.textual_embeddings and not self.neural_embeddings:
            return self.textual_embeddings
        
        if self.neural_embeddings and not self.textual_embeddings:
            return self.neural_embeddings
        
        if not self.textual_embeddings and not self.neural_embeddings:
            return {}
        
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
        
        self.logger.info(f"Created combined embeddings for {len(combined)} nodes")
        return combined
    
    def _load_model(self):
        """Load quantum model from file"""
        model_path = self.config["model_path"]
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            self.logger.info(f"Loading model from {model_path}")
            
            if torch.cuda.is_available():
                model = torch.load(model_path)
            else:
                model = torch.load(model_path, map_location=torch.device('cpu'))
            
            self.logger.info("Loaded quantum model successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def _initialize_meta_path_reasoner(self):
        """Initialize meta-path reasoner"""
        from importlib.util import find_spec
        
        # Check if the AdvancedMetaPathReasoner is available
        
        try:
            from advance_kg import AdvancedMetaPathReasoner
            
            # Create reasoner
            reasoner = AdvancedMetaPathReasoner(self.graph)
            reasoner.extract_meta_path_schemas(max_length=4, sample_nodes=100)
            
            self.logger.info("Initialized AdvancedMetaPathReasoner")
            return reasoner
            
        except ImportError:
            self.logger.warning("Could not import AdvancedMetaPathReasoner, using simplified version")
        
        # Simplified version from GraphReasoner
        return self.graph_reasoner._create_mock_meta_path_reasoner()
    
    def _initialize_community_detector(self):
        """Initialize community detector"""
        from importlib.util import find_spec
        
        # Check if the EnhancedCommunityDetector is available
    
        try:
            from advance_kg import EnhancedCommunityDetector
            
            # Create detector
            detector = EnhancedCommunityDetector(self.graph)
            
            # Run detection
            detector.detect_communities(
                method="louvain",
                resolution=0.7,
                min_size=5,
                hierarchical=True,
                overlapping=True,
                levels=3
            )
            
            self.logger.info("Initialized EnhancedCommunityDetector")
            return detector
            
        except ImportError:
            self.logger.warning("Could not import EnhancedCommunityDetector, using simplified version")
    
        # Try to load from cached file
        community_profiles_path = self.config["community_profiles_path"]
        
        if os.path.exists(community_profiles_path):
            try:
                with open(community_profiles_path, 'r') as f:
                    community_profiles = json.load(f)
                
                # Need to implement integration with cached profiles
                # This is more complex and would require creating a custom detector class
                self.logger.info("Loaded community profiles from file")
            except Exception as e:
                self.logger.error(f"Error loading community profiles: {str(e)}")
        
        # Simplified version from GraphReasoner
        return self.graph_reasoner._create_mock_community_detector()
    
    def process_message(self, user_id, message):
        """Process a user message and generate a response"""
        # Ensure conversation state exists for this user
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                "history": [],
                "user_node": None,
                "entities": {},
                "preferences": {},
                "last_reasoning_context": {}
            }
        
        conversation = self.conversations[user_id]
        
        # Step 1: Recognize entities in the message
        entities = self.entity_recognizer.recognize_entities(
            message,
            conversation_history=conversation["history"]
        )
        
        # Add to conversation state
        current_turn = {
            "user": message,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 2: Recognize intent
        intent = self.intent_recognizer.recognize_intent(
            message,
            conversation_history=conversation["history"],
            entities=entities
        )
        
        current_turn["intent"] = intent
        
        # Step 3: Reasoning with knowledge graph
        reasoning_context = self.graph_reasoner.generate_reasoning_context(
            entities,
            user_node=conversation["user_node"]
        )
        
        conversation["last_reasoning_context"] = reasoning_context
        
        # Add reasoning to current turn for debugging
        current_turn["reasoning_context"] = reasoning_context
        
        # Step 4: Determine if we should recommend now
        should_recommend = intent.get("ready_for_recommendation", False)
        
        followup_question = None
        recommendations = None
        
        if should_recommend:
            # Generate recommendations
            recommendations = self.recommendation_generator.generate_recommendations(
                message,
                entities,
                reasoning_context,
                conversation["history"],
                user_node=conversation["user_node"]
            )
            
            current_turn["recommendations"] = recommendations
            
            # Generate follow-up question if we have recommendations
            if recommendations and len(recommendations) > 0:
                followup_question = self.followup_generator.generate_followup_question(
                    message,
                    entities,
                    reasoning_context,
                    conversation["history"],
                    intent=intent,
                    language=self.config["language"]
                )
        else:
            # Generate follow-up question if intent requires it
            if intent.get("requires_followup", True):
                followup_question = self.followup_generator.generate_followup_question(
                    message,
                    entities,
                    reasoning_context,
                    conversation["history"],
                    intent=intent,
                    language=self.config["language"]
                )
        
        current_turn["followup_question"] = followup_question
        
        # Step 5: Generate response
        response = self.response_generator.generate_response(
            message,
            intent,
            recommendations=recommendations,
            followup_question=followup_question,
            reasoning_context=reasoning_context,
            conversation_history=conversation["history"]
        )
        
        current_turn["system"] = response
        
        # Step 6: Update conversation history
        conversation["history"].append(current_turn)
        
        # Limit history size
        max_history = self.config["max_context_length"]
        if len(conversation["history"]) > max_history:
            conversation["history"] = conversation["history"][-max_history:]
        
        return response
    
    def get_conversation_state(self, user_id):
        """Get the current conversation state for a user"""
        if user_id in self.conversations:
            return self.conversations[user_id]
        return None
    
    def reset_conversation(self, user_id):
        """Reset conversation for a user"""
        if user_id in self.conversations:
            self.conversations[user_id] = {
                "history": [],
                "user_node": None,
                "entities": {},
                "preferences": {},
                "last_reasoning_context": {}
            }
            return True
        return False

# Command line interface for the agent
def run_cli():
    """Run the agent in command line interface mode"""
    # Load configuration
    config = Config("config.yaml")
    
    # Create agent
    agent = BollywoodConversationalAgent(config)
    
    print("=" * 50)
    print("Bollywood Conversational Recommender System")
    print("=" * 50)
    print("Type 'exit' or 'quit' to end the conversation")
    print()
    
    user_id = "cli_user"
    
    while True:
        # Get user input
        user_message = input("You: ")
        
        # Check for exit command
        if user_message.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nThank you for using our recommendation system. Goodbye!")
            break
        
        # Process message
        try:
            response = agent.process_message(user_id, user_message)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Sorry, there was an error processing your message. Please try again.")



# Main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bollywood Conversational Recommender System")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--web", action="store_true", help="Run with web interface")
    parser.add_argument("--port", type=int, default=5000, help="Port for web interface")
    
    args = parser.parse_args()
    
    
    run_cli()