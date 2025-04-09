# QIRS: Quantum-Inspired Conversational Movie Recommender System

![System Architecture](architecture.png)

## 📌 Overview

QIRS (Quantum-Inspired Recommender System) is a novel conversational recommender system for Bollywood movies that synergistically integrates temporal hypergraph knowledge representations, neural ordinary differential equations, and quantum-inspired inference mechanisms. The system combines the power of structured knowledge graphs with continuous preference modeling and quantum computing concepts to provide personalized, diverse, and explainable movie recommendations through natural conversation.

This work establishes a new paradigm for conversational recommenders that effectively navigates the multidimensional space of user preferences while providing transparent, context-aware, and culturally nuanced recommendations.

## 🌟 Key Features

- **Quantum-Enhanced Recommendation Engine**: Uses principles like superposition and entanglement for better uncertainty handling and recommendation diversity
- **Temporal Hypergraph Knowledge Representation**: Models complex many-to-many relationships between movies, actors, directors, genres, and users
- **Neural ODE-Based Preference Modeling**: Captures continuous evolution of user preferences over time
- **Advanced Meta-Path Reasoning**: Provides explainable recommendations through knowledge graph paths
- **Community Detection**: Identifies clusters of related entities for better recommendations
- **Multilingual Conversation Support**: Handles conversations in English and Hinglish
- **Contextual Understanding**: Maintains conversation context for personalized recommendations

## 🏗️ System Architecture

The system consists of five main components as shown in the architecture diagram:

1. **Data Collector**: Flexible component for collecting and processing movie, user, and ratings data
2. **Temporal Hypergraph Construction**: Creates a rich knowledge graph with textual embeddings
3. **Enhanced Representation**: Detects communities and generates meta-paths for reasoning
4. **Quantum-Inspired Neural Processing**: Combines Neural ODE and Quantum blocks for advanced inference
5. **Conversational Recommendation**: Handles entity recognition, intent understanding, and natural language response generation

## 📊 Community Structure

![Community Structure Visualization](quantum_community_structure.png)

The system employs spectral community detection to identify clusters of related entities in the knowledge graph. As shown in the visualization, the Bollywood movie domain naturally organizes into distinct communities (274 detected), often centered around genres, directors, or time periods. This community structure helps the recommender system identify patterns and make better suggestions.

![Drama Community](quantum.png)

The second visualization shows a focused view of the "Drama" community, highlighting how movies, actors, and directors in this genre are interconnected. These community insights power both the recommendation engine and the explanation generation.

## 🔬 Methodology

### Temporal Hypergraph

Unlike traditional knowledge graphs that use binary relations, our hypergraph allows many-to-many relationships through hyperedges. For example, a movie's production team can be represented as a single hyperedge connecting the movie, director, cinematographer, and music director:

e_production = ({m, d, c, md}, t, w)

where m ∈ V_movie, d ∈ V_director, c ∈ V_cinematographer, md ∈ V_music_director, t is the timestamp, and w is the hyperedge weight.

### Neural ODE-based Preference Modeling

We model user preferences as a continuous-time dynamical system using neural ordinary differential equations:

dh_u(t)/dt = f_θ(h_u(t), c(t), t)

where h_u(t) represents the user's preference state, c(t) is the conversational context, and f_θ is parameterized by a neural network.

### Quantum-Inspired Inference

Instead of representing user preferences as a single vector h_u(t), we employ a density matrix ρ_u(t) ∈ ℂ^(d×d), which can be interpreted as a probabilistic mixture of preference states:

ρ_u(t) = ∑_i p_i |ψ_i⟩⟨ψ_i|

This quantum formulation enables maintaining multiple preference hypotheses simultaneously, particularly valuable when user preferences are ambiguous or evolving during conversation.

## 📚 Datasets

The system was evaluated using three datasets:

1. **MovieLens-100K**: A widely used benchmark containing 100,000 ratings from 943 users on 1,682 movies
2. **FlickScore (Indian Regional Movie Dataset)**: Specialized dataset with 10,000 ratings across 2,851 movies in 118 languages 
3. **BollyCRS**: A conversational dataset containing 1,000 multi-turn conversations based on Indian cinema

## 💻 Implementation

The implementation consists of several core Python modules:

- **graph_text.py**: Constructs the knowledge graph with textual embeddings
- **ode_rating3.py**: Implements the Neural ODE components for preference modeling
- **advance_kg.py**: Provides advanced knowledge graph reasoning capabilities
- **conversation2.py**: Implements the conversational agent interface

Key dependencies include:
- PyTorch and PyTorch Geometric
- NetworkX
- NumPy and Pandas
- Scikit-learn
- OpenAI API (for NLP components)

## 📈 Results

The system shows substantial improvements over traditional recommendation approaches:

- **12.7% improvement in RMSE** and **15.3% improvement in MAE** on the FlickScore dataset
- **10-15% gain in NDCG and recall** on MovieLens-100K
- **27.6% increase in user satisfaction** in user studies
- **31.2% improvement in recommendation diversity**

The quantum-inspired approach significantly enhances the system's ability to handle preference uncertainty and provide diverse recommendations, particularly valuable in conversational settings.

## 🔍 Example Conversation


