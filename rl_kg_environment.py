"""
Proper RL Environment for Knowledge Graph Healing that works with stable-baselines3
"""

import numpy as np
import networkx as nx
import random
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces  
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


class KGHealingEnv(gym.Env):
    """Gymnasium environment for RL-based Knowledge Graph Healing."""
    
    def __init__(self, dataset, max_candidates=8, max_steps=5, embedding_dim=64):
        """Initialize the KG healing environment.
        
        Args:
            dataset: List of (doc_idx, doc) tuples
            max_candidates: Maximum action candidates per step
            max_steps: Maximum steps per episode  
            embedding_dim: Dimension of node embeddings
        """
        super().__init__()
        
        self.dataset = dataset
        self.max_candidates = max_candidates
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim
        
        # Action space: 0 to max_candidates-1 for actions, max_candidates for STOP
        self.action_space = spaces.Discrete(max_candidates + 1)
        
        # Observation space: flattened graph state features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(embedding_dim + 10,),  # node embs + graph stats
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Pick random document
        self.doc_idx, self.doc = random.choice(self.dataset)
        
        # Build current graph
        self.current_graph = self._build_doc_graph(self.doc)
        self.node_embs = self._generate_node_embeddings(self.current_graph)
        
        # Track initial state
        self.initial_redundancy = self._redundancy_count(self.current_graph)
        self.initial_semantic = self._mean_head_tail_cosine(self.current_graph, self.node_embs)
        self.prev_redundancy = self.initial_redundancy
        self.prev_semantic = self.initial_semantic
        
        # Generate candidates
        self.candidates = self._generate_candidates(self.current_graph, self.node_embs)
        
        # Reset step counter
        self.steps = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute action and return new state."""
        done = False
        reward = 0.0
        info = self._get_info()
        
        # STOP action
        if action == self.max_candidates:
            done = True
            # Final reward based on overall improvement
            final_semantic = self._mean_head_tail_cosine(self.current_graph, self.node_embs)
            final_redundancy = self._redundancy_count(self.current_graph)
            
            semantic_improvement = final_semantic - self.initial_semantic
            redundancy_reduction = self.initial_redundancy - final_redundancy
            
            reward = 0.5 * semantic_improvement + 0.3 * redundancy_reduction + 0.2
            
        elif action >= len(self.candidates):
            # Invalid action
            reward = -0.1
            
        else:
            # Apply valid healing action
            candidate = self.candidates[action]
            reward = self._apply_healing_action(candidate)
            
            # Update candidates for next step
            self.candidates = self._generate_candidates(self.current_graph, self.node_embs)
        
        # Check if max steps reached
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        obs = self._get_observation()
        truncated = False  # For gymnasium compatibility
        
        return obs, reward, done, truncated, info
    
    def _build_doc_graph(self, doc: Dict) -> nx.DiGraph:
        """Build NetworkX graph from document."""
        G = nx.DiGraph()
        
        # Add entity nodes
        for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
            if entity_cluster:
                entity_name = entity_cluster[0]['name']
                entity_type = entity_cluster[0].get('type', 'UNK')
                G.add_node(ent_idx, name=entity_name, type=entity_type)
        
        # Add relation edges
        for rel in doc.get('labels', []):
            head_idx = rel['h']
            tail_idx = rel['t']
            relation = rel['r']
            if head_idx in G.nodes() and tail_idx in G.nodes():
                G.add_edge(head_idx, tail_idx, relation=relation)
        
        return G
    
    def _generate_node_embeddings(self, G: nx.DiGraph) -> Dict[int, np.ndarray]:
        """Generate deterministic node embeddings."""
        node_embs = {}
        
        for node_id, node_data in G.nodes(data=True):
            name = node_data.get('name', '')
            entity_type = node_data.get('type', 'UNK')
            
            # Create deterministic embedding
            name_hash = hash(name.lower()) % 10000
            type_hash = hash(entity_type) % 1000
            
            # Use deterministic random seed
            np.random.seed(name_hash + type_hash + 42)
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            
            # Add structured features
            if self.embedding_dim > 3:
                embedding[0] = (name_hash % 100) / 100.0  
                embedding[1] = (type_hash % 100) / 100.0  
                embedding[2] = G.degree(node_id) / max(10.0, 1.0)
            
            node_embs[node_id] = embedding.astype(np.float32)
        
        return node_embs
    
    def _redundancy_count(self, G: nx.DiGraph) -> int:
        """Count 2-hop redundancies: A->B->C where A->C exists."""
        count = 0
        for a in list(G.nodes):
            for b in G.successors(a):
                for c in G.successors(b):
                    if G.has_edge(a, c) and a != c:
                        count += 1
        return count
    
    def _mean_head_tail_cosine(self, G: nx.DiGraph, node_embs: Dict[int, np.ndarray]) -> float:
        """Mean cosine similarity between head and tail embeddings."""
        similarities = []
        for u, v in G.edges():
            if u in node_embs and v in node_embs:
                emb_u = node_embs[u].reshape(1, -1)
                emb_v = node_embs[v].reshape(1, -1)
                sim = cosine_similarity(emb_u, emb_v)[0][0]
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _generate_candidates(self, G: nx.DiGraph, node_embs: Dict[int, np.ndarray]) -> List[Dict]:
        """Generate healing action candidates."""
        candidates = []
        
        # MERGE candidates - find similar entities
        nodes = list(G.nodes())
        if len(nodes) > 1:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    
                    if node_i in node_embs and node_j in node_embs:
                        # Compute similarity
                        emb_i = node_embs[node_i].reshape(1, -1)
                        emb_j = node_embs[node_j].reshape(1, -1)
                        sim = cosine_similarity(emb_i, emb_j)[0][0]
                        
                        # Add type compatibility bonus
                        data_i = G.nodes[node_i]
                        data_j = G.nodes[node_j]
                        type_bonus = 0.1 if data_i.get('type') == data_j.get('type') else -0.1
                        
                        score = sim + type_bonus
                        
                        if score > 0.7:  # Threshold for merge candidates
                            candidates.append({
                                'type': 'merge',
                                'data': (node_i, node_j),
                                'score': float(score)
                            })
        
        # CHAIN candidates - remove redundant transitive relations
        for a in list(G.nodes):
            for b in list(G.successors(a)):
                for c in list(G.successors(b)):
                    if G.has_edge(a, c) and a != c:
                        candidates.append({
                            'type': 'chain_removal',
                            'data': (a, b, c),
                            'score': 0.6
                        })
        
        # Sort by score and limit
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.max_candidates]
    
    def _apply_healing_action(self, candidate: Dict) -> float:
        """Apply healing action and return reward."""
        action_type = candidate['type']
        reward = 0.0
        
        if action_type == 'merge':
            entity1_id, entity2_id = candidate['data']
            
            if (entity1_id in self.current_graph.nodes() and 
                entity2_id in self.current_graph.nodes() and 
                entity1_id != entity2_id):
                
                # Compute reward before merge
                old_semantic = self._mean_head_tail_cosine(self.current_graph, self.node_embs)
                old_redundancy = self._redundancy_count(self.current_graph)
                
                # Merge embeddings
                emb1 = self.node_embs.get(entity1_id, np.zeros(self.embedding_dim))
                emb2 = self.node_embs.get(entity2_id, np.zeros(self.embedding_dim))
                new_emb = (emb1 + emb2) / 2.0
                
                # Contract nodes
                try:
                    self.current_graph = nx.contracted_nodes(
                        self.current_graph, entity1_id, entity2_id, self_loops=False
                    )
                    
                    # Update embeddings
                    self.node_embs[entity1_id] = new_emb
                    if entity2_id in self.node_embs:
                        del self.node_embs[entity2_id]
                    
                    # Compute reward
                    new_semantic = self._mean_head_tail_cosine(self.current_graph, self.node_embs)
                    new_redundancy = self._redundancy_count(self.current_graph)
                    
                    semantic_reward = new_semantic - old_semantic
                    structure_reward = (old_redundancy - new_redundancy) * 0.1
                    
                    reward = 0.6 * semantic_reward + 0.4 * structure_reward + 0.1
                    
                except Exception:
                    reward = -0.2  # Penalty for failed merge
                    
        elif action_type == 'chain_removal':
            a, b, c = candidate['data']
            
            if self.current_graph.has_edge(a, c):
                # Remove redundant edge
                old_redundancy = self._redundancy_count(self.current_graph)
                self.current_graph.remove_edge(a, c)
                new_redundancy = self._redundancy_count(self.current_graph)
                
                redundancy_improvement = old_redundancy - new_redundancy
                reward = 0.3 * redundancy_improvement + 0.1
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        # Graph statistics
        num_nodes = len(self.current_graph.nodes())
        num_edges = len(self.current_graph.edges())
        density = nx.density(self.current_graph) if num_nodes > 1 else 0.0
        
        try:
            clustering = nx.average_clustering(self.current_graph.to_undirected())
        except:
            clustering = 0.0
        
        redundancy = self._redundancy_count(self.current_graph)
        semantic_sim = self._mean_head_tail_cosine(self.current_graph, self.node_embs)
        
        # Graph features
        graph_features = np.array([
            num_nodes / 20.0,  # Normalize
            num_edges / 50.0,
            density,
            clustering,
            redundancy / 10.0,
            semantic_sim,
            len(self.candidates) / self.max_candidates,
            self.steps / self.max_steps,
            float(self.initial_redundancy > 0),
            float(len(self.node_embs) > 0)
        ], dtype=np.float32)
        
        # Mean node embedding
        if self.node_embs:
            mean_emb = np.mean(np.stack(list(self.node_embs.values())), axis=0)
        else:
            mean_emb = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Combine features
        observation = np.concatenate([mean_emb, graph_features])
        
        # Ensure correct shape
        if len(observation) != self.observation_space.shape[0]:
            # Pad or trim to match expected size
            expected_size = self.observation_space.shape[0]
            if len(observation) < expected_size:
                observation = np.pad(observation, (0, expected_size - len(observation)))
            else:
                observation = observation[:expected_size]
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            'num_nodes': len(self.current_graph.nodes()),
            'num_edges': len(self.current_graph.edges()),
            'redundancy_count': self._redundancy_count(self.current_graph),
            'semantic_similarity': self._mean_head_tail_cosine(self.current_graph, self.node_embs),
            'candidates_available': len(self.candidates),
            'steps_taken': self.steps
        }


if __name__ == "__main__":
    # Test the environment
    test_dataset = [
        (0, {
            'vertexSet': [
                [{'name': 'Apple Inc.', 'type': 'ORG'}],
                [{'name': 'Tim Cook', 'type': 'PER'}], 
                [{'name': 'California', 'type': 'LOC'}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},  
                {'h': 0, 't': 2, 'r': 'P131'}
            ]
        })
    ]
    
    env = KGHealingEnv(test_dataset)
    obs, info = env.reset()
    
    print(f"Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    # Test a few random actions
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
        
        if done:
            break
    
    print("Environment test completed!")
