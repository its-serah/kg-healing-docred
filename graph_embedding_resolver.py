"""
Graph Embedding Approach for Entity Resolution.
Uses Node2Vec to learn node embeddings from the document graph structure,
then applies clustering to detect duplicate entities.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import random


class Node2VecWalk:
    """Simple Node2Vec random walk implementation."""
    
    def __init__(self, p: float = 1.0, q: float = 1.0):
        """Initialize Node2Vec walker.
        
        Args:
            p: Return parameter
            q: In-out parameter
        """
        self.p = p
        self.q = q
    
    def node2vec_walk(self, graph: nx.Graph, start_node, walk_length: int):
        """Perform a single Node2Vec walk.
        
        Args:
            graph: NetworkX graph
            start_node: Starting node
            walk_length: Length of walk
            
        Returns:
            List representing the walk
        """
        walk = [start_node]
        
        for _ in range(walk_length - 1):
            cur = walk[-1]
            neighbors = list(graph.neighbors(cur))
            
            if not neighbors:
                break
                
            if len(walk) == 1:
                # First step, uniform random
                next_node = random.choice(neighbors)
            else:
                # Node2Vec biased walk
                prev = walk[-2]
                probs = []
                
                for neighbor in neighbors:
                    if neighbor == prev:
                        # Return to previous node
                        prob = 1.0 / self.p
                    elif graph.has_edge(prev, neighbor):
                        # Common neighbor
                        prob = 1.0
                    else:
                        # New exploration
                        prob = 1.0 / self.q
                    probs.append(prob)
                
                # Normalize probabilities
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                    next_node = np.random.choice(neighbors, p=probs)
                else:
                    next_node = random.choice(neighbors)
            
            walk.append(next_node)
        
        return walk


class SimpleWord2Vec:
    """Simple Word2Vec-like implementation for node embeddings."""
    
    def __init__(self, embedding_dim: int = 128, window_size: int = 5, 
                 lr: float = 0.01, epochs: int = 10):
        """Initialize Word2Vec model.
        
        Args:
            embedding_dim: Dimension of embeddings
            window_size: Context window size
            lr: Learning rate
            epochs: Training epochs
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = lr
        self.epochs = epochs
        self.embeddings = {}
        self.vocab = set()
    
    def fit(self, walks: List[List]):
        """Train embeddings on walks.
        
        Args:
            walks: List of walks (sequences of nodes)
        """
        # Build vocabulary
        for walk in walks:
            for node in walk:
                self.vocab.add(node)
        
        # Initialize embeddings randomly
        for node in self.vocab:
            self.embeddings[node] = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Simple skip-gram training
        for epoch in range(self.epochs):
            for walk in walks:
                for i, center in enumerate(walk):
                    # Get context window
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context = walk[j]
                            
                            # Simple gradient update (simplified skip-gram)
                            center_emb = self.embeddings[center]
                            context_emb = self.embeddings[context]
                            
                            # Compute similarity
                            sim = np.dot(center_emb, context_emb)
                            prob = 1 / (1 + np.exp(-sim))  # Sigmoid
                            
                            # Gradient update
                            grad = (1 - prob) * self.lr
                            self.embeddings[center] += grad * context_emb
                            self.embeddings[context] += grad * center_emb
    
    def get_embedding(self, node):
        """Get embedding for a node."""
        return self.embeddings.get(node, np.zeros(self.embedding_dim))


class GraphEmbeddingResolver:
    """Graph embedding-based entity resolution."""
    
    def __init__(self, embedding_dim: int = 128, num_walks: int = 10, 
                 walk_length: int = 80, p: float = 1.0, q: float = 1.0):
        """Initialize graph embedding resolver.
        
        Args:
            embedding_dim: Dimension of node embeddings
            num_walks: Number of walks per node
            walk_length: Length of each walk
            p: Node2Vec return parameter
            q: Node2Vec in-out parameter
        """
        self.embedding_dim = embedding_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walker = Node2VecWalk(p=p, q=q)
        self.word2vec = SimpleWord2Vec(embedding_dim=embedding_dim)
        self.node_embeddings = {}
        self.entity_mapping = {}
    
    def build_graph(self, docs: List[Dict]) -> nx.Graph:
        """Build a graph from documents.
        
        Args:
            docs: List of documents
            
        Returns:
            NetworkX graph
        """
        graph = nx.Graph()
        self.entity_mapping = {}  # (doc_idx, ent_idx) -> node_id
        node_counter = 0
        
        # Add entity nodes
        for doc_idx, doc in enumerate(docs):
            entities = doc.get('vertexSet', [])
            
            for ent_idx, entity_cluster in enumerate(entities):
                if entity_cluster:
                    node_id = f"entity_{node_counter}"
                    entity_name = entity_cluster[0]['name']
                    entity_type = entity_cluster[0].get('type', 'UNK')
                    
                    graph.add_node(node_id, 
                                 name=entity_name, 
                                 type=entity_type,
                                 doc_idx=doc_idx,
                                 ent_idx=ent_idx)
                    
                    self.entity_mapping[(doc_idx, ent_idx)] = node_id
                    node_counter += 1
        
        # Add relation edges
        for doc_idx, doc in enumerate(docs):
            relations = doc.get('labels', [])
            
            for rel in relations:
                head_key = (doc_idx, rel['h'])
                tail_key = (doc_idx, rel['t'])
                
                if head_key in self.entity_mapping and tail_key in self.entity_mapping:
                    head_node = self.entity_mapping[head_key]
                    tail_node = self.entity_mapping[tail_key]
                    graph.add_edge(head_node, tail_node, relation=rel['r'])
        
        # Add similarity edges based on entity names/types
        nodes = list(graph.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                # Add edge if entities have similar names or same type
                name_sim = self._compute_name_similarity(data1['name'], data2['name'])
                type_match = data1['type'] == data2['type']
                
                if name_sim > 0.6 or (type_match and name_sim > 0.3):
                    graph.add_edge(node1, node2, weight=name_sim)
        
        return graph
    
    def _compute_name_similarity(self, name1: str, name2: str) -> float:
        """Compute simple name similarity."""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        if name1 == name2:
            return 1.0
        
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # Simple character overlap
        chars1 = set(name1)
        chars2 = set(name2)
        overlap = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0
        
        return overlap
    
    def learn_embeddings(self, graph: nx.Graph):
        """Learn node embeddings using Node2Vec approach.
        
        Args:
            graph: Input graph
        """
        print(f"Learning embeddings for graph with {graph.number_of_nodes()} nodes...")
        
        # Generate walks
        all_walks = []
        nodes = list(graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)  # Randomize starting nodes
            for node in nodes:
                walk = self.walker.node2vec_walk(graph, node, self.walk_length)
                all_walks.append(walk)
        
        print(f"Generated {len(all_walks)} walks")
        
        # Train Word2Vec on walks
        self.word2vec.fit(all_walks)
        
        # Store embeddings
        for node in graph.nodes():
            self.node_embeddings[node] = self.word2vec.get_embedding(node)
        
        print("Embedding learning completed!")
    
    def find_duplicate_entities(self, docs: List[Dict]) -> Dict[str, Any]:
        """Find duplicate entities using graph embeddings.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary with duplicate detection results
        """
        # Build graph
        graph = self.build_graph(docs)
        
        # Learn embeddings
        self.learn_embeddings(graph)
        
        # Extract entity information
        entities = []
        embeddings = []
        
        for node_id, node_data in graph.nodes(data=True):
            entities.append({
                'node_id': node_id,
                'name': node_data['name'],
                'type': node_data['type'],
                'doc_idx': node_data['doc_idx'],
                'ent_idx': node_data['ent_idx']
            })
            embeddings.append(self.node_embeddings[node_id])
        
        embeddings = np.array(embeddings)
        
        # Apply DBSCAN clustering to find duplicates
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = dbscan.fit_predict(embeddings)
        
        # Extract duplicate pairs
        duplicates = []
        cluster_groups = defaultdict(list)
        
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # -1 means noise/no cluster
                cluster_groups[cluster_id].append(i)
        
        # Create duplicate pairs from clusters
        for cluster_id, entity_indices in cluster_groups.items():
            if len(entity_indices) > 1:
                for i in range(len(entity_indices)):
                    for j in range(i + 1, len(entity_indices)):
                        idx1, idx2 = entity_indices[i], entity_indices[j]
                        
                        # Compute similarity
                        similarity = cosine_similarity(
                            [embeddings[idx1]], [embeddings[idx2]]
                        )[0, 0]
                        
                        duplicates.append({
                            'entity1': entities[idx1],
                            'entity2': entities[idx2],
                            'similarity': similarity,
                            'cluster_id': cluster_id,
                            'method': 'graph_embedding'
                        })
        
        print(f"Found {len(duplicates)} duplicate pairs in {len(cluster_groups)} clusters")
        
        return {
            'duplicates': duplicates,
            'embeddings': embeddings,
            'entities': entities,
            'clusters': clusters,
            'num_clusters': len(cluster_groups),
            'method': 'graph_embedding'
        }
    
    def resolve_entities(self, docs: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Complete entity resolution pipeline.
        
        Args:
            docs: List of documents
            
        Returns:
            Tuple of (resolved_docs, resolution_stats)
        """
        results = self.find_duplicate_entities(docs)
        
        # Create resolution statistics
        stats = {
            'method': 'graph_embedding',
            'total_entities': len(results['entities']),
            'duplicate_pairs_found': len(results['duplicates']),
            'num_clusters': results['num_clusters'],
            'entities_merged': sum(1 for cluster_id in results['clusters'] if cluster_id != -1),
            'avg_similarity': np.mean([d['similarity'] for d in results['duplicates']]) if results['duplicates'] else 0.0
        }
        
        # Apply merging (simplified - just add metadata)
        resolved_docs = []
        for doc in docs:
            resolved_doc = doc.copy()
            resolved_doc['graph_embedding_resolution'] = {
                'duplicates_found': len([
                    d for d in results['duplicates']
                    if d['entity1']['doc_idx'] == docs.index(doc) or 
                       d['entity2']['doc_idx'] == docs.index(doc)
                ])
            }
            resolved_docs.append(resolved_doc)
        
        return resolved_docs, stats


if __name__ == "__main__":
    # Test the graph embedding resolver
    from utils import create_sample_document
    
    # Create test documents with potential duplicates
    test_docs = [
        create_sample_document(),
        {
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG'}],  # Should match "Apple Inc."
                [{'name': 'Tim Cook', 'type': 'PER'}],  # Exact match
                [{'name': 'Cupertino', 'type': 'LOC'}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},  # Tim works at Apple
                {'h': 0, 't': 2, 'r': 'P159'}   # Apple located in Cupertino
            ]
        }
    ]
    
    # Add document indices for tracking
    for i, doc in enumerate(test_docs):
        doc['doc_idx'] = i
    
    # Run graph embedding resolution
    resolver = GraphEmbeddingResolver()
    resolved_docs, stats = resolver.resolve_entities(test_docs)
    
    print("Graph Embedding Entity Resolution Results:")
    print(f"Statistics: {stats}")
    print("\nFound duplicates:")
    results = resolver.find_duplicate_entities(test_docs)
    for dup in results['duplicates']:
        print(f"- {dup['entity1']['name']} <-> {dup['entity2']['name']} (similarity: {dup['similarity']:.3f})")
