#!/usr/bin/env python3
"""
IMPROVED RL approach on real DocRED data with better tuning!
Fixed thresholds and reward structure for better duplicate detection.
"""

import sys
import os
import json
import numpy as np
import networkx as nx
import random
import time
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity

# Import your actual code functions
sys.path.append('/home/serah/Downloads')

try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("PPO not available - will run in simulation mode")

def load_real_docred_data(max_docs=30):
    """Load REAL DocRED data from your Downloads folder."""
    print("🔍 Looking for real DocRED data...")
    
    # Path to your actual DocRED data
    docred_path = '/home/serah/Downloads/DocRED-20250701T100402Z-1-001/DocRED'
    
    # Try different files in order of preference
    data_files = [
        'train_annotated.json',
        'dev.json', 
        'test.json',
        'train_distant.json'
    ]
    
    data = None
    used_file = None
    
    for filename in data_files:
        filepath = os.path.join(docred_path, filename)
        if os.path.exists(filepath):
            print(f"📁 Loading real DocRED data from: {filepath}")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                used_file = filename
                print(f"✅ Successfully loaded {len(data)} documents from {filename}")
                break
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
    
    if data is None:
        raise FileNotFoundError(f"No DocRED data found in {docred_path}")
    
    # Take subset for processing speed
    if len(data) > max_docs:
        print(f"📊 Using first {max_docs} documents (out of {len(data)}) for processing speed")
        data = data[:max_docs]
    
    return data, used_file

def find_ground_truth_duplicates(docs):
    """Find potential ground truth duplicates in real DocRED data."""
    print("\n🔍 Analyzing real DocRED data for potential duplicates...")
    
    ground_truth_duplicates = []
    
    for doc_idx, doc in enumerate(docs):
        entities = doc['vertexSet']
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if entities[i] and entities[j]:
                    name1 = entities[i][0].get('name', '').lower().strip()
                    name2 = entities[j][0].get('name', '').lower().strip()
                    type1 = entities[i][0].get('type', '')
                    type2 = entities[j][0].get('type', '')
                    
                    # Check for potential duplicates
                    is_duplicate = False
                    confidence = 0.0
                    
                    if name1 == name2 and name1:  # Exact match
                        is_duplicate = True
                        confidence = 1.0
                    elif name1 and name2:
                        # Check for containment (one name is substring of another)
                        if name1 in name2 or name2 in name1:
                            if type1 == type2:  # Same type increases confidence
                                is_duplicate = True
                                confidence = 0.9
                        # Check for similar names (simple heuristics)
                        elif len(name1) > 2 and len(name2) > 2:
                            # Simple Jaccard similarity on words
                            words1 = set(name1.split())
                            words2 = set(name2.split())
                            if words1 and words2:
                                jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
                                if jaccard > 0.5 and type1 == type2:
                                    is_duplicate = True
                                    confidence = min(0.8, jaccard)
                    
                    if is_duplicate:
                        ground_truth_duplicates.append({
                            'doc_idx': doc_idx,
                            'entity1_id': i,
                            'entity2_id': j,
                            'entity1_name': entities[i][0].get('name', ''),
                            'entity2_name': entities[j][0].get('name', ''),
                            'entity1_type': type1,
                            'entity2_type': type2,
                            'confidence': confidence
                        })
    
    print(f"📊 Ground truth analysis: Found {len(ground_truth_duplicates)} potential duplicates")
    return ground_truth_duplicates

def build_doc_graph(doc):
    """Build NetworkX graph from document."""
    G = nx.DiGraph()
    n_entities = len(doc['vertexSet'])
    G.add_nodes_from(range(n_entities))
    for rel in doc.get("labels", []):
        G.add_edge(rel['h'], rel['t'], relation=rel['r'], evidence=rel['evidence'])
    return G

def create_name_based_embeddings(doc, entity_id, emb_dim=128):
    """Create better embeddings based on entity names."""
    if entity_id >= len(doc['vertexSet']) or not doc['vertexSet'][entity_id]:
        return np.zeros(emb_dim, dtype=np.float32)
    
    entity_mentions = doc['vertexSet'][entity_id]
    entity_name = entity_mentions[0]['name']
    entity_type = entity_mentions[0].get('type', 'UNK')
    
    # Use both name and type for more meaningful embeddings
    combined_str = f"{entity_name.lower()}_{entity_type}"
    
    # Create deterministic embedding based on string
    np.random.seed(hash(combined_str) % 1000000)
    embedding = np.random.normal(0, 0.3, emb_dim).astype(np.float32)
    
    # Add some structure based on entity type
    type_offset = {'PER': 0.1, 'ORG': 0.2, 'LOC': 0.3, 'MISC': 0.4}.get(entity_type, 0.5)
    embedding[0] += type_offset
    
    return embedding

def generate_candidates_improved(G, doc, node_embs, merge_threshold=0.70):  # LOWERED threshold!
    """IMPROVED candidate generation with lower thresholds for real data."""
    candidates = []
    
    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    
    if n > 1:
        # Calculate similarities between entity names directly
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i >= j:  # Avoid duplicates
                    continue
                
                if node_i >= len(doc['vertexSet']) or node_j >= len(doc['vertexSet']):
                    continue
                
                # Get entity names
                entity1 = doc['vertexSet'][node_i]
                entity2 = doc['vertexSet'][node_j]
                
                if not entity1 or not entity2:
                    continue
                
                name1 = entity1[0].get('name', '').lower().strip()
                name2 = entity2[0].get('name', '').lower().strip()
                type1 = entity1[0].get('type', '')
                type2 = entity2[0].get('type', '')
                
                # Calculate name-based similarity
                name_similarity = calculate_name_similarity(name1, name2, type1, type2)
                
                # Also use embedding similarity
                if node_i in node_embs and node_j in node_embs:
                    emb1 = node_embs[node_i].reshape(1, -1)
                    emb2 = node_embs[node_j].reshape(1, -1)
                    emb_similarity = cosine_similarity(emb1, emb2)[0][0]
                else:
                    emb_similarity = 0.0
                
                # Combined score (weighted toward name similarity for duplicates)
                combined_score = 0.7 * name_similarity + 0.3 * max(0, emb_similarity)
                
                if combined_score >= merge_threshold:
                    candidates.append({
                        'type': 'merge', 
                        'data': (int(node_i), int(node_j)), 
                        'score': float(combined_score),
                        'name1': name1,
                        'name2': name2,
                        'type1': type1,
                        'type2': type2
                    })
    
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:20]  # Return top 20

def calculate_name_similarity(name1, name2, type1, type2):
    """Calculate similarity between entity names."""
    if not name1 or not name2:
        return 0.0
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # Same type bonus
    type_bonus = 0.1 if type1 == type2 else 0.0
    
    # Substring match
    if name1 in name2 or name2 in name1:
        return 0.8 + type_bonus
    
    # Word overlap (Jaccard similarity)
    words1 = set(name1.split())
    words2 = set(name2.split())
    
    if words1 and words2:
        jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
        return jaccard + type_bonus
    
    return 0.0

class ImprovedHybridGraphEnv(gym.Env):
    """Improved environment with better candidate generation."""
    
    def __init__(self, dataset, max_candidates=10, max_steps=8):  # More steps!
        super().__init__()
        self.dataset = dataset
        self.max_candidates = max_candidates
        self.max_steps = max_steps
        self.obs_dim = 128

        self.action_space = spaces.Discrete(self.max_candidates + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.doc = random.choice(self.dataset)
        self.current_graph = build_doc_graph(self.doc)
        
        # Create better embeddings
        self.node_embs = {}
        for node_id in self.current_graph.nodes():
            self.node_embs[node_id] = create_name_based_embeddings(self.doc, node_id, self.obs_dim)

        # Generate initial candidates with improved method
        self.candidates = generate_candidates_improved(
            self.current_graph, self.doc, self.node_embs, merge_threshold=0.70)[:self.max_candidates]

        self.steps = 0
        self.merges_made = 0  # Track successful merges
        
        return self._get_obs()

    def _get_obs(self):
        if not self.node_embs:
            return np.zeros(self.obs_dim, dtype=np.float32)
        mean_emb = np.mean(np.stack(list(self.node_embs.values())), axis=0)
        return mean_emb.astype(np.float32)

    def step(self, action):
        done = False
        info = {}
        reward = 0.0

        if action == self.max_candidates:  # STOP action
            done = True
            # Reward based on number of merges made
            reward += self.merges_made * 2.0  # Big reward for finding duplicates!
            return self._get_obs(), reward, done, info

        if action >= len(self.candidates):
            reward -= 0.1  # Small penalty
        else:
            cand = self.candidates[action]
            if cand['type'] == 'merge':
                i, j = cand['data']
                u, v = int(i), int(j)
                
                if (u in self.current_graph.nodes) and (v in self.current_graph.nodes) and u != v:
                    # Big reward for high-confidence merges!
                    merge_reward = cand['score'] * 3.0  # Scale reward by confidence
                    reward += merge_reward
                    
                    # Update node embedding: mean
                    emb_u = self.node_embs.get(u, np.zeros(self.obs_dim))
                    emb_v = self.node_embs.get(v, np.zeros(self.obs_dim))
                    new_emb = (emb_u + emb_v) / 2.0
                    
                    # Contract nodes
                    self.current_graph = nx.contracted_nodes(self.current_graph, u, v, self_loops=False)
                    self.node_embs[u] = new_emb
                    if v in self.node_embs:
                        del self.node_embs[v]
                    
                    self.merges_made += 1
                    
                    # Store merge info for tracking
                    info['merge'] = {
                        'entities': (cand.get('name1', ''), cand.get('name2', '')),
                        'score': cand['score']
                    }

        # Regenerate candidates
        self.candidates = generate_candidates_improved(
            self.current_graph, self.doc, self.node_embs, merge_threshold=0.70)[:self.max_candidates]

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), float(reward), done, info

def evaluate_improved_model(model, docs, ground_truth_duplicates):
    """Evaluate with improved tracking."""
    print("\n🎯 Evaluating improved RL model...")
    
    results = {
        'duplicates_found': [],
        'total_entities': sum(len(doc['vertexSet']) for doc in docs),
        'total_docs_processed': len(docs),
        'actions_taken': 0,
        'avg_reward': 0.0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': len(ground_truth_duplicates)
    }
    
    # Convert ground truth to lookup
    gt_lookup = set()
    for gt in ground_truth_duplicates:
        doc_idx = gt['doc_idx']
        e1, e2 = gt['entity1_id'], gt['entity2_id']
        key = (doc_idx, tuple(sorted([e1, e2])))
        gt_lookup.add(key)
    
    env = ImprovedHybridGraphEnv(docs, max_candidates=10, max_steps=8)
    total_reward = 0.0
    
    for doc_idx, doc in enumerate(docs):
        env.doc = doc
        obs = env.reset()
        
        done = False
        episode_reward = 0.0
        
        while not done:
            if PPO_AVAILABLE and model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Smart simulation - take best merge candidate if score > threshold
                if env.candidates:
                    best_cand = env.candidates[0]  # Already sorted by score
                    if best_cand['score'] > 0.75:  # Lower threshold for action
                        action = 0  # Take best candidate
                    else:
                        action = env.max_candidates  # STOP
                else:
                    action = env.max_candidates
            
            obs, reward, done, info = env.step(int(action))
            episode_reward += reward
            results['actions_taken'] += 1
            
            # Track merge info
            if 'merge' in info:
                merge_info = info['merge']
                i_node, j_node = [i for i in env.current_graph.nodes() if i < len(doc['vertexSet'])][:2]  # Get merged nodes
                
                # Find the actual merged entities from candidates
                if int(action) < len(env.candidates):
                    cand = env.candidates[int(action)] if int(action) < len(env.candidates) else None
                    if cand and cand['type'] == 'merge':
                        i_node, j_node = cand['data']
                        
                        duplicate_found = {
                            'entity1': merge_info['entities'][0],
                            'entity2': merge_info['entities'][1],
                            'confidence': merge_info['score'],
                            'doc_idx': doc_idx,
                            'entity1_id': i_node,
                            'entity2_id': j_node
                        }
                        results['duplicates_found'].append(duplicate_found)
                        
                        # Check against ground truth
                        key = (doc_idx, tuple(sorted([i_node, j_node])))
                        if key in gt_lookup:
                            results['true_positives'] += 1
                            results['false_negatives'] -= 1
                        else:
                            results['false_positives'] += 1
        
        total_reward += episode_reward
    
    results['avg_reward'] = total_reward / len(docs) if docs else 0.0
    
    # Calculate metrics
    tp = results['true_positives']
    fp = results['false_positives'] 
    fn = results['false_negatives']
    
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1_score = 2 * precision * recall / max(1e-12, precision + recall)
    
    results['precision'] = precision
    results['recall'] = recall
    results['f1_score'] = f1_score
    
    return results

def run_improved_rl():
    """Run improved RL approach."""
    print("=" * 80)
    print("🚀 IMPROVED RL APPROACH ON REAL DOCRED DATA")
    print("=" * 80)
    
    # Load data
    docs, used_file = load_real_docred_data(max_docs=25)
    ground_truth = find_ground_truth_duplicates(docs)
    
    print(f"\n📋 Using {len(docs)} documents from {used_file}")
    print(f"📋 Ground truth duplicates: {len(ground_truth)}")
    
    # Show some samples
    if ground_truth:
        print(f"\n🔍 Sample duplicates to find:")
        for i, dup in enumerate(ground_truth[:5]):
            print(f"  {i+1}. '{dup['entity1_name']}' ≈ '{dup['entity2_name']}' (conf: {dup['confidence']:.2f})")
    
    model = None
    if PPO_AVAILABLE:
        print("\n🤖 Training improved PPO agent...")
        def make_env():
            return ImprovedHybridGraphEnv(docs, max_candidates=10, max_steps=8)
        
        vec_env = DummyVecEnv([make_env])
        model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu', 
                   learning_rate=1e-3,  # Higher learning rate
                   n_steps=512, 
                   batch_size=64,
                   ent_coef=0.1)  # Encourage exploration
        
        start_time = time.time()
        model.learn(total_timesteps=10000)  # More training
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate
    results = evaluate_improved_model(model, docs, ground_truth)
    
    return results, ground_truth

if __name__ == "__main__":
    results, ground_truth = run_improved_rl()
    
    print("\n" + "=" * 80)
    print("🏆 IMPROVED REAL DOCRED RL RESULTS")
    print("=" * 80)
    
    print(f"📊 Documents processed: {results['total_docs_processed']}")
    print(f"📊 Total entities: {results['total_entities']}")
    print(f"📊 Ground truth duplicates: {len(ground_truth)}")
    print(f"📊 Duplicates found by RL: {len(results['duplicates_found'])}")
    print(f"📊 Actions taken: {results['actions_taken']}")
    print(f"📊 Average reward: {results['avg_reward']:.3f}")
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"✅ True Positives: {results['true_positives']}")
    print(f"❌ False Positives: {results['false_positives']}")
    print(f"❌ False Negatives: {results['false_negatives']}")
    print(f"🎯 Precision: {results['precision']:.3f}")
    print(f"🎯 Recall: {results['recall']:.3f}")
    print(f"🏆 F1 Score: {results['f1_score']:.3f}")
    
    print(f"\n📋 Duplicates found by RL:")
    for i, dup in enumerate(results['duplicates_found'][:10]):
        # Check if it matches ground truth
        gt_match = any(gt['doc_idx'] == dup['doc_idx'] and 
                      tuple(sorted([gt['entity1_id'], gt['entity2_id']])) == 
                      tuple(sorted([dup['entity1_id'], dup['entity2_id']])) 
                      for gt in ground_truth)
        marker = "✅" if gt_match else "❓"
        print(f"  {marker} {i+1}. '{dup['entity1']}' ↔ '{dup['entity2']}' (conf: {dup['confidence']:.3f})")
    
    if len(results['duplicates_found']) > 10:
        print(f"  ... and {len(results['duplicates_found']) - 10} more duplicates")
    
    print(f"\n🎯 Ground truth examples (first 5):")
    for i, gt in enumerate(ground_truth[:5]):
        found = any(r['doc_idx'] == gt['doc_idx'] and 
                   tuple(sorted([r['entity1_id'], r['entity2_id']])) == 
                   tuple(sorted([gt['entity1_id'], gt['entity2_id']])) 
                   for r in results['duplicates_found'])
        marker = "✅ FOUND" if found else "❌ MISSED"
        print(f"  {marker}: '{gt['entity1_name']}' ↔ '{gt['entity2_name']}' (conf: {gt['confidence']:.2f})")
    
    print(f"\n🎉 Improved RL evaluation completed!")
    print(f"💡 F1 Score: {results['f1_score']:.3f} (vs 0.000 baseline)")
