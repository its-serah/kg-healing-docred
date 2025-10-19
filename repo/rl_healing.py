"""
RL-based healing functions extracted from NEWFUCKINGCODE.ipynb
This module provides the core healing logic for:
- Candidate generation based on embeddings and graph structure
- Action application for merge, refine, and chain operations
- Graph metrics computation for reward calculation
"""
import numpy as np
import networkx as nx
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Gym and RL imports
import gym
from gym import spaces
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available. PPO training will not work.")


def redundancy_count(G):
    """Count number of 2-hop redundancies: A->B->C where A->C exists."""
    red = 0
    for a in list(G.nodes):
        for b in G.successors(a):
            for c in G.successors(b):
                if G.has_edge(a, c):
                    red += 1
    return red


def mean_head_tail_cosine(G, node_embs):
    """
    Semantic proxy: mean cosine similarity between head and tail node embeddings
    for all edges in G. node_embs: dict node_id -> np.array
    """
    sims = []
    for u, v in G.edges():
        if (u in node_embs) and (v in node_embs):
            a = node_embs[u].reshape(1, -1)
            b = node_embs[v].reshape(1, -1)
            s = cosine_similarity(a, b)[0][0]
            sims.append(s)
    return float(np.mean(sims)) if sims else 0.0


def clustering_score(G):
    try:
        return float(nx.average_clustering(G.to_undirected()))
    except Exception:
        return 0.0


def generate_candidates_from_state(G, node_embs, rotatE_map, relation2idx,
                                   merge_threshold=0.80, top_k_merge_per_entity=5,
                                   top_k_refines_per_edge=3, max_merge_candidates=200):
    """
    Return candidates: list of dicts {'type': 'merge'|'refine'|'chain', 'data':... , 'score':...}
    - MERGE: data=(i,j)
    - REFINE: data=(u,v, old_rel, proposed_new_rel)
    - CHAIN: data=(a,b,c) where A->B->C and A->C exists (we propose to remove A->C)
    """
    candidates = []

    # --- MERGE --- (vectorized on current nodes)
    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    if n > 1:
        emb_matrix = []
        node_idx_map = {}
        for i, nid in enumerate(nodes):
            node_idx_map[nid] = i
            emb = node_embs.get(nid, None)
            if emb is None:
                emb = np.zeros(1)  # fallback small vector
            emb_matrix.append(emb)
        emb_matrix = np.stack(emb_matrix)  # shape [n, d]
        # normalize
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = emb_matrix / norms
        # pairwise cosine via dot
        sim_mat = emb_norm.dot(emb_norm.T)
        for i_idx, i in enumerate(nodes):
            # find top neighbors for this node
            sims = sim_mat[i_idx]
            neigh_idx_sorted = np.argsort(-sims)
            c = 0
            for jpos in neigh_idx_sorted:
                if jpos == i_idx:
                    continue
                j = nodes[jpos]
                score = float(sims[jpos])
                if score >= merge_threshold:
                    candidates.append({'type': 'merge', 'data': (int(i), int(j)), 'score': score})
                    c += 1
                    if c >= top_k_merge_per_entity:
                        break

    # deduplicate unordered pairs (keep highest score)
    merge_seen = {}
    merges = []
    for cand in [c for c in candidates if c['type']=='merge']:
        a, b = cand['data']
        key = tuple(sorted((a,b)))
        if key not in merge_seen or cand['score'] > merge_seen[key]['score']:
            merge_seen[key] = cand
    merges = list(merge_seen.values())
    merges.sort(key=lambda x: x['score'], reverse=True)
    merges = merges[:max_merge_candidates]

    # --- CHAIN --- propose to REMOVE direct A->C when A->B->C and A->C exists
    chain_cands = []
    for a in list(G.nodes):
        for b in list(G.successors(a)):
            for c in list(G.successors(b)):
                if G.has_edge(a, c):
                    # propose to remove direct edge (a,c)
                    chain_cands.append({'type':'chain', 'data': (int(a), int(b), int(c)), 'score': 0.6})
    # deduplicate
    unique = []
    seen = set()
    for c in chain_cands:
        key = tuple(c['data'])
        if key not in seen:
            unique.append(c)
            seen.add(key)
    chain_cands = unique

    # --- REFINE --- for each edge propose top-k similar relation labels via rotatE vectors
    refine_cands = []
    if rotatE_map:
        rel_labels = sorted(list(rotatE_map.keys()))
        rel_mat = np.stack([rotatE_map[r] for r in rel_labels])
        rel_norms = np.linalg.norm(rel_mat, axis=1, keepdims=True)
        rel_norms[rel_norms==0] = 1.0
        rel_mat_norm = rel_mat / rel_norms
        for (u,v,data) in G.edges(data=True):
            old_rel = data.get('relation', None)
            if old_rel is None:
                continue
            if old_rel not in rotatE_map:
                continue
            old_vec = rotatE_map[old_rel]
            old_vec_norm = old_vec / (np.linalg.norm(old_vec) + 1e-12)
            sims = rel_mat_norm.dot(old_vec_norm)
            sorted_idx = np.argsort(-sims)
            proposals = []
            count = 0
            for idx in sorted_idx:
                candidate_label = rel_labels[idx]
                if candidate_label == old_rel:
                    continue
                score = float(sims[idx])
                proposals.append((candidate_label, score))
                count += 1
                if count >= top_k_refines_per_edge:
                    break
            for prop_label, sc in proposals:
                refine_cands.append({'type':'refine', 'data': (int(u), int(v), old_rel, prop_label), 'score': sc})

    # combine and sort by score
    combined = merges + chain_cands + refine_cands
    combined.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    return combined


def score_and_filter(candidates, top_k=10):
    """Candidates already contain 'score' values; sort & keep top_k."""
    if not candidates:
        return []
    candidates_sorted = sorted(candidates, key=lambda x: x.get('score', 0.0), reverse=True)
    return candidates_sorted[:top_k]


def apply_action(graph, node_embs, candidate):
    """
    Apply a healing action to the graph and update node embeddings.
    Returns the modified graph and updated node_embs dict.
    
    Args:
        graph: NetworkX graph to modify
        node_embs: dict mapping node_id -> embedding vector
        candidate: dict with 'type' and 'data' keys
        
    Returns:
        tuple: (modified_graph, updated_node_embs)
    """
    if candidate['type'] == 'merge':
        i, j = candidate['data']
        # choose canonical id = min(i,j)
        u, v = int(i), int(j)
        if (u in graph.nodes) and (v in graph.nodes) and u != v:
            # update node embedding: mean
            emb_u = node_embs.get(u, np.zeros(1))  # fallback
            emb_v = node_embs.get(v, np.zeros(1))  # fallback
            new_emb = (emb_u + emb_v) / 2.0
            # contract nodes: v -> u
            graph = nx.contracted_nodes(graph, u, v, self_loops=False)
            # update node_embs dict: set u -> new_emb, delete v (if exists)
            node_embs[u] = new_emb
            if v in node_embs:
                del node_embs[v]

    elif candidate['type'] == 'refine':
        u, v, old_rel, new_rel = candidate['data']
        if graph.has_edge(u, v):
            # set relation label to proposed
            graph[u][v]['relation'] = new_rel

    elif candidate['type'] == 'chain':
        a, b, c = candidate['data']
        # remove the direct redundant edge A->C if exists
        if graph.has_edge(a, c):
            graph.remove_edge(a, c)

    return graph, node_embs


# Legacy function for compatibility with doc-based data structure
def generate_candidate_actions(doc_data, entity_emb_dir, relation2idx=None, rotatE_edge_embeds=None,
                              merge_threshold=0.85, top_k_merge_per_entity=5, max_merge_candidates=200,
                              refine_threshold=0.6, top_k_refines_per_edge=3):
    """
    Generate healing candidates from doc_data structure (for compatibility).
    This is the original function from the notebook adapted for module use.
    
    Args:
        doc_data: Document data dict with 'vertexSet' and 'labels'
        entity_emb_dir: Directory path containing entity embeddings
        relation2idx: Optional relation to index mapping
        rotatE_edge_embeds: Dict mapping relation labels to embeddings
        merge_threshold: Cosine similarity threshold for merge candidates
        top_k_merge_per_entity: Max merge candidates per entity
        max_merge_candidates: Global max merge candidates
        refine_threshold: Similarity threshold for relation refinement
        top_k_refines_per_edge: Max refine proposals per edge
        
    Returns:
        dict with 'merge_candidates', 'chain_candidates', 'refine_candidates'
    """
    import os
    import glob
    
    # --- Load entity embeddings for this doc ---
    entity_files = sorted(glob.glob(os.path.join(entity_emb_dir, "entity_*.npy")))
    n_entities = len(entity_files)
    if n_entities == 0:
        return {"merge_candidates": [], "chain_candidates": [], "refine_candidates": []}
    entity_embs = np.stack([np.load(f) for f in entity_files])  # shape [n_entities, d]

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(entity_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    entity_embs_norm = entity_embs / norms

    # ---------- MERGE CANDIDATES (vectorized with nearest neighbors) ----------
    # Use NearestNeighbors with cosine metric (via dot-product on normalized vectors)
    nbrs = NearestNeighbors(n_neighbors=min(n_entities, top_k_merge_per_entity+1), metric='cosine').fit(entity_embs_norm)
    distances, indices = nbrs.kneighbors(entity_embs_norm)  # distances are 1 - cosine for normalized
    merge_candidates = []
    for i in range(n_entities):
        for nbr_idx, dist in zip(indices[i,1:], distances[i,1:]):  # skip first neighbor
            j = int(nbr_idx)
            if i == j:
                continue  # skip self-merges
            score = 1.0 - float(dist)  # cosine similarity
            if score >= merge_threshold:
                head_type = None
                tail_type = None
                try:
                    head_type = doc_data['vertexSet'][i][0].get('type')
                    tail_type = doc_data['vertexSet'][j][0].get('type')
                except Exception:
                    pass
                if head_type != tail_type:
                    score *= 0.5
                merge_candidates.append((i, j, score, head_type, tail_type))

    # Remove duplicates (i,j) and (j,i) keep with highest score and cap results
    seen_pairs = {}
    for a,b,score,ht,tt in merge_candidates:
        key = tuple(sorted((a,b)))
        if key not in seen_pairs or score > seen_pairs[key][2]:
            seen_pairs[key] = (a,b,score,ht,tt)
    merge_candidates = list(seen_pairs.values())
    # sort descending by score and cap
    merge_candidates.sort(key=lambda x: x[2], reverse=True)
    merge_candidates = merge_candidates[:max_merge_candidates]

    # ---------- CHAIN CANDIDATES (A -> B -> C where A->C exists) ----------
    # Build adjacency (avoid defaultdict side-effects)
    adjacency = {}
    for rel in doc_data.get('labels', []):
        h, t = rel['h'], rel['t']
        adjacency.setdefault(h, set()).add(t)

    chain_candidates = []
    # iterate snapshot of keys
    for a in list(adjacency.keys()):
        for b in list(adjacency.get(a, [])):
            # skip if b has no outgoing
            for c in list(adjacency.get(b, [])):
                if c in adjacency.get(a, set()):
                    chain_candidates.append((int(a), int(b), int(c)))
    # deduplicate chain candidates
    chain_candidates = list(dict.fromkeys(chain_candidates))

    # ---------- REFINE CANDIDATES (requires RotatE relation embeddings) ----------
    refine_candidates = []
    if rotatE_edge_embeds:
        # Build relation label list and matrix
        rel_labels = sorted(list(rotatE_edge_embeds.keys()))
        rel_mat = np.stack([rotatE_edge_embeds[r] for r in rel_labels])
        rel_norms = np.linalg.norm(rel_mat, axis=1, keepdims=True)
        rel_norms[rel_norms==0] = 1.0
        rel_mat_norm = rel_mat / rel_norms

        for edge_idx, rel in enumerate(doc_data.get('labels', [])):
            old_label = rel['r']
            if old_label not in rotatE_edge_embeds:
                continue

            old_vec = rotatE_edge_embeds[old_label]
            old_vec_norm = old_vec / (np.linalg.norm(old_vec) + 1e-12)
            sims = rel_mat_norm.dot(old_vec_norm)

            # Get top-K most similar relations (excluding self)
            sorted_idx = np.argsort(-sims)
            proposals = []
            for idx in sorted_idx:
                candidate_label = rel_labels[idx]
                if candidate_label == old_label:
                    continue
                score = float(sims[idx])
                proposals.append((candidate_label, score))
                if len(proposals) >= top_k_refines_per_edge:  # Use top-K only
                    break

            if proposals:
                refine_candidates.append({
                    "edge_idx": edge_idx,
                    "head": int(rel['h']),
                    "tail": int(rel['t']),
                    "old_rel": old_label,
                    "proposals": proposals
                })

    return {
        "merge_candidates": merge_candidates,
        "chain_candidates": chain_candidates,
        "refine_candidates": refine_candidates
    }


# ---------------------
# Gym environment (minimal class required by SB3)
# ---------------------
class HybridGraphEnv(gym.Env):
    """
    Gym env that:
      - picks a random doc from `dataset` at reset,
      - builds current graph (networkx) and node_embs dict from saved rgcn node embeddings,
      - generates candidates from current state, filters top-K,
      - action = index into candidate list OR last index = STOP.
    """
    def __init__(self, dataset, node_emb_base='embeddings/rgcn_nodes', rotatE_map=None, relation2idx=None,
                 max_candidates=10, max_steps=6, device='cpu'):
        super().__init__()
        self.dataset = dataset  # list of tuples: (doc_idx, data, doc)
        self.node_emb_base = node_emb_base
        self.rotatE_map = rotatE_map
        self.relation2idx = relation2idx
        self.max_candidates = max_candidates
        self.max_steps = max_steps
        self.device = device

        # infer obs dim by loading one doc's embedding if available
        self._infer_obs_dim()

        # action space: 0..(max_candidates-1) for candidates, max_candidates == STOP
        self.action_space = spaces.Discrete(self.max_candidates + 1)
        # observation: mean node embedding vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def _infer_obs_dim(self):
        # try to find any saved doc dir
        if not os.path.isdir(self.node_emb_base):
            self.obs_dim = 128  # fallback
            return
        dirs = sorted([d for d in os.listdir(self.node_emb_base) if d.startswith("doc_")])
        if not dirs:
            self.obs_dim = 128
            return
        sample_dir = os.path.join(self.node_emb_base, dirs[0])
        files = sorted([f for f in os.listdir(sample_dir) if f.startswith("entity_")])
        if not files:
            self.obs_dim = 128
            return
        emb = np.load(os.path.join(sample_dir, files[0]))
        self.obs_dim = emb.shape[0]

    def reset(self):
        # pick random dataset entry
        self.doc_idx, self.data, self.doc = random.choice(self.dataset)
        # current graph (mutable)
        self.current_graph = self._build_doc_graph(self.doc)  # uses your function defined earlier
        # load node embeddings for this doc into dict {node_id: np.array}
        emb_dir = os.path.join(self.node_emb_base, f"doc_{self.doc_idx}")
        self.node_embs = {}
        if os.path.isdir(emb_dir):
            for fname in os.listdir(emb_dir):
                if not fname.startswith("entity_"):
                    continue
                ent_idx = int(fname.split("_")[1].split(".")[0])
                try:
                    self.node_embs[ent_idx] = np.load(os.path.join(emb_dir, fname)).astype(np.float32)
                except Exception:
                    # fallback zero vector
                    self.node_embs[ent_idx] = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            # fallback: create zeros for nodes
            for n in self.current_graph.nodes():
                self.node_embs[int(n)] = np.zeros(self.obs_dim, dtype=np.float32)

        # bookkeeping for rewards
        self.prev_redundancy = redundancy_count(self.current_graph)
        self.prev_node_count = self.current_graph.number_of_nodes()
        self.prev_semantic = mean_head_tail_cosine(self.current_graph, self.node_embs)
        self.prev_clustering = clustering_score(self.current_graph)

        # candidates for this new state
        self.candidates = score_and_filter(generate_candidates_from_state(
            self.current_graph, self.node_embs, self.rotatE_map, self.relation2idx), top_k=self.max_candidates)

        self.steps = 0
        # initial observation
        return self._get_obs()

    def _build_doc_graph(self, doc):
        """Build NetworkX graph from document data"""
        G = nx.MultiDiGraph()
        for i, cluster in enumerate(doc.get("vertexSet", [])):
            G.add_node(i, mentions=cluster, title=doc.get("title"))
        for r in doc.get("labels", []):
            h, t, rel = r.get("h"), r.get("t"), r.get("r")
            if h is None or t is None or rel is None:
                continue
            G.add_edge(h, t, relation=rel, evidence=r.get("evidence", []))
        return G

    def _get_obs(self):
        if not self.node_embs:
            return np.zeros(self.obs_dim, dtype=np.float32)
        mean_emb = np.mean(np.stack(list(self.node_embs.values())), axis=0)
        return mean_emb.astype(np.float32)

    def step(self, action):
        done = False
        info = {}
        reward = 0.0

        # STOP action
        if action == self.max_candidates:
            done = True
            # end-of-episode bonus/penalty: small bonus if graph quality improved from start
            final_struct = redundancy_count(self.current_graph)
            final_sem = mean_head_tail_cosine(self.current_graph, self.node_embs)
            reward += 0.5 * (self.prev_semantic - final_sem)  # encourage semantic improvement
            return self._get_obs(), reward, done, info

        # if invalid candidate index (maybe fewer than max_candidates available)
        if action >= len(self.candidates):
            # small penalty for invalid/no-op
            reward -= 0.01
        else:
            cand = self.candidates[action]
            # Apply action
            if cand['type'] == 'merge':
                i, j = cand['data']
                # choose canonical id = min(i,j)
                u, v = int(i), int(j)
                if (u in self.current_graph.nodes) and (v in self.current_graph.nodes) and u != v:
                    # update node embedding: mean
                    emb_u = self.node_embs.get(u, np.zeros(self.obs_dim))
                    emb_v = self.node_embs.get(v, np.zeros(self.obs_dim))
                    new_emb = (emb_u + emb_v) / 2.0
                    # contract nodes: v -> u
                    self.current_graph = nx.contracted_nodes(self.current_graph, u, v, self_loops=False)
                    # update node_embs dict: set u -> new_emb, delete v (if exists)
                    self.node_embs[u] = new_emb
                    if v in self.node_embs:
                        del self.node_embs[v]

            elif cand['type'] == 'refine':
                u, v, old_rel, new_rel = cand['data']
                if self.current_graph.has_edge(u, v):
                    # set relation label to proposed
                    self.current_graph[u][v]['relation'] = new_rel

            elif cand['type'] == 'chain':
                a, b, c = cand['data']
                # remove the direct redundant edge A->C if exists
                if self.current_graph.has_edge(a, c):
                    self.current_graph.remove_edge(a, c)

        # recompute metrics and rewards as *delta* from previous snapshot
        new_redundancy = redundancy_count(self.current_graph)
        new_node_count = self.current_graph.number_of_nodes()
        new_semantic = mean_head_tail_cosine(self.current_graph, self.node_embs)
        new_clustering = clustering_score(self.current_graph)

        # structural reward: reduction in redundancy + node reduction
        delta_red = float(self.prev_redundancy - new_redundancy)
        delta_nodes = float(self.prev_node_count - new_node_count) / (self.prev_node_count + 1e-12)

        # semantic reward: increase in head-tail cosine
        delta_sem = float(new_semantic - self.prev_semantic)

        # global reward: clustering change + connectivity
        delta_cluster = float(new_clustering - self.prev_clustering)
        # connectivity bonus if number of connected components decreased
        conn_prev = nx.number_connected_components(self.current_graph.to_undirected()) if self.current_graph.number_of_nodes()>0 else 1
        # For simplicity we won't compute previous components here (could be stored), keep cluster delta only.

        # Weighted sum
        w_struct = 0.4
        w_sem = 0.4
        w_global = 0.2
        reward = w_struct * (delta_red + delta_nodes) + w_sem * (delta_sem) + w_global * (delta_cluster)

        # update previous metrics
        self.prev_redundancy = new_redundancy
        self.prev_node_count = new_node_count
        self.prev_semantic = new_semantic
        self.prev_clustering = new_clustering

        # regenerate candidates from updated state
        self.candidates = score_and_filter(generate_candidates_from_state(
            self.current_graph, self.node_embs, self.rotatE_map, self.relation2idx), top_k=self.max_candidates)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), float(reward), done, info


# ---------------------
# Training helper (SB3 PPO)
# ---------------------
def train_rl_on_dataset(dataset, node_emb_base='embeddings/rgcn_nodes', rotatE_map=None, relation2idx=None,
                        timesteps=20000, save_path="ppo_graph_healer", max_candidates=10, max_steps=6):
    """
    dataset: your `dataset` variable (list of (doc_idx, data, doc))
    node_emb_base: path where RGCN node embeddings were saved ('embeddings/rgcn_nodes')
    rotatE_map: rotatE_edge_embeds dict
    relation2idx: mapping in your notebook
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required for PPO training. Install with: pip install stable-baselines3")
    
    def make_env():
        return HybridGraphEnv(dataset, node_emb_base=node_emb_base, rotatE_map=rotatE_map,
                              relation2idx=relation2idx, max_candidates=max_candidates, max_steps=max_steps,
                              device='cuda' if torch.cuda.is_available() else 'cpu')

    env = DummyVecEnv([make_env])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    return model


# ---------------------
# Evaluation / toolkit
# ---------------------
def graph_stats(G):
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_connected_components(G.to_undirected()),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())
    }


def evaluate_healer(model, dataset, node_emb_base='embeddings/rgcn_nodes',
                    rotatE_map=None, relation2idx=None,
                    n=2, max_candidates=10, max_steps=6):
    """
    Evaluate a trained RL model on a dataset.
    
    Args:
        model: Trained PPO model
        dataset: List of (doc_idx, data, doc) tuples
        node_emb_base: Path to node embeddings
        rotatE_map: Relation embeddings dict
        relation2idx: Relation to index mapping
        n: Number of examples to evaluate
        max_candidates: Max candidates per step
        max_steps: Max steps per episode
    """
    if not SB3_AVAILABLE:
        print("Warning: stable-baselines3 not available. Cannot evaluate RL model.")
        return
        
    env = HybridGraphEnv(dataset, node_emb_base=node_emb_base,
                         rotatE_map=rotatE_map,
                         relation2idx=relation2idx,
                         max_candidates=max_candidates,
                         max_steps=max_steps)

    for i in range(n):
        obs = env.reset()
        G_before = env.current_graph.copy()

        # collect stats before healing
        stats_before = graph_stats(G_before)
        
        print(f"\n--- Example {i+1} BEFORE healing (doc_idx={env.doc_idx}) ---")
        print("Stats:", stats_before)
        print("Candidates available:", len(env.candidates))
        
        done, steps = False, 0
        total_reward = 0
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            total_reward += reward
            steps += 1
            
        # Stats after healing
        stats_after = graph_stats(env.current_graph)
        
        print(f"--- AFTER healing (steps={steps}) ---")
        print("Stats:", stats_after)
        print(f"Total reward: {total_reward:.4f}")
        
        # Calculate improvements
        redundancy_reduction = stats_before.get('redundancy', 0) - stats_after.get('redundancy', 0)
        print(f"Redundancy reduction: {redundancy_reduction}")
        print("-" * 50)
