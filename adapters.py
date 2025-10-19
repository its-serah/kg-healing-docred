# 2) ADAPTERS to call your existing code (EDIT ONLY NAMES if needed)
import networkx as nx
import numpy as np

# Import RL-based healing functions
from repo.rl_healing import (
    generate_candidates_from_state,
    apply_action,
    score_and_filter,
    generate_candidate_actions,
    HybridGraphEnv,
    train_rl_on_dataset,
    evaluate_healer
)

# 2.1 graph builder — if you already defined build_doc_graph(), we'll use it
if 'build_doc_graph' not in globals():
    def build_doc_graph(doc):
        G = nx.MultiDiGraph()
        for i, cluster in enumerate(doc.get("vertexSet", [])):
            G.add_node(i, mentions=cluster, title=doc.get("title"))
        for r in doc.get("labels", []):
            h,t,rel = r.get("h"), r.get("t"), r.get("r")
            if h is None or t is None or rel is None: 
                continue
            G.add_edge(h, t, relation=rel, evidence=r.get("evidence", []))
        return G

# 2.2 candidate generation — using RL-based implementation
def generate_candidates(doc, G, entity_embs, rel_embs, merge_thr=0.92, refine_margin=0.15, topk=50):
    """
    Generate healing candidates using RL-based approach.
    
    Args:
        doc: Document data dict
        G: NetworkX graph
        entity_embs: dict mapping node_id -> embedding (or None for directory-based loading)
        rel_embs: dict mapping relation_label -> embedding 
        merge_thr: threshold for merge candidates
        refine_margin: margin for relation refinement (unused in RL approach)
        topk: max candidates to return
    
    Returns:
        list of candidate dicts with 'type', 'data', 'score' keys
    """
    # Convert entity_embs format if needed (handle both dict and directory formats)
    node_embs = entity_embs if isinstance(entity_embs, dict) else {}
    
    # If entity_embs is None or empty, try to create simple fallback embeddings
    if not node_embs:
        # Create dummy embeddings for each node - in real use you'd load from saved embeddings
        for node_id in G.nodes():
            node_embs[node_id] = np.random.randn(128).astype(np.float32)  # placeholder
    
    # Use RL-based candidate generation
    candidates = generate_candidates_from_state(
        G=G,
        node_embs=node_embs,
        rotatE_map=rel_embs,
        relation2idx=None,  # Will be inferred if needed
        merge_threshold=merge_thr,
        top_k_merge_per_entity=5,
        top_k_refines_per_edge=3,
        max_merge_candidates=topk
    )
    
    # Filter and return top candidates
    return score_and_filter(candidates, top_k=topk)

# 2.3 apply single action — using RL-based implementation
def apply_candidate(G, cand, node_embs=None):
    """
    Apply a healing candidate to the graph using RL-based action logic.
    
    Args:
        G: NetworkX graph to modify
        cand: candidate dict with 'type' and 'data' keys
        node_embs: optional dict mapping node_id -> embedding (for merge operations)
        
    Returns:
        NetworkX graph (modified)
    """
    # Create dummy node_embs if not provided (for merge operations)
    if node_embs is None:
        node_embs = {node_id: np.random.randn(128).astype(np.float32) for node_id in G.nodes()}
    
    # Apply the action using RL-based logic
    modified_graph, updated_node_embs = apply_action(G.copy(), node_embs, cand)
    
    return modified_graph

# 2.4 relation embeddings (RotatE) — optional
def load_relation_embeddings(path):
    if not path: return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None
