# 2) ADAPTERS to call your existing code (EDIT ONLY NAMES if needed)
import networkx as nx
import numpy as np

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

# 2.2 candidate generation — point to YOUR function name(s) here
def generate_candidates(doc, G, entity_embs, rel_embs, merge_thr=0.92, refine_margin=0.15, topk=50):
    # replace with your function; examples:
    if 'generate_candidate_actions' in globals():
        return generate_candidate_actions(doc, G, entity_embs, rel_embs,
                                          merge_threshold=merge_thr, refine_margin=refine_margin, topk=topk)
    if 'generate_candidates_from_state' in globals():
        return generate_candidates_from_state(doc, G, entity_embs, rel_embs,
                                              merge_threshold=merge_thr, refine_margin=refine_margin, topk=topk)
    return []  # fallback

# 2.3 apply single action — point to YOUR apply function name here
def apply_candidate(G, cand):
    if 'apply_action' in globals():
        return apply_action(G, cand)  # your real implementation
    return G  # fallback

# 2.4 relation embeddings (RotatE) — optional
def load_relation_embeddings(path):
    if not path: return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None
