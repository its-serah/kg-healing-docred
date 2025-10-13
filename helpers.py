# 1) HELPERS (metrics + IO; tiny + safe)
import os
import json
import numpy as np
import networkx as nx
from config import EMB_CACHE

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def save_graph_json(G, path):
    data = nx.readwrite.json_graph.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _to_simple_digraph(G):
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        D.add_edges_from((u,v) for u,v,_ in G.edges(keys=True))
    else:
        D.add_edges_from(G.edges())
    return D

def redundancy_count(G) -> int:
    D = _to_simple_digraph(G)
    count = 0
    for a in D.nodes():
        succ_a = set(D.successors(a))
        for c in succ_a:
            if succ_a & set(D.predecessors(c)):
                count += 1
    return count

def structure_summary(G):
    U = G.to_undirected()
    n = G.number_of_nodes() or 0
    m = G.number_of_edges() or 0
    comps = nx.number_connected_components(U) if n else 0
    density = nx.density(U) if n > 1 else 0.0
    avg_deg = (2*m)/n if n else 0.0
    return dict(nodes=n, edges=m, components=comps, density=density,
                redundancy=redundancy_count(G), avg_degree=avg_deg)

def load_cached_entity_embeddings(split_name, doc_idx, base_dir=EMB_CACHE, dim=768):
    d = os.path.join(base_dir, split_name, f"doc_{doc_idx}")
    if not os.path.isdir(d): return np.zeros((0, dim), np.float32)
    files = sorted([f for f in os.listdir(d) if f.startswith("entity_") and f.endswith(".npy")],
                   key=lambda x: int(x.split("_")[1].split(".")[0]))
    embs = [np.load(os.path.join(d, f)) for f in files]
    return np.stack(embs, 0) if embs else np.zeros((0, dim), np.float32)
