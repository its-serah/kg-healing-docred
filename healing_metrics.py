# 4) LAYER 2 — HEALING-SPECIFIC METRICS (semantic gain + action efficiency)
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from config import SPLIT, EMB_CACHE

# 4.1 Semantic Consistency Gain (avg cosine over edges, before vs after)
def _load_graph(path):
    with open(path) as f: data = json.load(f)
    return nx.readwrite.json_graph.node_link_graph(data, directed=True, multigraph=True)

def _cos(a,b,eps=1e-8):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na<eps or nb<eps: return 0.0
    return float(np.dot(a,b)/(na*nb))

def _avg_edge_cosine(G, split, doc_idx, emb_base):
    doc_dir = os.path.join(emb_base, split, f"doc_{doc_idx}")
    if not os.path.isdir(doc_dir): return np.nan
    files = sorted([f for f in os.listdir(doc_dir) if f.startswith("entity_") and f.endswith(".npy")],
                   key=lambda x: int(x.split("_")[1].split(".")[0]))
    if not files: return np.nan
    embs = [np.load(os.path.join(doc_dir, f)) for f in files]
    embs = np.stack(embs,0)
    vals = []
    for u,v,_k in G.edges(keys=True):
        if u<len(embs) and v<len(embs):
            vals.append(_cos(embs[u], embs[v]))
    return float(np.mean(vals)) if vals else np.nan

def compute_semantic_gain(exp_dir, split=SPLIT, emb_base=EMB_CACHE):
    df = pd.read_csv(os.path.join(exp_dir, "sanity_table.csv"))
    rows = []
    for doc_idx in df["doc_idx"].unique():
        G0 = _load_graph(os.path.join(exp_dir,"graphs",f"doc{doc_idx}_before.json"))
        G1 = _load_graph(os.path.join(exp_dir,"graphs",f"doc{doc_idx}_after.json"))
        b = _avg_edge_cosine(G0, split, doc_idx, emb_base)
        a = _avg_edge_cosine(G1, split, doc_idx, emb_base)
        rows.append({"doc_idx": doc_idx, "semantic_before": b, "semantic_after": a,
                     "semantic_gain": (a-b) if (not np.isnan(b) and not np.isnan(a)) else np.nan})
    scg = pd.DataFrame(rows)
    scg.to_csv(os.path.join(exp_dir, "semantic_gain.csv"), index=False)
    print("Mean semantic gain:", scg["semantic_gain"].mean())
    return scg

# 4.2 Action Efficiency (avg actions per doc)
def compute_action_efficiency(exp_dir):
    acts = pd.read_csv(os.path.join(exp_dir, "actions.csv"))
    eff  = acts.groupby("doc_idx").size().rename("actions_per_doc").reset_index()
    eff.to_csv(os.path.join(exp_dir, "action_efficiency.csv"), index=False)
    print("Avg actions per doc:", eff["actions_per_doc"].mean())
    return eff
