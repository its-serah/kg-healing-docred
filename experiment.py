# 3) RUN ONCE & FREEZE (creates before/after graphs + structural tables = LAYER 1)
import pandas as pd
import json
import os
from datetime import datetime
from config import split_map, SPLIT, OUT_ROOT, EMB_CACHE, ROTATE_NPY
from helpers import ensure_dir, save_graph_json, structure_summary, load_cached_entity_embeddings
from adapters import build_doc_graph, generate_candidates, apply_candidate, load_relation_embeddings

def run_once_and_save(split=SPLIT, n_docs=25,
                      merge_thr=0.92, refine_margin=0.15, topk=50,
                      relation_emb_path=ROTATE_NPY):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUT_ROOT, f"{ts}_{split}_exp")
    ensure_dir(exp_dir); ensure_dir(os.path.join(exp_dir, "graphs"))
    print("Saving to:", exp_dir)

    rel_embs = load_relation_embeddings(relation_emb_path)
    sanity_rows, actions_rows = [], []
    agg_before = {k:0 for k in ["nodes","edges","redundancy","components","density","avg_degree"]}
    agg_after  = {k:0 for k in ["nodes","edges","redundancy","components","density","avg_degree"]}

    docs = split_map[split][:n_docs]
    for i, doc in enumerate(docs):
        G0 = build_doc_graph(doc)
        s0 = structure_summary(G0)
        for k in agg_before: agg_before[k]+=s0[k]
        save_graph_json(G0, os.path.join(exp_dir, "graphs", f"doc{i}_before.json"))

        ent = load_cached_entity_embeddings(split, i, base_dir=EMB_CACHE)

        C = generate_candidates(doc, G0, ent, rel_embs, merge_thr=merge_thr, refine_margin=refine_margin, topk=topk)
        C_sorted = sorted(C, key=lambda x: x.get("score",0), reverse=True)

        G = G0.copy()
        for cand in C_sorted:
            G = apply_candidate(G, cand)
            actions_rows.append({
                "doc_idx": i,
                "type":  cand.get("type"),
                "score": cand.get("score"),
                **{f"p_{k}": v for k,v in cand.get("payload", {}).items()}
            })

        s1 = structure_summary(G)
        for k in agg_after: agg_after[k]+=s1[k]
        save_graph_json(G, os.path.join(exp_dir, "graphs", f"doc{i}_after.json"))

        sanity_rows.append({"doc_idx": i,
                            **{f"before_{k}": v for k,v in s0.items()},
                            **{f"after_{k}": v for k,v in s1.items()}})

    pd.DataFrame(sanity_rows).to_csv(os.path.join(exp_dir, "sanity_table.csv"), index=False)
    pd.DataFrame(actions_rows).to_csv(os.path.join(exp_dir, "actions.csv"), index=False)

    n = max(1,len(docs))
    metrics = {
        "docs": len(docs),
        "avg_before": {k: agg_before[k]/n for k in agg_before},
        "avg_after":  {k: agg_after[k]/n for k in agg_after},
        "delta": {k: (agg_after[k]-agg_before[k])/n for k in agg_after}
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({"split": split, "n_docs": n_docs,
                   "merge_thr": merge_thr, "refine_margin": refine_margin, "topk": topk,
                   "relation_emb_path": relation_emb_path, "emb_cache": EMB_CACHE}, f, indent=2)

    print("DONE ->", exp_dir)
    return exp_dir
