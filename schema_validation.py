# 7) (Optional) SCHEMA SANITY (type-constraint violations before/after)
import os
import json
import pandas as pd
import networkx as nx

ALLOWED = {  # extend this as needed
    "born_in":   ("PERSON","CITY"),
    "located_in":("ORG","CITY"),
}

def guess_type(node_attrs):
    # TODO: plug your type inference; fallback:
    return node_attrs.get("type","UNKNOWN")

def count_schema_violations(G):
    bad = 0
    for u,v,k,edata in G.edges(keys=True, data=True):
        rel = edata.get("relation")
        if rel in ALLOWED:
            tu, tv = guess_type(G.nodes[u]), guess_type(G.nodes[v])
            if (tu,tv) != ALLOWED[rel]:
                bad += 1
    return bad

def schema_report(exp_dir):
    rows = []
    graphs_dir = os.path.join(exp_dir,"graphs")
    if not os.path.exists(graphs_dir):
        print("No graphs directory found in", exp_dir)
        return pd.DataFrame()
        
    for fname in os.listdir(graphs_dir):
        if fname.endswith(".json"):
            path = os.path.join(graphs_dir, fname)
            try:
                with open(path) as f: 
                    data = json.load(f)
                G = nx.readwrite.json_graph.node_link_graph(data, directed=True, multigraph=True)
                doc_idx = int(fname.split("doc")[1].split("_")[0])
                phase = "before" if "before" in fname else "after"
                rows.append({"doc_idx": doc_idx, "phase": phase, "schema_violations": count_schema_violations(G)})
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue
                
    if not rows:
        print("No valid graph files found")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    pivot_df = df.pivot(index="doc_idx", columns="phase", values="schema_violations")
    
    # Handle missing columns
    for col in ["before", "after"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
            
    rep = pivot_df.reset_index()
    rep.to_csv(os.path.join(exp_dir,"schema_violations.csv"), index=False)
    print("Schema violations report saved")
    print(rep.head())
    return rep
