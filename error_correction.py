# 5) LAYER 3 — ERROR-CORRECTION METRICS (needs your human gold CSV)
import os
import numpy as np
import pandas as pd

# Path to your annotated gold (from the template we made earlier)
GOLD = "human_validation_sheet.csv"  # change if needed

def compute_error_correction_metrics(exp_dir, gold_path=GOLD):
    """Compute precision/recall/F1 metrics against gold standard annotations"""
    
    if not os.path.exists(gold_path):
        print("No GOLD CSV found — skipping error-correction metrics for now.")
        return None
    
    gold = pd.read_csv(gold_path)
    acts = pd.read_csv(os.path.join(exp_dir, "actions.csv"))

    results = {}
    
    # --- MERGE precision/recall/F1 + PR curve ---
    pm = acts[acts["type"]=="MERGE"].copy()
    # Expect payload columns p_head_id / p_tail_id — edit if your names differ
    if not pm.empty and "p_head_id" in pm.columns and "p_tail_id" in pm.columns:
        pm["key"] = list(zip(pm["doc_idx"],
                             pm[["p_head_id","p_tail_id"]].min(axis=1),
                             pm[["p_head_id","p_tail_id"]].max(axis=1)))

        gm = gold[gold["action_type"].str.upper()=="MERGE"].copy()
        if not gm.empty and "head_id" in gm.columns and "tail_id" in gm.columns:
            gm["key"] = list(zip(gm["doc_idx"], gm[["head_id","tail_id"]].min(axis=1), gm[["head_id","tail_id"]].max(axis=1)))

            def pr_from_sets(pred_keys, gold_keys):
                TP = len(pred_keys & gold_keys)
                FP = len(pred_keys - gold_keys)
                FN = len(gold_keys - pred_keys)
                P = TP/(TP+FP) if (TP+FP)>0 else 0.0
                R = TP/(TP+FN) if (TP+FN)>0 else 0.0
                F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
                return P,R,F1,TP,FP,FN

            # single operating point
            P,R,F1,TP,FP,FN = pr_from_sets(set(pm["key"]), set(gm["key"]))
            merge_results = pd.DataFrame([{"metric":"MERGE","precision":P,"recall":R,"f1":F1,"tp":TP,"fp":FP,"fn":FN}])
            merge_results.to_csv(os.path.join(exp_dir,"error_correction_merge.csv"), index=False)
            results["merge"] = merge_results

            # PR curve (sweep score thresholds)
            if "score" in pm.columns:
                ths  = np.round(np.linspace(0.85, 0.97, 13), 2)
                rows = []
                gset = set(gm["key"])
                for th in ths:
                    sel = pm[pm["score"]>=th]
                    pset = set(sel["key"])
                    P,R,F1,TP,FP,FN = pr_from_sets(pset, gset)
                    rows.append({"threshold":th,"precision":P,"recall":R,"f1":F1,"tp":TP,"fp":FP,"fn":FN,"n_pred":len(pset)})
                pr_curve = pd.DataFrame(rows)
                pr_curve.to_csv(os.path.join(exp_dir,"merge_pr_curve.csv"), index=False)
                results["merge_pr_curve"] = pr_curve

    # --- REFINE accuracy (pred new_relation must match gold) ---
    prf = acts[acts["type"]=="REFINE"].copy()
    if not prf.empty and "p_new_relation" in prf.columns and "p_head_id" in prf.columns and "p_tail_id" in prf.columns:
        prf["key"] = list(zip(prf["doc_idx"], prf["p_head_id"], prf["p_tail_id"], prf["p_new_relation"]))
        
        grf = gold[gold["action_type"].str.upper()=="REFINE"].copy()
        if not grf.empty and all(col in grf.columns for col in ["head_id", "tail_id", "new_relation"]):
            grf["key"] = list(zip(grf["doc_idx"], grf["head_id"], grf["tail_id"], grf["new_relation"]))
            acc = len(set(prf["key"]) & set(grf["key"])) / max(1,len(prf))
            refine_results = pd.DataFrame([{"metric":"REFINE_ACC","accuracy":acc,"n_pred":len(prf)}])
            refine_results.to_csv(os.path.join(exp_dir,"error_correction_refine.csv"), index=False)
            results["refine"] = refine_results

    # --- CHAIN removal coverage (optional; define a key that identifies removed A->C) ---
    # Do similarly if you annotated CHAIN in gold.
    
    return results
