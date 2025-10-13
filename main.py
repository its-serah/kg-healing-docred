#!/usr/bin/env python3
"""
Main runner script for KG Healing pipeline
Orchestrates all layers and generates comprehensive summary
"""

import json
import os
import pandas as pd
from experiment import run_once_and_save
from healing_metrics import compute_semantic_gain, compute_action_efficiency
from error_correction import compute_error_correction_metrics
from schema_validation import schema_report
from config import ROTATE_NPY, GOLD

def run_complete_pipeline(n_docs=25, split="dev", 
                         merge_thr=0.92, refine_margin=0.15, topk=50,
                         with_ablation=False):
    """
    Run the complete KG healing pipeline with all evaluation layers
    
    Args:
        n_docs: Number of documents to process
        split: Which split to use (train/dev/test)
        merge_thr: Merge threshold for candidate generation
        refine_margin: Refinement margin for candidates
        topk: Top-k candidates to consider
        with_ablation: Whether to run RotatE ablation study
    """
    
    print("="*60)
    print("KG HEALING PIPELINE - COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # RUN LAYER 1: Basic experiment with structural metrics
    print("\nLAYER 1: Running main experiment...")
    exp_dir = run_once_and_save(
        split=split, n_docs=n_docs,
        merge_thr=merge_thr, refine_margin=refine_margin, topk=topk,
        relation_emb_path=ROTATE_NPY
    )
    
    # LAYER 2: Healing-specific metrics
    print("\nLAYER 2: Computing healing-specific metrics...")
    try:
        scg_df = compute_semantic_gain(exp_dir)
        eff_df = compute_action_efficiency(exp_dir)
    except Exception as e:
        print(f"Error in Layer 2 metrics: {e}")
        scg_df = None
        eff_df = None
    
    # LAYER 3: Error-correction metrics (if gold standard available)
    print("\nLAYER 3: Computing error-correction metrics...")
    try:
        error_results = compute_error_correction_metrics(exp_dir)
    except Exception as e:
        print(f"Error in Layer 3 metrics: {e}")
        error_results = None
    
    # Optional: Schema validation
    print("\nOptional: Schema validation...")
    try:
        schema_df = schema_report(exp_dir)
    except Exception as e:
        print(f"Error in schema validation: {e}")
        schema_df = None
    
    # Optional: RotatE ablation study
    if with_ablation and os.path.exists(GOLD):
        print("\nOptional: RotatE ablation study...")
        try:
            exp_with = run_once_and_save(
                split=split, n_docs=n_docs,
                relation_emb_path=ROTATE_NPY
            )
            exp_without = run_once_and_save(
                split=split, n_docs=n_docs,
                relation_emb_path=None
            )
            
            # Compare results
            def refine_accuracy(exp_dir, gold_path):
                acts = pd.read_csv(os.path.join(exp_dir,"actions.csv"))
                preds = acts[acts["type"]=="REFINE"].copy()
                if preds.empty or "p_new_relation" not in preds.columns: 
                    return float('nan')
                preds["key"] = list(zip(preds["doc_idx"], preds["p_head_id"], preds["p_tail_id"], preds["p_new_relation"]))
                gold = pd.read_csv(gold_path)
                gold = gold[gold["action_type"].str.upper()=="REFINE"].copy()
                gold["key"] = list(zip(gold["doc_idx"], gold["head_id"], gold["tail_id"], gold["new_relation"]))
                return len(set(preds["key"]) & set(gold["key"])) / max(1, len(preds))

            acc_with = refine_accuracy(exp_with, GOLD)
            acc_without = refine_accuracy(exp_without, GOLD)
            print(f"Refine accuracy — with RotatE: {acc_with} | without RotatE: {acc_without}")
            
        except Exception as e:
            print(f"Error in ablation study: {e}")
    
    # Generate comprehensive summary
    print("\nGenerating comprehensive summary...")
    summary = generate_summary(exp_dir, scg_df, eff_df, error_results, schema_df)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {exp_dir}")
    print("="*60)
    
    return exp_dir, summary

def generate_summary(exp_dir, scg_df=None, eff_df=None, error_results=None, schema_df=None):
    """Generate comprehensive summary of all results"""
    
    summary = {}
    
    # Layer 1: Structural metrics
    try:
        with open(os.path.join(exp_dir,"metrics.json")) as f:
            summary["structural"] = json.load(f)
    except:
        summary["structural"] = {}
    
    # Layer 2: Healing-specific metrics
    if scg_df is not None:
        summary["semantic_gain_mean"] = float(scg_df["semantic_gain"].mean())
    else:
        summary["semantic_gain_mean"] = None
        
    if eff_df is not None:
        summary["avg_actions_per_doc"] = float(eff_df["actions_per_doc"].mean())
    else:
        summary["avg_actions_per_doc"] = None
    
    # Layer 3: Error correction metrics
    if error_results:
        summary["error_correction"] = {}
        for metric_name, df in error_results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                summary["error_correction"][metric_name] = df.to_dict('records')
    
    # Schema validation
    if schema_df is not None and not schema_df.empty:
        summary["schema_violations"] = {
            "avg_before": float(schema_df["before"].mean()) if "before" in schema_df.columns else 0,
            "avg_after": float(schema_df["after"].mean()) if "after" in schema_df.columns else 0
        }
    
    # Save summary
    with open(os.path.join(exp_dir,"summary_snapshot.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Wrote:", os.path.join(exp_dir,"summary_snapshot.json"))
    return summary

if __name__ == "__main__":
    # Run with default parameters
    exp_dir, summary = run_complete_pipeline()
    
    # Print key results
    print("\nKEY RESULTS:")
    print("-" * 40)
    if "structural" in summary and "docs" in summary["structural"]:
        print(f"Documents processed: {summary['structural']['docs']}")
    if summary.get("semantic_gain_mean"):
        print(f"Mean semantic gain: {summary['semantic_gain_mean']:.4f}")
    if summary.get("avg_actions_per_doc"):
        print(f"Avg actions per doc: {summary['avg_actions_per_doc']:.2f}")
    print(f"Full results: {exp_dir}")
