#!/usr/bin/env python3
"""
Main runner script for KG Healing pipeline
Orchestrates all layers and generates comprehensive summary

Now supports both:
1. Traditional evaluation-only pipeline (backward compatible)
2. Complete thesis pipeline with ML training + evaluation
"""

import json
import os
import argparse
import pandas as pd
from experiment import run_once_and_save
from healing_metrics import compute_semantic_gain, compute_action_efficiency
from error_correction import compute_error_correction_metrics
from schema_validation import schema_report
from config import ROTATE_NPY
from error_correction import GOLD

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

def run_thesis_mode(args):
    """
    Run complete thesis pipeline: EDA -> Embeddings -> Training -> RL -> Evaluation
    """
    try:
        from run_thesis_pipeline import run_thesis_pipeline, ThesisPipelineConfig
    except ImportError:
        print("Error: Thesis components not available. Install required dependencies.")
        return None, None
    
    # Create config from args
    config = ThesisPipelineConfig()
    config.data_dir = args.data_dir
    config.max_docs_train = args.max_docs_train
    config.eval_n_docs = args.n_docs
    config.eval_split = args.split
    
    # Stage control from args
    if hasattr(args, 'skip_eda') and args.skip_eda:
        config.run_eda = False
    if hasattr(args, 'skip_training') and args.skip_training:
        config.run_entity_embeddings = False
        config.run_rotate_training = False
        config.run_rgcn_training = False
        config.run_rl_training = False
    
    return run_thesis_pipeline(config)

def run_evaluation_only_mode(args):
    """
    Run traditional evaluation-only pipeline (backward compatible)
    """
    return run_complete_pipeline(
        n_docs=args.n_docs,
        split=args.split,
        merge_thr=getattr(args, 'merge_thr', 0.92),
        refine_margin=getattr(args, 'refine_margin', 0.15),
        topk=getattr(args, 'topk', 50),
        with_ablation=getattr(args, 'with_ablation', False)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Healing Pipeline")
    parser.add_argument("--mode", choices=["evaluation", "thesis"], default="evaluation",
                        help="Pipeline mode: evaluation-only or complete thesis workflow")
    
    # Common parameters
    parser.add_argument("--n_docs", type=int, default=25, help="Number of documents to process")
    parser.add_argument("--split", default="dev", help="Data split to use")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    
    # Evaluation-only parameters
    parser.add_argument("--merge_thr", type=float, default=0.92, help="Merge threshold")
    parser.add_argument("--refine_margin", type=float, default=0.15, help="Refine margin")
    parser.add_argument("--topk", type=int, default=50, help="Top-k candidates")
    parser.add_argument("--with_ablation", action="store_true", help="Run RotatE ablation")
    
    # Thesis pipeline parameters
    parser.add_argument("--max_docs_train", type=int, default=500, help="Max docs for training")
    parser.add_argument("--skip_eda", action="store_true", help="Skip EDA stage")
    parser.add_argument("--skip_training", action="store_true", help="Skip all training stages")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced params")
    
    args = parser.parse_args()
    
    print(f"Running KG Healing Pipeline in {args.mode.upper()} mode")
    print("=" * 60)
    
    if args.mode == "thesis":
        exp_dir, results = run_thesis_mode(args)
    else:
        exp_dir, results = run_evaluation_only_mode(args)
    
    # Print results
    if exp_dir and results:
        print("\nPIPELINE COMPLETED")
        print("-" * 40)
        if args.mode == "thesis":
            print(f"Thesis pipeline results: {exp_dir}")
        else:
            # Traditional results display
            summary = results if isinstance(results, dict) else {}
            if "structural" in summary and "docs" in summary["structural"]:
                print(f"Documents processed: {summary['structural']['docs']}")
            if summary.get("semantic_gain_mean"):
                print(f"Mean semantic gain: {summary['semantic_gain_mean']:.4f}")
            if summary.get("avg_actions_per_doc"):
                print(f"Avg actions per doc: {summary['avg_actions_per_doc']:.2f}")
            print(f"Evaluation results: {exp_dir}")
    else:
        print("Pipeline failed or was interrupted")
