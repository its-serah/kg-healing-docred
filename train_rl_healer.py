#!/usr/bin/env python3
"""
RL Training Script for Graph Healing
Uses PPO to train an agent on the graph healing task
"""

import json
import os
from repo.rl_healing import (
    train_rl_on_dataset, 
    evaluate_healer, 
    HybridGraphEnv
)
from helpers import load_dataset
from config import SPLIT, DOCRED_PATH

def prepare_dataset_for_rl(split="train", max_docs=100):
    """
    Prepare dataset in the format expected by RL training.
    Returns list of (doc_idx, data, doc) tuples
    """
    print(f"Loading {split} data for RL training...")
    
    # Load the DocRED dataset 
    docs = load_dataset(DOCRED_PATH, split)
    
    dataset = []
    for doc_idx, doc in enumerate(docs[:max_docs]):
        # Create placeholder data object (not used in RL training)
        data = None  
        dataset.append((doc_idx, data, doc))
    
    print(f"Prepared {len(dataset)} documents for RL training")
    return dataset

def train_healing_agent(
    split="train", 
    max_docs=500, 
    timesteps=50000,
    node_emb_base="embeddings/rgcn_nodes",
    rotatE_path=None,
    save_path="models/ppo_graph_healer",
    max_candidates=15,
    max_steps=8
):
    """
    Train RL agent for graph healing task.
    
    Args:
        split: Dataset split to use for training
        max_docs: Maximum number of documents to use for training
        timesteps: Number of RL training timesteps
        node_emb_base: Path to node embeddings directory
        rotatE_path: Path to RotatE relation embeddings
        save_path: Where to save the trained model
        max_candidates: Maximum candidates per RL step
        max_steps: Maximum steps per episode
    """
    
    print("="*60)
    print("RL TRAINING - GRAPH HEALING AGENT")
    print("="*60)
    
    # Prepare dataset
    dataset = prepare_dataset_for_rl(split=split, max_docs=max_docs)
    
    # Load relation embeddings if provided
    rotatE_map = None
    if rotatE_path and os.path.exists(rotatE_path):
        print(f"Loading RotatE embeddings from {rotatE_path}")
        try:
            import numpy as np
            rotatE_map = np.load(rotatE_path, allow_pickle=True).item()
            print(f"Loaded {len(rotatE_map)} relation embeddings")
        except Exception as e:
            print(f"Error loading RotatE embeddings: {e}")
    
    # Create relation2idx mapping (simple enumeration)
    if rotatE_map:
        relation2idx = {rel: idx for idx, rel in enumerate(sorted(rotatE_map.keys()))}
    else:
        relation2idx = None
    
    print(f"\nTraining Parameters:")
    print(f"- Dataset: {split} split, {len(dataset)} docs")
    print(f"- Timesteps: {timesteps}")
    print(f"- Node embeddings: {node_emb_base}")
    print(f"- Relation embeddings: {'Yes' if rotatE_map else 'No'}")
    print(f"- Max candidates: {max_candidates}")
    print(f"- Max steps: {max_steps}")
    print(f"- Save path: {save_path}")
    
    # Train the model
    print(f"\nStarting RL training...")
    try:
        model = train_rl_on_dataset(
            dataset=dataset,
            node_emb_base=node_emb_base,
            rotatE_map=rotatE_map,
            relation2idx=relation2idx,
            timesteps=timesteps,
            save_path=save_path,
            max_candidates=max_candidates,
            max_steps=max_steps
        )
        
        print(f"\n✅ Training completed! Model saved to {save_path}")
        
        # Quick evaluation on a few examples
        print(f"\nRunning quick evaluation...")
        eval_dataset = dataset[:5]  # Use first 5 docs for quick eval
        evaluate_healer(
            model=model,
            dataset=eval_dataset,
            node_emb_base=node_emb_base,
            rotatE_map=rotatE_map,
            relation2idx=relation2idx,
            n=3,
            max_candidates=max_candidates,
            max_steps=max_steps
        )
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def evaluate_pretrained_model(
    model_path="models/ppo_graph_healer.zip",
    split="dev",
    max_docs=10,
    n_examples=5
):
    """
    Evaluate a pre-trained RL model.
    
    Args:
        model_path: Path to saved PPO model
        split: Dataset split for evaluation
        max_docs: Max docs to load for evaluation
        n_examples: Number of examples to evaluate
    """
    try:
        from stable_baselines3 import PPO
        
        print("="*60)
        print("RL EVALUATION - PRETRAINED MODEL")
        print("="*60)
        
        # Load the trained model
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        # Prepare evaluation dataset
        dataset = prepare_dataset_for_rl(split=split, max_docs=max_docs)
        
        # Run evaluation
        evaluate_healer(
            model=model,
            dataset=dataset,
            n=n_examples
        )
        
    except ImportError:
        print("❌ stable-baselines3 not available for evaluation")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate RL graph healing agent")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="Whether to train a new model or evaluate existing one")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--max_docs", type=int, default=200, 
                        help="Maximum documents to use")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="RL training timesteps")
    parser.add_argument("--model_path", default="models/ppo_graph_healer", 
                        help="Path to save/load model")
    parser.add_argument("--node_emb_base", default="embeddings/rgcn_nodes",
                        help="Directory with node embeddings")
    parser.add_argument("--rotate_path", default=None,
                        help="Path to RotatE relation embeddings")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        train_healing_agent(
            split=args.split,
            max_docs=args.max_docs,
            timesteps=args.timesteps,
            node_emb_base=args.node_emb_base,
            rotatE_path=args.rotate_path,
            save_path=args.model_path,
        )
        
    elif args.mode == "eval":
        evaluate_pretrained_model(
            model_path=f"{args.model_path}.zip",
            split=args.split,
            max_docs=args.max_docs
        )
