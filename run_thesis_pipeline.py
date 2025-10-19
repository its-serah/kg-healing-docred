#!/usr/bin/env python3
"""
Complete pipeline runner that integrates the thesis workflow with the 3-layer evaluation system.
Runs: EDA -> Entity Embeddings -> RotatE Training -> R-GCN Training -> RL Training -> Evaluation
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Import thesis components
from thesis_components import (
    EntityEmbedder, RGCNGraphNode, RelDecoder, 
    build_pyg_graph, augment_node_features_with_edge,
    create_pos_neg_samples_safe, make_graph_dataset,
    generate_candidate_actions, setup_nltk
)

# Import preprocessing
from preprocessing import (
    load_docred_data, run_comprehensive_analysis,
    basic_dataset_stats
)

# Import existing evaluation system
from main import run_complete_pipeline
from helpers import load_dataset
from config import SPLIT, N_DOCS, DATA_DIR

# Import RL components
from repo.rl_healing import train_rl_on_dataset
from train_rl_healer import train_healing_agent

# =====================================
# Pipeline Configuration
# =====================================

class ThesisPipelineConfig:
    """Configuration for the complete thesis pipeline"""
    
    def __init__(self):
        # Data paths
        self.data_dir = "data"
        self.output_root = "experiments"
        
        # Embedding settings
        self.entity_emb_dir = "embeddings/transformer"
        self.rgcn_emb_dir = "embeddings/rgcn_nodes"
        self.rotate_emb_dir = "rotatE_embeddings/train"
        self.model_dir = "models"
        
        # Model hyperparameters
        self.transformer_model = "distilroberta-base"
        self.node_feat_dim = 768
        self.rel_feat_dim = 200  # RotatE dim (real+imag)
        self.hidden_dim = 256
        self.num_rgcn_layers = 3
        self.pool_method = 'set2set'
        
        # Training settings
        self.rotate_epochs = 50
        self.rgcn_epochs = 6
        self.rgcn_batch_size = 32
        self.rgcn_lr = 1e-3
        self.rl_timesteps = 50000
        self.max_docs_train = 500
        
        # Evaluation settings
        self.eval_n_docs = 25
        self.eval_split = "dev"
        
        # Pipeline stages
        self.run_eda = True
        self.run_entity_embeddings = True
        self.run_rotate_training = True
        self.run_rgcn_training = True
        self.run_rl_training = True
        self.run_evaluation = True

def create_experiment_dir(config):
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.output_root, f"{timestamp}_thesis_pipeline")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(exp_dir, "pipeline_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"Experiment directory: {exp_dir}")
    return exp_dir

# =====================================
# Pipeline Stages
# =====================================

def stage_eda(config, train_data, dev_data, test_data):
    """Stage 1: Comprehensive EDA"""
    print("\n" + "="*60)
    print("STAGE 1: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    data_splits = {'train': train_data, 'dev': dev_data, 'test': test_data}
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis(
        data_dir=config.data_dir,
        max_docs=200,
        create_plots=True,
        save_report=True
    )
    
    print("✅ EDA completed")
    return results

def stage_entity_embeddings(config, train_data, dev_data, test_data):
    """Stage 2: Generate entity embeddings"""
    print("\n" + "="*60)
    print("STAGE 2: ENTITY EMBEDDING GENERATION")
    print("="*60)
    
    # Initialize embedder
    embedder = EntityEmbedder(
        model_name=config.transformer_model,
        max_length=512
    )
    
    # Prepare data splits
    data_splits = {'train': train_data, 'dev': dev_data, 'test': test_data}
    
    # Compute embeddings for all splits
    for split_name, data in data_splits.items():
        if not data:
            continue
            
        split_dir = os.path.join(config.entity_emb_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\nProcessing {split_name} split ({len(data)} documents)...")
        
        for doc_idx, doc in enumerate(data):
            if doc_idx >= config.max_docs_train and split_name == 'train':
                break
                
            doc_dir = os.path.join(split_dir, f"doc_{doc_idx}")
            
            # Skip if already exists
            if os.path.exists(doc_dir) and len(os.listdir(doc_dir)) > 0:
                continue
                
            os.makedirs(doc_dir, exist_ok=True)
            
            # Generate embeddings for this document
            sents = doc.get('sents', [])
            
            for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
                contexts = []
                for mention in entity_cluster:
                    sent_id = mention.get('sent_id', 0)
                    start = max(0, sent_id - 1)  # context window
                    end = min(len(sents), sent_id + 2)
                    context_text = " ".join([" ".join(s) for s in sents[start:end]])
                    contexts.append(context_text)
                
                if contexts:
                    # Get embeddings for all mentions in cluster
                    embeddings = []
                    for context in contexts:
                        emb = embedder.get_embedding(context)
                        embeddings.append(emb)
                    
                    # Average cluster embedding
                    cluster_emb = np.mean(np.stack(embeddings), axis=0)
                    
                    # Save embedding
                    np.save(os.path.join(doc_dir, f"entity_{ent_idx}.npy"), cluster_emb)
            
            if doc_idx % 50 == 0:
                print(f"  Processed {doc_idx+1}/{min(len(data), config.max_docs_train)} documents")
    
    print("✅ Entity embeddings generated")

def stage_rotate_training(config, train_data, dev_data, test_data):
    """Stage 3: Train RotatE relation embeddings"""
    print("\n" + "="*60)
    print("STAGE 3: ROTATE RELATION EMBEDDING TRAINING")
    print("="*60)
    
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
        from thesis_components import collect_triples_from_docs
        
        # Collect triples
        print("Collecting triples from documents...")
        train_triples = collect_triples_from_docs(train_data[:config.max_docs_train])
        dev_triples = collect_triples_from_docs(dev_data)
        
        print(f"Training triples: {len(train_triples)}")
        print(f"Dev triples: {len(dev_triples)}")
        
        # Create TriplesFactory
        training_factory = TriplesFactory.from_labeled_triples(np.array(train_triples))
        validation_factory = TriplesFactory.from_labeled_triples(
            np.array(dev_triples),
            entity_to_id=training_factory.entity_to_id,
            relation_to_id=training_factory.relation_to_id
        ) if dev_triples else None
        
        # Train RotatE
        print("Training RotatE model...")
        result = pipeline(
            training=training_factory,
            validation=validation_factory,
            model='RotatE',
            model_kwargs=dict(embedding_dim=config.rel_feat_dim // 2),  # Complex embedding
            training_kwargs=dict(
                num_epochs=config.rotate_epochs, 
                use_tqdm=True, 
                batch_size=256
            ),
            random_seed=42,
        )
        
        # Extract and save embeddings
        rel_labels = list(training_factory.relation_to_id.keys())
        rel_embeds = result.model.relation_representations[0](indices=None)
        
        os.makedirs(config.rotate_emb_dir, exist_ok=True)
        rotatE_edge_embeds = {}
        
        for r in rel_labels:
            idx = training_factory.relation_to_id[r]
            embedding = rel_embeds[idx].detach().cpu().numpy()
            
            # Convert complex to real+imag concatenation
            if np.iscomplexobj(embedding):
                embedding = np.concatenate([embedding.real, embedding.imag])
            
            rotatE_edge_embeds[r] = embedding
            np.save(os.path.join(config.rotate_emb_dir, f"{r}.npy"), embedding)
        
        print(f"✅ RotatE training completed. Saved {len(rotatE_edge_embeds)} relation embeddings")
        return rotatE_edge_embeds
        
    except ImportError:
        print("⚠️  PyKEEN not available. Skipping RotatE training.")
        return {}

def stage_rgcn_training(config, train_data, rotatE_embeds):
    """Stage 4: Train R-GCN model"""
    print("\n" + "="*60)
    print("STAGE 4: R-GCN MODEL TRAINING")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build relation mapping
    all_relations = set()
    for doc in train_data[:config.max_docs_train]:
        for rel in doc.get('labels', []):
            all_relations.add(rel['r'])
    
    relation2idx = {rel: idx for idx, rel in enumerate(sorted(all_relations))}
    num_relations = len(relation2idx)
    
    print(f"Number of unique relations: {num_relations}")
    
    # Prepare dataset
    entity_base_dir = os.path.join(config.entity_emb_dir, "train")
    dataset = make_graph_dataset(
        train_data[:config.max_docs_train], 
        entity_base_dir, 
        rotatE_embeds
    )
    
    print(f"Prepared {len(dataset)} graphs for training")
    
    if not dataset:
        print("⚠️  No training data available. Skipping R-GCN training.")
        return
    
    # Model setup
    augmented_in_dim = config.node_feat_dim + config.rel_feat_dim
    rgcn_model = RGCNGraphNode(
        in_dim=augmented_in_dim,
        hidden_dim=config.hidden_dim,
        num_relations=num_relations,
        num_layers=config.num_rgcn_layers,
        pool_method=config.pool_method
    ).to(device)
    
    decoder = RelDecoder(
        node_dim=config.hidden_dim,
        rel_dim=config.rel_feat_dim,
        hidden_dim=config.hidden_dim
    ).to(device)
    
    # Relation embedding tensor
    relation_emb_tensor = torch.zeros((num_relations, config.rel_feat_dim), dtype=torch.float32)
    for rel_label, idx in relation2idx.items():
        if rel_label in rotatE_embeds:
            relation_emb_tensor[idx] = torch.tensor(rotatE_embeds[rel_label])
    relation_emb_tensor = relation_emb_tensor.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(rgcn_model.parameters()) + list(decoder.parameters()),
        lr=config.rgcn_lr
    )
    criterion = torch.nn.BCELoss()
    
    # Training loop
    rgcn_model.train()
    decoder.train()
    
    for epoch in range(config.rgcn_epochs):
        total_loss = 0.0
        
        for i, (doc_idx, data, doc) in enumerate(dataset):
            if i >= len(dataset) // config.rgcn_batch_size * config.rgcn_batch_size:
                break  # Skip partial batches
            
            data = data.to(device)
            
            # Augment node features with edge information
            x_aug = augment_node_features_with_edge(data.x, data.edge_index, data.edge_attr)
            
            # Forward pass
            node_embs, graph_emb = rgcn_model(x_aug, data.edge_index, data.edge_type, data.batch)
            
            # Generate positive and negative samples
            pos, neg = create_pos_neg_samples_safe(data, neg_ratio=1)
            
            # Predictions
            h_emb_pos = node_embs[pos[0]]
            t_emb_pos = node_embs[pos[1]]  
            r_emb_pos = relation_emb_tensor[pos[2]]
            pos_scores = decoder(h_emb_pos, r_emb_pos, t_emb_pos)
            
            h_emb_neg = node_embs[neg[0]]
            t_emb_neg = node_embs[neg[1]]
            r_emb_neg = relation_emb_tensor[neg[2]]
            neg_scores = decoder(h_emb_neg, r_emb_neg, t_emb_neg)
            
            # Loss computation
            y_pos = torch.ones_like(pos_scores, device=device)
            y_neg = torch.zeros_like(neg_scores, device=device)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([y_pos, y_neg])
            
            loss = criterion(scores, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{config.rgcn_epochs}, Loss: {avg_loss:.4f}")
    
    # Save models
    os.makedirs(config.model_dir, exist_ok=True)
    torch.save(rgcn_model.state_dict(), os.path.join(config.model_dir, 'rgcn_thesis.pt'))
    torch.save(decoder.state_dict(), os.path.join(config.model_dir, 'decoder_thesis.pt'))
    
    # Save node embeddings
    rgcn_model.eval()
    os.makedirs(config.rgcn_emb_dir, exist_ok=True)
    
    with torch.no_grad():
        for doc_idx, data, doc in dataset:
            data = data.to(device)
            x_aug = augment_node_features_with_edge(data.x, data.edge_index, data.edge_attr)
            node_embs, _ = rgcn_model(x_aug, data.edge_index, data.edge_type, data.batch)
            node_embs = node_embs.cpu().numpy()
            
            out_dir = os.path.join(config.rgcn_emb_dir, f"doc_{doc_idx}")
            os.makedirs(out_dir, exist_ok=True)
            
            for ent_idx in range(node_embs.shape[0]):
                np.save(os.path.join(out_dir, f"entity_{ent_idx}.npy"), node_embs[ent_idx])
    
    print("✅ R-GCN training completed")
    return rgcn_model, decoder, relation2idx

def stage_rl_training(config, train_data, rotatE_embeds, relation2idx):
    """Stage 5: RL training for graph healing"""
    print("\n" + "="*60)
    print("STAGE 5: REINFORCEMENT LEARNING TRAINING")
    print("="*60)
    
    # Prepare dataset for RL
    rl_dataset = []
    for doc_idx, doc in enumerate(train_data[:config.max_docs_train]):
        rl_dataset.append((doc_idx, None, doc))  # (doc_idx, data, doc) format
    
    print(f"Prepared {len(rl_dataset)} documents for RL training")
    
    try:
        # Train RL agent
        model = train_rl_on_dataset(
            dataset=rl_dataset,
            node_emb_base=config.rgcn_emb_dir,
            rotatE_map=rotatE_embeds,
            relation2idx=relation2idx,
            timesteps=config.rl_timesteps,
            save_path=os.path.join(config.model_dir, "ppo_thesis_healer"),
            max_candidates=15,
            max_steps=8
        )
        
        print("✅ RL training completed")
        return model
        
    except ImportError:
        print("⚠️  stable-baselines3 not available. Skipping RL training.")
        return None
    except Exception as e:
        print(f"⚠️  RL training failed: {e}")
        return None

def stage_evaluation(config, exp_dir):
    """Stage 6: Run 3-layer evaluation system"""
    print("\n" + "="*60)
    print("STAGE 6: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Run the complete evaluation pipeline
    eval_exp_dir, summary = run_complete_pipeline(
        n_docs=config.eval_n_docs,
        split=config.eval_split
    )
    
    print(f"✅ Evaluation completed. Results in: {eval_exp_dir}")
    
    # Copy evaluation results to main experiment directory
    import shutil
    eval_target = os.path.join(exp_dir, "evaluation_results")
    if os.path.exists(eval_target):
        shutil.rmtree(eval_target)
    shutil.copytree(eval_exp_dir, eval_target)
    
    return eval_exp_dir, summary

# =====================================
# Main Pipeline Runner
# =====================================

def run_thesis_pipeline(config):
    """Run the complete thesis pipeline"""
    print("🚀 STARTING COMPLETE THESIS PIPELINE")
    print("="*60)
    print(f"Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config)
    
    # Setup NLTK
    setup_nltk()
    
    # Load data
    print("\nLoading DocRED dataset...")
    train_data, dev_data, test_data = load_docred_data(config.data_dir)
    
    # Create output directories
    for dir_path in [config.entity_emb_dir, config.rgcn_emb_dir, 
                     config.rotate_emb_dir, config.model_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Pipeline execution
    results = {}
    start_time = time.time()
    
    # Stage 1: EDA
    if config.run_eda:
        stage_start = time.time()
        results['eda'] = stage_eda(config, train_data, dev_data, test_data)
        print(f"⏱️  EDA completed in {time.time() - stage_start:.1f}s")
    
    # Stage 2: Entity Embeddings
    if config.run_entity_embeddings:
        stage_start = time.time()
        stage_entity_embeddings(config, train_data, dev_data, test_data)
        print(f"⏱️  Entity embeddings completed in {time.time() - stage_start:.1f}s")
    
    # Stage 3: RotatE Training
    rotatE_embeds = {}
    if config.run_rotate_training:
        stage_start = time.time()
        rotatE_embeds = stage_rotate_training(config, train_data, dev_data, test_data)
        print(f"⏱️  RotatE training completed in {time.time() - stage_start:.1f}s")
    
    # Stage 4: R-GCN Training
    relation2idx = {}
    if config.run_rgcn_training:
        stage_start = time.time()
        rgcn_results = stage_rgcn_training(config, train_data, rotatE_embeds)
        if rgcn_results:
            rgcn_model, decoder, relation2idx = rgcn_results
            results['rgcn'] = {'model': rgcn_model, 'decoder': decoder, 'relation2idx': relation2idx}
        print(f"⏱️  R-GCN training completed in {time.time() - stage_start:.1f}s")
    
    # Stage 5: RL Training
    if config.run_rl_training:
        stage_start = time.time()
        rl_model = stage_rl_training(config, train_data, rotatE_embeds, relation2idx)
        if rl_model:
            results['rl'] = rl_model
        print(f"⏱️  RL training completed in {time.time() - stage_start:.1f}s")
    
    # Stage 6: Evaluation
    if config.run_evaluation:
        stage_start = time.time()
        eval_dir, eval_summary = stage_evaluation(config, exp_dir)
        results['evaluation'] = {'dir': eval_dir, 'summary': eval_summary}
        print(f"⏱️  Evaluation completed in {time.time() - stage_start:.1f}s")
    
    total_time = time.time() - start_time
    
    # Save final results
    results_summary = {
        'experiment_dir': exp_dir,
        'total_runtime_seconds': total_time,
        'stages_completed': {
            'eda': config.run_eda,
            'entity_embeddings': config.run_entity_embeddings,
            'rotate_training': config.run_rotate_training,
            'rgcn_training': config.run_rgcn_training,
            'rl_training': config.run_rl_training,
            'evaluation': config.run_evaluation
        },
        'config': vars(config)
    }
    
    with open(os.path.join(exp_dir, 'pipeline_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("\n🎉 THESIS PIPELINE COMPLETED!")
    print("="*60)
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Experiment directory: {exp_dir}")
    print(f"Results summary saved to: {exp_dir}/pipeline_results.json")
    
    return exp_dir, results

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Run complete thesis pipeline")
    parser.add_argument("--data_dir", default="data", help="DocRED data directory")
    parser.add_argument("--max_docs", type=int, default=500, help="Max training documents")
    parser.add_argument("--rgcn_epochs", type=int, default=6, help="R-GCN training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=50000, help="RL training timesteps")
    parser.add_argument("--eval_n_docs", type=int, default=25, help="Evaluation documents")
    
    # Stage selection
    parser.add_argument("--skip_eda", action="store_true", help="Skip EDA stage")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip_rotate", action="store_true", help="Skip RotatE training")
    parser.add_argument("--skip_rgcn", action="store_true", help="Skip R-GCN training")
    parser.add_argument("--skip_rl", action="store_true", help="Skip RL training")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced parameters")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ThesisPipelineConfig()
    config.data_dir = args.data_dir
    config.max_docs_train = args.max_docs
    config.rgcn_epochs = args.rgcn_epochs
    config.rl_timesteps = args.rl_timesteps
    config.eval_n_docs = args.eval_n_docs
    
    # Stage selection
    config.run_eda = not args.skip_eda
    config.run_entity_embeddings = not args.skip_embeddings
    config.run_rotate_training = not args.skip_rotate
    config.run_rgcn_training = not args.skip_rgcn
    config.run_rl_training = not args.skip_rl
    config.run_evaluation = not args.skip_evaluation
    
    # Quick mode adjustments
    if args.quick:
        config.max_docs_train = 100
        config.rgcn_epochs = 3
        config.rl_timesteps = 10000
        config.eval_n_docs = 10
        print("🏃 Quick mode enabled - using reduced parameters")
    
    # Run pipeline
    exp_dir, results = run_thesis_pipeline(config)
    
    return exp_dir, results

if __name__ == "__main__":
    exp_dir, results = main()
