#!/usr/bin/env python3
"""
Training script for RL-based Knowledge Graph Healing.
This script properly trains the PPO agent on the dataset.
"""

import time
import numpy as np
from typing import List, Dict, Any
from rl_kg_healer import RLKnowledgeGraphHealer
from rl_kg_environment import KGHealingEnv
from comprehensive_evaluation import GroundTruthGenerator, EntityResolutionEvaluator

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("stable-baselines3 not available for proper RL training")


class RLTrainer:
    """Trainer for RL-based KG healing approach."""
    
    def __init__(self, total_timesteps: int = 10000):
        """Initialize RL trainer.
        
        Args:
            total_timesteps: Number of timesteps to train
        """
        self.total_timesteps = total_timesteps
        self.model = None
        self.trained = False
        
    def create_training_dataset(self, num_docs: int = 50) -> List[Dict]:
        """Create training dataset for RL agent.
        
        Args:
            num_docs: Number of documents to generate
            
        Returns:
            Training dataset
        """
        print(f"Generating training dataset with {num_docs} documents...")
        gt_gen = GroundTruthGenerator()
        docs, _ = gt_gen.create_ground_truth_dataset(num_docs=num_docs)
        
        # Convert to format expected by RL environment
        dataset = []
        for i, doc in enumerate(docs):
            dataset.append((i, doc))
        
        return dataset
    
    def train_rl_agent(self, training_dataset: List[Dict]) -> PPO:
        """Train PPO agent on KG healing task.
        
        Args:
            training_dataset: Dataset for training
            
        Returns:
            Trained PPO model
        """
        if not STABLE_BASELINES_AVAILABLE:
            print("Cannot train RL agent - stable-baselines3 not available")
            return None
        
        print("Setting up RL training environment...")
        
        # Create environment
        env = KGHealingEnv(
            dataset=training_dataset,
            max_candidates=8,
            max_steps=5,
            embedding_dim=64
        )
        
        # Wrap in vectorized environment
        env = DummyVecEnv([lambda: env])
        
        # Create PPO model
        print("Initializing PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./rl_kg_healing_tensorboard/"
        )
        
        # Train the model
        print(f"Starting RL training for {self.total_timesteps} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"RL training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        model.save("rl_kg_healer_model")
        print("Trained model saved as 'rl_kg_healer_model'")
        
        return model
    
    def evaluate_trained_model(self, model: PPO, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Evaluate trained RL model on test dataset.
        
        Args:
            model: Trained PPO model
            test_dataset: Test dataset
            
        Returns:
            Evaluation results
        """
        print("Evaluating trained RL model...")
        
        if not STABLE_BASELINES_AVAILABLE or model is None:
            print("Cannot evaluate - no trained model available")
            return {'error': 'No trained model available'}
        
        # Create RL healer with trained model
        rl_healer = RLKnowledgeGraphHealer()
        rl_healer.model = model
        rl_healer.trained = True
        
        # Convert dataset format
        docs = [item[1] if isinstance(item, tuple) else item for item in test_dataset]
        
        # Apply RL-based entity resolution
        start_time = time.time()
        results = rl_healer.find_duplicate_entities(docs)
        processing_time = time.time() - start_time
        
        results['processing_time'] = processing_time
        results['model_trained'] = True
        
        print(f"RL evaluation completed in {processing_time:.3f} seconds")
        print(f"Found {len(results.get('duplicates', []))} duplicate pairs")
        
        return results


def run_rl_training_and_evaluation():
    """Main function to run complete RL training and evaluation."""
    print("=" * 80)
    print("RL-BASED KNOWLEDGE GRAPH HEALING - TRAINING AND EVALUATION")
    print("=" * 80)
    
    if not STABLE_BASELINES_AVAILABLE:
        print("❌ Cannot run RL training - stable-baselines3 not installed properly")
        print("Please install: pip install stable-baselines3")
        return None
    
    # Initialize trainer
    trainer = RLTrainer(total_timesteps=5000)  # Reduced for faster training
    
    # Create datasets
    print("\n1. Creating training and test datasets...")
    training_data = trainer.create_training_dataset(num_docs=30)  # Training set
    test_data = trainer.create_training_dataset(num_docs=15)      # Test set
    
    print(f"Training dataset: {len(training_data)} documents")
    print(f"Test dataset: {len(test_data)} documents")
    
    # Train RL agent
    print("\n2. Training RL agent...")
    trained_model = trainer.train_rl_agent(training_data)
    
    if trained_model is None:
        print("❌ RL training failed")
        return None
    
    # Evaluate trained model
    print("\n3. Evaluating trained RL model...")
    rl_results = trainer.evaluate_trained_model(trained_model, test_data)
    
    # Compare with other approaches using the same test data
    print("\n4. Comparing with other approaches...")
    evaluator = EntityResolutionEvaluator()
    
    # Convert test data to the format expected by evaluator
    test_docs = [item[1] if isinstance(item, tuple) else item for item in test_data]
    gt_gen = GroundTruthGenerator()
    
    # Create ground truth for test data
    entity_map = {}
    for doc_idx, doc in enumerate(test_docs):
        for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
            if entity_cluster:
                entity_name = entity_cluster[0]['name']
                entity_type = entity_cluster[0].get('type', 'UNK')
                # Create a simple mapping for evaluation
                entity_map[(doc_idx, ent_idx)] = {
                    'canonical_name': entity_name,
                    'type': entity_type,
                    'entity_id': f"{entity_name}_{entity_type}"
                }
    
    ground_truth_pairs = gt_gen.compute_ground_truth_pairs(entity_map)
    
    # Extract predicted pairs from RL results
    predicted_pairs = evaluator.extract_predicted_pairs(rl_results, test_docs)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predicted_pairs, ground_truth_pairs)
    
    print("\n" + "=" * 60)
    print("TRAINED RL MODEL RESULTS")
    print("=" * 60)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Processing Time: {rl_results.get('processing_time', 0):.3f}s")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Duplicates Found: {len(rl_results.get('duplicates', []))}")
    print(f"Ground Truth Pairs: {len(ground_truth_pairs)}")
    
    return {
        'trained_model': trained_model,
        'rl_results': rl_results,
        'metrics': metrics,
        'training_completed': True
    }


if __name__ == "__main__":
    # Run the complete RL training and evaluation
    results = run_rl_training_and_evaluation()
    
    if results and results.get('training_completed'):
        print(f"\n🎉 RL training and evaluation completed successfully!")
        print(f"📊 Final F1 Score: {results['metrics']['f1_score']:.3f}")
        print(f"⚡ Processing Time: {results['rl_results'].get('processing_time', 0):.3f}s")
    else:
        print("❌ RL training failed or was incomplete")
