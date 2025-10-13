"""
Comprehensive Evaluation Framework for Entity Resolution Approaches.
Compares Graph Embedding, Rule-Based, String Similarity, and Your Original Approach
with real F1 scores and detailed performance metrics.
"""

import time
import json
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import random
import numpy as np

# Import all approaches
from graph_embedding_resolver import GraphEmbeddingResolver
from rule_based_resolver import RuleBasedResolver  
from string_similarity_resolver import StringSimilarityResolver
from original_approach_wrapper import OriginalApproachWrapper  # Your original approach
from rl_kg_healer import RLKnowledgeGraphHealer  # RL-based approach


class GroundTruthGenerator:
    """Generates ground truth for entity resolution evaluation."""
    
    def __init__(self):
        """Initialize ground truth generator."""
        self.duplicate_rules = {
            # Organizations
            ('Apple Inc.', 'Apple'): True,
            ('Apple Inc.', 'Apple Corporation'): True,
            ('Apple', 'Apple Corporation'): True,
            ('International Business Machines', 'IBM'): True,
            ('Microsoft Corp.', 'Microsoft'): True,
            ('Google Inc.', 'Google'): True,
            ('Alphabet Inc.', 'Google'): False,  # Different entities
            
            # People  
            ('Tim Cook', 'Timothy Cook'): True,
            ('Tim Cook', 'T. Cook'): True,
            ('Bill Gates', 'William Gates'): True,
            ('Bill Gates', 'William H. Gates'): True,
            
            # Locations
            ('United States', 'USA'): True,
            ('United States', 'US'): True,
            ('United Kingdom', 'UK'): True,
            ('California', 'california'): True,  # Case difference
            ('New York City', 'NYC'): True,
            
            # Non-matches (to test false positives)
            ('Apple Inc.', 'Microsoft'): False,
            ('Tim Cook', 'Bill Gates'): False,
            ('California', 'Seattle'): False,
        }
    
    def create_ground_truth_dataset(self, num_docs: int = 20) -> Tuple[List[Dict], Dict]:
        """Create a dataset with known ground truth duplicates.
        
        Args:
            num_docs: Number of documents to generate
            
        Returns:
            Tuple of (documents, ground_truth_mapping)
        """
        # Define entity pools with known duplicates
        org_entities = [
            {'names': ['Apple Inc.', 'Apple', 'Apple Corporation'], 'type': 'ORG'},
            {'names': ['Microsoft Corp.', 'Microsoft'], 'type': 'ORG'},
            {'names': ['Google Inc.', 'Google'], 'type': 'ORG'}, 
            {'names': ['IBM', 'International Business Machines'], 'type': 'ORG'},
            {'names': ['Meta', 'Facebook'], 'type': 'ORG'},
            {'names': ['Tesla'], 'type': 'ORG'},  # Single name, no duplicates
            {'names': ['Amazon'], 'type': 'ORG'},
        ]
        
        per_entities = [
            {'names': ['Tim Cook', 'Timothy Cook', 'T. Cook'], 'type': 'PER'},
            {'names': ['Bill Gates', 'William Gates'], 'type': 'PER'},
            {'names': ['Elon Musk'], 'type': 'PER'},
            {'names': ['Jeff Bezos'], 'type': 'PER'},
            {'names': ['Satya Nadella'], 'type': 'PER'},
            {'names': ['Sundar Pichai'], 'type': 'PER'},
        ]
        
        loc_entities = [
            {'names': ['United States', 'USA', 'US'], 'type': 'LOC'},
            {'names': ['California', 'california'], 'type': 'LOC'},  # Case difference
            {'names': ['New York', 'NYC'], 'type': 'LOC'},
            {'names': ['Seattle'], 'type': 'LOC'},
            {'names': ['Cupertino'], 'type': 'LOC'},
            {'names': ['Mountain View'], 'type': 'LOC'},
        ]
        
        all_entity_groups = org_entities + per_entities + loc_entities
        
        # Generate documents
        docs = []
        global_entity_map = {}  # Maps (doc_idx, ent_idx) to canonical entity
        entity_counter = 0
        
        random.seed(42)  # For reproducibility
        
        for doc_idx in range(num_docs):
            # Randomly select 2-4 entity groups for this document
            selected_groups = random.sample(all_entity_groups, random.randint(2, 4))
            
            vertex_set = []
            labels = []
            
            for ent_idx, group in enumerate(selected_groups):
                # Randomly pick a name variation from the group
                name_variant = random.choice(group['names'])
                
                entity_mention = {
                    'name': name_variant,
                    'type': group['type'],
                    'sent_id': ent_idx
                }
                
                vertex_set.append([entity_mention])
                
                # Map to canonical entity (first name in group)
                canonical_name = group['names'][0]
                global_entity_map[(doc_idx, ent_idx)] = {
                    'canonical_name': canonical_name,
                    'type': group['type'],
                    'entity_id': f"{canonical_name}_{group['type']}"
                }
            
            # Add some random relations
            if len(vertex_set) >= 2:
                # Add 1-2 relations
                num_relations = min(2, len(vertex_set) - 1)
                for i in range(num_relations):
                    labels.append({
                        'h': i,
                        't': i + 1,
                        'r': 'P108',  # works_for relation
                    })
            
            doc = {
                'title': f'Document {doc_idx}',
                'vertexSet': vertex_set,
                'labels': labels,
                'sents': [['Sample', 'sentence', f'{doc_idx}', '.']]
            }
            
            docs.append(doc)
        
        return docs, global_entity_map
    
    def compute_ground_truth_pairs(self, global_entity_map: Dict) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Compute ground truth duplicate pairs.
        
        Args:
            global_entity_map: Mapping from (doc_idx, ent_idx) to canonical entity
            
        Returns:
            Set of ground truth duplicate pairs
        """
        # Group entities by canonical ID
        canonical_groups = defaultdict(list)
        
        for (doc_idx, ent_idx), entity_info in global_entity_map.items():
            canonical_id = entity_info['entity_id']
            canonical_groups[canonical_id].append((doc_idx, ent_idx))
        
        # Generate all pairs within each group
        ground_truth_pairs = set()
        
        for canonical_id, entity_positions in canonical_groups.items():
            if len(entity_positions) > 1:
                # All pairs within this group are duplicates
                for i in range(len(entity_positions)):
                    for j in range(i + 1, len(entity_positions)):
                        pos1, pos2 = entity_positions[i], entity_positions[j]
                        # Ensure consistent ordering
                        pair = tuple(sorted([pos1, pos2]))
                        ground_truth_pairs.add(pair)
        
        return ground_truth_pairs


class EntityResolutionEvaluator:
    """Evaluator for entity resolution approaches."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.ground_truth_gen = GroundTruthGenerator()
    
    def extract_predicted_pairs(self, results: Dict, docs: List[Dict]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Extract predicted duplicate pairs from results.
        
        Args:
            results: Results from entity resolution method
            docs: Original documents
            
        Returns:
            Set of predicted duplicate pairs
        """
        predicted_pairs = set()
        
        for duplicate in results['duplicates']:
            entity1 = duplicate['entity1']
            entity2 = duplicate['entity2']
            
            # Extract positions
            pos1 = (entity1.get('doc_idx', -1), entity1.get('ent_idx', -1))
            pos2 = (entity2.get('doc_idx', -1), entity2.get('ent_idx', -1))
            
            if pos1[0] != -1 and pos1[1] != -1 and pos2[0] != -1 and pos2[1] != -1:
                # Ensure consistent ordering
                pair = tuple(sorted([pos1, pos2]))
                predicted_pairs.add(pair)
        
        return predicted_pairs
    
    def compute_metrics(self, predicted_pairs: Set, ground_truth_pairs: Set) -> Dict[str, float]:
        """Compute precision, recall, and F1 score.
        
        Args:
            predicted_pairs: Set of predicted duplicate pairs
            ground_truth_pairs: Set of ground truth duplicate pairs
            
        Returns:
            Dictionary with precision, recall, F1, and other metrics
        """
        # True positives: predictions that match ground truth
        true_positives = len(predicted_pairs & ground_truth_pairs)
        
        # False positives: predictions not in ground truth
        false_positives = len(predicted_pairs - ground_truth_pairs)
        
        # False negatives: ground truth pairs not predicted
        false_negatives = len(ground_truth_pairs - predicted_pairs)
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_predicted': len(predicted_pairs),
            'total_ground_truth': len(ground_truth_pairs)
        }
    
    def evaluate_approach(self, resolver, docs: List[Dict], ground_truth_pairs: Set, 
                         approach_name: str) -> Dict[str, Any]:
        """Evaluate a single entity resolution approach.
        
        Args:
            resolver: Entity resolution method
            docs: Documents to process
            ground_truth_pairs: Ground truth duplicate pairs
            approach_name: Name of the approach
            
        Returns:
            Evaluation results
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {approach_name}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            # Run the approach - all approaches now have find_duplicate_entities
            results = resolver.find_duplicate_entities(docs)
            
            processing_time = time.time() - start_time
            
            # Extract predicted pairs
            predicted_pairs = self.extract_predicted_pairs(results, docs)
            
            # Compute metrics
            metrics = self.compute_metrics(predicted_pairs, ground_truth_pairs)
            
            # Add additional info
            metrics['approach'] = approach_name
            metrics['processing_time'] = processing_time
            metrics['approach_specific_stats'] = results.get('statistics', {})
            
            print(f"Results for {approach_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
            print(f"  Processing Time: {metrics['processing_time']:.3f}s")
            print(f"  True Positives: {metrics['true_positives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {approach_name}: {e}")
            return {
                'approach': approach_name,
                'error': str(e),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of all approaches.
        
        Returns:
            Complete evaluation results
        """
        print("=" * 80)
        print("COMPREHENSIVE ENTITY RESOLUTION EVALUATION")
        print("=" * 80)
        
        # Generate ground truth dataset
        print("\nGenerating ground truth dataset...")
        docs, ground_truth_map = self.ground_truth_gen.create_ground_truth_dataset(num_docs=15)
        ground_truth_pairs = self.ground_truth_gen.compute_ground_truth_pairs(ground_truth_map)
        
        print(f"Created dataset with {len(docs)} documents")
        print(f"Total entities: {sum(len(doc['vertexSet']) for doc in docs)}")
        print(f"Ground truth duplicate pairs: {len(ground_truth_pairs)}")
        
        # Initialize all approaches
        approaches = {
            'String Similarity (Baseline)': StringSimilarityResolver(threshold=0.7),
            'Rule-Based': RuleBasedResolver(),
            'Graph Embedding': GraphEmbeddingResolver(embedding_dim=64, num_walks=5, walk_length=20),  # Smaller for speed
            'Your Original Approach': OriginalApproachWrapper(similarity_threshold=0.7),
            'RL-Based KG Healer': RLKnowledgeGraphHealer(embedding_dim=128, max_candidates=10)
        }
        
        # Evaluate each approach
        results = {}
        
        for approach_name, resolver in approaches.items():
            results[approach_name] = self.evaluate_approach(
                resolver, docs, ground_truth_pairs, approach_name
            )
        
        # Generate comparison report
        print(f"\n{'='*80}")
        print("FINAL COMPARISON RESULTS")
        print('='*80)
        
        # Sort by F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        print(f"{'Approach':<30} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Time(s)':<10}")
        print("-" * 70)
        
        for approach_name, metrics in sorted_results:
            if 'error' not in metrics:
                print(f"{approach_name:<30} "
                      f"{metrics['precision']:<10.3f} "
                      f"{metrics['recall']:<8.3f} "
                      f"{metrics['f1_score']:<8.3f} "
                      f"{metrics['processing_time']:<10.3f}")
            else:
                print(f"{approach_name:<30} ERROR: {metrics['error']}")
        
        # Find best approach
        best_approach = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
        print(f"\n🏆 BEST APPROACH: {best_approach[0]} (F1: {best_approach[1].get('f1_score', 0):.3f})")
        
        # Additional analysis
        print(f"\n{'='*40}")
        print("DETAILED ANALYSIS")
        print('='*40)
        
        for approach_name, metrics in sorted_results:
            if 'error' not in metrics:
                print(f"\n{approach_name}:")
                print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
                print(f"  Predicted {metrics['total_predicted']} pairs out of {metrics['total_ground_truth']} actual duplicates")
        
        # Save results
        evaluation_summary = {
            'dataset_info': {
                'num_documents': len(docs),
                'total_entities': sum(len(doc['vertexSet']) for doc in docs),
                'ground_truth_pairs': len(ground_truth_pairs)
            },
            'approach_results': results,
            'best_approach': {
                'name': best_approach[0],
                'f1_score': best_approach[1].get('f1_score', 0),
                'precision': best_approach[1].get('precision', 0),
                'recall': best_approach[1].get('recall', 0)
            }
        }
        
        return evaluation_summary


if __name__ == "__main__":
    # Run the comprehensive evaluation
    evaluator = EntityResolutionEvaluator()
    
    try:
        results = evaluator.run_comprehensive_evaluation()
        
        # Save results to file
        with open('/home/serah/Downloads/kg-healing-docred/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Evaluation completed! Results saved to evaluation_results.json")
        print(f"\n📊 Summary:")
        print(f"   Best Approach: {results['best_approach']['name']}")
        print(f"   Best F1 Score: {results['best_approach']['f1_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
