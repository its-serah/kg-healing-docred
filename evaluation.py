"""
Evaluation module for knowledge graph healing.
Provides metrics and evaluation tools for assessing healing performance.
"""

import json
import time
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, Counter
import networkx as nx


class KGHealingEvaluator:
    """Evaluator for knowledge graph healing performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def evaluate_entity_resolution(self, 
                                   original_docs: List[Dict], 
                                   resolved_docs: List[Dict],
                                   ground_truth: Dict = None) -> Dict[str, Any]:
        """Evaluate entity resolution performance.
        
        Args:
            original_docs: Documents before entity resolution
            resolved_docs: Documents after entity resolution
            ground_truth: Optional ground truth for entity clusters
            
        Returns:
            Entity resolution evaluation metrics
        """
        metrics = {
            'entity_merges_detected': 0,
            'entity_merges_applied': 0,
            'duplicate_surface_forms': 0,
            'merge_confidence_distribution': [],
            'processing_statistics': {}
        }
        
        # Count merges and analyze confidence
        total_merges = 0
        confidence_scores = []
        
        for resolved_doc in resolved_docs:
            entity_resolutions = resolved_doc.get('entity_resolutions', [])
            for resolution in entity_resolutions:
                if 'resolution' in resolution:
                    res_data = resolution['resolution']
                    original_mentions = len(res_data.get('original_mentions', []))
                    n_clusters = res_data.get('n_clusters', original_mentions)
                    
                    if n_clusters < original_mentions:
                        merges = original_mentions - n_clusters
                        total_merges += merges
                        confidence_scores.append(res_data.get('merge_confidence', 0.0))
        
        metrics['entity_merges_applied'] = total_merges
        metrics['merge_confidence_distribution'] = {
            'mean': sum(confidence_scores) / max(1, len(confidence_scores)),
            'min': min(confidence_scores) if confidence_scores else 0,
            'max': max(confidence_scores) if confidence_scores else 0,
            'count': len(confidence_scores)
        }
        
        # If ground truth is available, compute precision/recall
        if ground_truth:
            precision, recall, f1 = self._compute_entity_resolution_prf(
                resolved_docs, ground_truth
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
        
        return metrics
    
    def evaluate_relation_completion(self,
                                     original_docs: List[Dict],
                                     completed_docs: List[Dict],
                                     ground_truth: Dict = None) -> Dict[str, Any]:
        """Evaluate relation completion performance.
        
        Args:
            original_docs: Documents before relation completion
            completed_docs: Documents after relation completion
            ground_truth: Optional ground truth for missing relations
            
        Returns:
            Relation completion evaluation metrics
        """
        metrics = {
            'relations_added': 0,
            'relations_by_method': defaultdict(int),
            'confidence_distribution': [],
            'relation_types_added': Counter(),
            'completion_coverage': 0.0
        }
        
        total_added = 0
        confidence_scores = []
        
        # Compare original and completed documents
        for orig_doc, comp_doc in zip(original_docs, completed_docs):
            orig_relations = set()
            for rel in orig_doc.get('labels', []):
                orig_relations.add((rel['h'], rel['t'], rel['r']))
            
            comp_relations = set()
            for rel in comp_doc.get('labels', []):
                rel_tuple = (rel['h'], rel['t'], rel['r'])
                comp_relations.add(rel_tuple)
                
                # Check if this is a newly added relation
                if (rel_tuple not in orig_relations and 
                    'completion_method' in rel):
                    total_added += 1
                    metrics['relation_types_added'][rel['r']] += 1
                    metrics['relations_by_method'][rel['completion_method']] += 1
                    
                    if 'confidence' in rel:
                        confidence_scores.append(rel['confidence'])
        
        metrics['relations_added'] = total_added
        metrics['confidence_distribution'] = {
            'mean': sum(confidence_scores) / max(1, len(confidence_scores)),
            'min': min(confidence_scores) if confidence_scores else 0,
            'max': max(confidence_scores) if confidence_scores else 0,
            'count': len(confidence_scores)
        }
        
        # Compute coverage (fraction of documents that got new relations)
        docs_with_additions = sum(1 for doc in completed_docs 
                                 if doc.get('completion_metadata', {}).get('added_relations', 0) > 0)
        metrics['completion_coverage'] = docs_with_additions / max(1, len(completed_docs))
        
        # If ground truth is available, compute precision/recall
        if ground_truth:
            precision, recall, f1 = self._compute_relation_completion_prf(
                completed_docs, ground_truth
            )
            metrics['precision'] = precision
            metrics['recall'] = recall  
            metrics['f1_score'] = f1
        
        return metrics
    
    def evaluate_overall_healing(self,
                                 original_docs: List[Dict],
                                 healed_docs: List[Dict],
                                 healing_stats: Dict) -> Dict[str, Any]:
        """Evaluate overall healing performance.
        
        Args:
            original_docs: Documents before healing
            healed_docs: Documents after healing
            healing_stats: Statistics from the healing process
            
        Returns:
            Overall healing evaluation metrics
        """
        metrics = {
            'knowledge_graph_quality': {},
            'completeness_improvement': {},
            'consistency_improvement': {},
            'efficiency_metrics': {},
            'overall_score': 0.0
        }
        
        # Knowledge graph quality metrics
        orig_quality = self._compute_kg_quality_metrics(original_docs)
        healed_quality = self._compute_kg_quality_metrics(healed_docs)
        
        metrics['knowledge_graph_quality'] = {
            'original': orig_quality,
            'healed': healed_quality,
            'improvement': {
                'density': healed_quality['density'] - orig_quality['density'],
                'connectivity': healed_quality['connectivity'] - orig_quality['connectivity'],
                'entity_coverage': healed_quality['entity_coverage'] - orig_quality['entity_coverage']
            }
        }
        
        # Completeness improvement
        orig_relations = sum(len(doc.get('labels', [])) for doc in original_docs)
        healed_relations = sum(len(doc.get('labels', [])) for doc in healed_docs)
        
        metrics['completeness_improvement'] = {
            'original_relations': orig_relations,
            'healed_relations': healed_relations,
            'relations_added': healed_relations - orig_relations,
            'improvement_percentage': ((healed_relations - orig_relations) / max(1, orig_relations)) * 100
        }
        
        # Efficiency metrics
        metrics['efficiency_metrics'] = {
            'processing_time': healing_stats.get('processing_time', 0),
            'documents_per_second': len(original_docs) / max(0.001, healing_stats.get('processing_time', 0.001)),
            'relations_per_second': healed_relations / max(0.001, healing_stats.get('processing_time', 0.001))
        }
        
        # Overall score (weighted combination of improvements)
        quality_score = (
            0.3 * metrics['knowledge_graph_quality']['improvement']['density'] +
            0.3 * metrics['knowledge_graph_quality']['improvement']['connectivity'] +
            0.4 * (metrics['completeness_improvement']['improvement_percentage'] / 100)
        )
        metrics['overall_score'] = max(0, min(1, quality_score))
        
        return metrics
    
    def _compute_kg_quality_metrics(self, docs: List[Dict]) -> Dict[str, float]:
        """Compute knowledge graph quality metrics for documents.
        
        Args:
            docs: List of documents
            
        Returns:
            Quality metrics dictionary
        """
        total_entities = 0
        total_relations = 0
        total_connected_components = 0
        total_density = 0
        
        for doc in docs:
            entities = doc.get('vertexSet', [])
            relations = doc.get('labels', [])
            
            n_entities = len(entities)
            n_relations = len(relations)
            
            total_entities += n_entities
            total_relations += n_relations
            
            if n_entities > 1:
                # Build graph to compute connectivity metrics
                graph = nx.Graph()
                graph.add_nodes_from(range(n_entities))
                
                for rel in relations:
                    graph.add_edge(rel['h'], rel['t'])
                
                # Density = actual edges / possible edges
                max_edges = n_entities * (n_entities - 1) / 2
                density = n_relations / max_edges if max_edges > 0 else 0
                total_density += density
                
                # Connected components
                total_connected_components += nx.number_connected_components(graph)
        
        n_docs = max(1, len(docs))
        
        return {
            'density': total_density / n_docs,
            'connectivity': 1 - (total_connected_components / max(1, total_entities)),
            'entity_coverage': total_entities / n_docs,
            'relation_coverage': total_relations / n_docs
        }
    
    def _compute_entity_resolution_prf(self, 
                                       resolved_docs: List[Dict],
                                       ground_truth: Dict) -> Tuple[float, float, float]:
        """Compute precision, recall, F1 for entity resolution.
        
        Args:
            resolved_docs: Documents with entity resolution results
            ground_truth: Ground truth entity clusters
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        # Placeholder implementation
        # In practice, would compare predicted entity clusters with ground truth
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # This would need actual ground truth comparison logic
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(0.001, precision + recall)
        
        return precision, recall, f1
    
    def _compute_relation_completion_prf(self,
                                         completed_docs: List[Dict],
                                         ground_truth: Dict) -> Tuple[float, float, float]:
        """Compute precision, recall, F1 for relation completion.
        
        Args:
            completed_docs: Documents with relation completion results
            ground_truth: Ground truth for missing relations
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        # Placeholder implementation
        # In practice, would compare added relations with ground truth
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(0.001, precision + recall)
        
        return precision, recall, f1
    
    def generate_evaluation_report(self,
                                   original_docs: List[Dict],
                                   healed_docs: List[Dict], 
                                   healing_stats: Dict) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            original_docs: Documents before healing
            healed_docs: Documents after healing
            healing_stats: Healing statistics
            
        Returns:
            Formatted evaluation report
        """
        # Run all evaluations
        entity_eval = self.evaluate_entity_resolution(original_docs, healed_docs)
        relation_eval = self.evaluate_relation_completion(original_docs, healed_docs)
        overall_eval = self.evaluate_overall_healing(original_docs, healed_docs, healing_stats)
        
        report = []
        report.append("=" * 60)
        report.append("KNOWLEDGE GRAPH HEALING EVALUATION REPORT")
        report.append("=" * 60)
        
        # Entity Resolution Evaluation
        report.append("\nENTITY RESOLUTION EVALUATION:")
        report.append(f"  Merges applied: {entity_eval.get('entity_merges_applied', 0)}")
        conf_dist = entity_eval.get('merge_confidence_distribution', {})
        report.append(f"  Average merge confidence: {conf_dist.get('mean', 0):.3f}")
        report.append(f"  Confidence range: [{conf_dist.get('min', 0):.3f}, {conf_dist.get('max', 0):.3f}]")
        
        if 'precision' in entity_eval:
            report.append(f"  Precision: {entity_eval['precision']:.3f}")
            report.append(f"  Recall: {entity_eval['recall']:.3f}")
            report.append(f"  F1 Score: {entity_eval['f1_score']:.3f}")
        
        # Relation Completion Evaluation
        report.append("\nRELATION COMPLETION EVALUATION:")
        report.append(f"  Relations added: {relation_eval.get('relations_added', 0)}")
        conf_dist = relation_eval.get('confidence_distribution', {})
        report.append(f"  Average completion confidence: {conf_dist.get('mean', 0):.3f}")
        report.append(f"  Completion coverage: {relation_eval.get('completion_coverage', 0):.1%}")
        
        methods = relation_eval.get('relations_by_method', {})
        if methods:
            report.append("  Relations by method:")
            for method, count in methods.items():
                report.append(f"    {method}: {count}")
        
        if 'precision' in relation_eval:
            report.append(f"  Precision: {relation_eval['precision']:.3f}")
            report.append(f"  Recall: {relation_eval['recall']:.3f}")
            report.append(f"  F1 Score: {relation_eval['f1_score']:.3f}")
        
        # Overall Evaluation
        report.append("\nOVERALL HEALING EVALUATION:")
        completeness = overall_eval.get('completeness_improvement', {})
        report.append(f"  Relations improvement: {completeness.get('improvement_percentage', 0):.1f}%")
        
        quality = overall_eval.get('knowledge_graph_quality', {})
        if 'improvement' in quality:
            improvements = quality['improvement']
            report.append(f"  Graph density improvement: {improvements.get('density', 0):.3f}")
            report.append(f"  Graph connectivity improvement: {improvements.get('connectivity', 0):.3f}")
        
        efficiency = overall_eval.get('efficiency_metrics', {})
        report.append(f"  Processing speed: {efficiency.get('documents_per_second', 0):.1f} docs/sec")
        report.append(f"  Overall quality score: {overall_eval.get('overall_score', 0):.3f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class BenchmarkRunner:
    """Runner for benchmarking KG healing algorithms."""
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.evaluator = KGHealingEvaluator()
        
    def run_benchmark(self,
                      test_datasets: List[Tuple[str, List[Dict]]],
                      healing_configs: List[Dict],
                      output_file: str = None) -> Dict[str, Any]:
        """Run benchmark across multiple datasets and configurations.
        
        Args:
            test_datasets: List of (dataset_name, documents) tuples
            healing_configs: List of healing configuration dictionaries
            output_file: Optional file to save results
            
        Returns:
            Benchmark results dictionary
        """
        results = {
            'benchmark_info': {
                'datasets': [name for name, _ in test_datasets],
                'configurations': len(healing_configs),
                'start_time': time.time()
            },
            'dataset_results': {},
            'config_comparison': {},
            'summary': {}
        }
        
        # Import here to avoid circular imports
        from kg_healer import KGHealer
        
        for dataset_name, docs in test_datasets:
            print(f"Running benchmark on {dataset_name} ({len(docs)} documents)...")
            
            dataset_results = {}
            
            for config_idx, config in enumerate(healing_configs):
                config_name = config.get('name', f'config_{config_idx}')
                print(f"  Testing configuration: {config_name}")
                
                # Initialize healer with configuration
                healer = KGHealer(**{k: v for k, v in config.items() if k != 'name'})
                
                # Apply healing
                start_time = time.time()
                healed_docs, healing_stats = healer.heal_documents(docs.copy())
                healing_time = time.time() - start_time
                
                # Evaluate results
                evaluation = self.evaluator.evaluate_overall_healing(docs, healed_docs, healing_stats)
                evaluation['healing_time'] = healing_time
                
                dataset_results[config_name] = {
                    'healing_stats': healing_stats,
                    'evaluation': evaluation
                }
            
            results['dataset_results'][dataset_name] = dataset_results
        
        # Generate comparison and summary
        results['config_comparison'] = self._compare_configurations(results['dataset_results'])
        results['summary'] = self._generate_benchmark_summary(results)
        results['benchmark_info']['end_time'] = time.time()
        results['benchmark_info']['total_time'] = (
            results['benchmark_info']['end_time'] - results['benchmark_info']['start_time']
        )
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Benchmark results saved to {output_file}")
        
        return results
    
    def _compare_configurations(self, dataset_results: Dict) -> Dict[str, Any]:
        """Compare performance across configurations."""
        comparison = {
            'best_overall_score': {},
            'best_processing_time': {},
            'best_relations_added': {}
        }
        
        for dataset_name, results in dataset_results.items():
            # Find best configurations for different metrics
            best_score = ('', 0)
            best_time = ('', float('inf'))
            best_relations = ('', 0)
            
            for config_name, config_results in results.items():
                evaluation = config_results['evaluation']
                
                # Overall score
                score = evaluation.get('overall_score', 0)
                if score > best_score[1]:
                    best_score = (config_name, score)
                
                # Processing time
                time_taken = evaluation.get('healing_time', float('inf'))
                if time_taken < best_time[1]:
                    best_time = (config_name, time_taken)
                
                # Relations added
                relations = evaluation.get('completeness_improvement', {}).get('relations_added', 0)
                if relations > best_relations[1]:
                    best_relations = (config_name, relations)
            
            comparison['best_overall_score'][dataset_name] = best_score
            comparison['best_processing_time'][dataset_name] = best_time
            comparison['best_relations_added'][dataset_name] = best_relations
        
        return comparison
    
    def _generate_benchmark_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            'total_datasets': len(results['dataset_results']),
            'total_configurations': 0,
            'average_improvement': 0,
            'average_processing_time': 0
        }
        
        all_improvements = []
        all_times = []
        
        for dataset_results in results['dataset_results'].values():
            summary['total_configurations'] = len(dataset_results)
            
            for config_results in dataset_results.values():
                evaluation = config_results['evaluation']
                
                improvement = evaluation.get('completeness_improvement', {}).get('improvement_percentage', 0)
                all_improvements.append(improvement)
                
                time_taken = evaluation.get('healing_time', 0)
                all_times.append(time_taken)
        
        if all_improvements:
            summary['average_improvement'] = sum(all_improvements) / len(all_improvements)
        
        if all_times:
            summary['average_processing_time'] = sum(all_times) / len(all_times)
        
        return summary


if __name__ == "__main__":
    # Example usage
    evaluator = KGHealingEvaluator()
    
    # Create sample data for testing
    original_docs = [
        {
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG'}],
                [{'name': 'Tim Cook', 'type': 'PER'}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'}  # Tim works at Apple
            ]
        }
    ]
    
    healed_docs = [
        {
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG'}],
                [{'name': 'Tim Cook', 'type': 'PER'}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},  # Original relation
                {'h': 0, 't': 1, 'r': 'P169', 'completion_method': 'transitivity', 'confidence': 0.8}  # Added
            ]
        }
    ]
    
    healing_stats = {'processing_time': 1.5}
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(original_docs, healed_docs, healing_stats)
    print(report)
