"""
Main demonstration script for Knowledge Graph Healing on DocRED.
Shows the complete healing pipeline with sample data and evaluation.
"""

import argparse
import sys
from typing import List, Dict
import json

from data_loader import DocREDLoader
from entity_resolution import EntityResolver
from relation_completion import RelationCompleter
from kg_healer import KGHealer
from evaluation import KGHealingEvaluator, BenchmarkRunner
from utils import create_sample_document, compute_document_statistics

# Import new entity resolution approaches
from string_similarity_resolver import StringSimilarityResolver
from rule_based_resolver import RuleBasedResolver
from graph_embedding_resolver import GraphEmbeddingResolver
from rl_kg_healer import RLKnowledgeGraphHealer
from comprehensive_evaluation import EntityResolutionEvaluator


def create_demo_dataset() -> List[Dict]:
    """Create a demonstration dataset with various KG healing opportunities.
    
    Returns:
        List of sample documents
    """
    demo_docs = [
        # Document 1: Basic entities and relations
        {
            'title': 'Apple Inc. and Tim Cook',
            'vertexSet': [
                [{'name': 'Apple Inc.', 'type': 'ORG', 'sent_id': 0}],
                [{'name': 'Tim Cook', 'type': 'PER', 'sent_id': 0}],
                [{'name': 'California', 'type': 'LOC', 'sent_id': 1}],
                [{'name': 'United States', 'type': 'LOC', 'sent_id': 1}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},  # Tim works at Apple
                {'h': 0, 't': 2, 'r': 'P131'},  # Apple located in California  
                {'h': 2, 't': 3, 'r': 'P131'},  # California located in US
            ]
        },
        
        # Document 2: Potential duplicates and missing relations
        {
            'title': 'Apple Company and CEO',
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG', 'sent_id': 0}],        # Duplicate of Apple Inc.
                [{'name': 'Timothy Cook', 'type': 'PER', 'sent_id': 0}], # Duplicate of Tim Cook
                [{'name': 'iPhone', 'type': 'MISC', 'sent_id': 1}],
                [{'name': 'Cupertino', 'type': 'LOC', 'sent_id': 1}]
            ],
            'labels': [
                {'h': 0, 't': 2, 'r': 'P176'},  # Apple manufacturer of iPhone
                {'h': 0, 't': 3, 'r': 'P159'},  # Apple headquarters in Cupertino
                # Missing: Timothy Cook works at Apple (should be inferred)
            ]
        },
        
        # Document 3: Complex relations with chains
        {
            'title': 'Tech Companies and Locations',
            'vertexSet': [
                [{'name': 'Google', 'type': 'ORG', 'sent_id': 0}],
                [{'name': 'Sundar Pichai', 'type': 'PER', 'sent_id': 0}],
                [{'name': 'Mountain View', 'type': 'LOC', 'sent_id': 1}],
                [{'name': 'California', 'type': 'LOC', 'sent_id': 1}],  # Duplicate location
                [{'name': 'Alphabet Inc.', 'type': 'ORG', 'sent_id': 2}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},  # Sundar works at Google
                {'h': 0, 't': 2, 'r': 'P159'},  # Google HQ in Mountain View
                {'h': 2, 't': 3, 'r': 'P131'},  # Mountain View in California
                {'h': 0, 't': 4, 'r': 'P749'},  # Google subsidiary of Alphabet
                # Missing relations can be inferred through transitivity
            ]
        },
        
        # Document 4: Family and personal relations
        {
            'title': 'Musk Family',
            'vertexSet': [
                [{'name': 'Elon Musk', 'type': 'PER', 'sent_id': 0}],
                [{'name': 'Tesla', 'type': 'ORG', 'sent_id': 0}],
                [{'name': 'SpaceX', 'type': 'ORG', 'sent_id': 1}],
                [{'name': 'South Africa', 'type': 'LOC', 'sent_id': 2}]
            ],
            'labels': [
                {'h': 0, 't': 1, 'r': 'P108'},  # Elon works at Tesla
                {'h': 0, 't': 2, 'r': 'P108'},  # Elon works at SpaceX
                {'h': 0, 't': 3, 'r': 'P19'},   # Elon born in South Africa
            ]
        },
        
        # Document 5: Minimal document for testing edge cases
        {
            'title': 'Minimal Example',
            'vertexSet': [
                [{'name': 'Microsoft', 'type': 'ORG', 'sent_id': 0}],
                [{'name': 'Seattle', 'type': 'LOC', 'sent_id': 0}]
            ],
            'labels': [
                {'h': 0, 't': 1, 'r': 'P131'},  # Microsoft located in Seattle
            ]
        }
    ]
    
    return demo_docs


def run_basic_demo():
    """Run a basic demonstration of the KG healing pipeline."""
    print("=" * 60)
    print("KNOWLEDGE GRAPH HEALING DEMO")
    print("=" * 60)
    
    # Create demo dataset
    print("\n1. Creating demonstration dataset...")
    docs = create_demo_dataset()
    print(f"Created {len(docs)} sample documents")
    
    # Show initial statistics
    print("\n2. Initial dataset statistics:")
    initial_stats = compute_document_statistics(docs)
    print(f"  Total entities: {initial_stats['total_entities']}")
    print(f"  Total relations: {initial_stats['total_relations']}")
    print(f"  Entity types: {list(initial_stats['entity_types'].keys())}")
    print(f"  Relation types: {list(initial_stats['relation_types'].keys())}")
    
    # Initialize KG healer
    print("\n3. Initializing KG healer...")
    healer = KGHealer(
        entity_similarity_threshold=0.7,
        relation_confidence_threshold=0.5,
        verbose=True
    )
    
    # Apply healing
    print("\n4. Applying KG healing...")
    healed_docs, healing_stats = healer.heal_documents(docs)
    
    # Generate healing report
    print("\n5. Healing Results:")
    report = healer.generate_healing_report(healing_stats)
    print(report)
    
    # Evaluation
    print("\n6. Evaluation:")
    evaluator = KGHealingEvaluator()
    eval_report = evaluator.generate_evaluation_report(docs, healed_docs, healing_stats)
    print(eval_report)
    
    return healed_docs, healing_stats


def run_component_demo():
    """Demonstrate individual components of the healing pipeline."""
    print("=" * 60)
    print("COMPONENT-WISE DEMONSTRATION")
    print("=" * 60)
    
    docs = create_demo_dataset()
    
    # Entity Resolution Demo
    print("\n1. ENTITY RESOLUTION DEMO:")
    print("-" * 40)
    
    entity_resolver = EntityResolver(similarity_threshold=0.7)
    
    print("Finding duplicate entities...")
    duplicates = entity_resolver.find_duplicate_entities(docs)
    print(f"Found {len(duplicates)} potential duplicate surface forms:")
    for surface_form, mentions in list(duplicates.items())[:3]:  # Show first 3
        print(f"  '{surface_form}': {len(mentions)} mentions across documents")
    
    print("\nApplying entity resolution...")
    resolved_docs, entity_stats = entity_resolver.apply_entity_resolution(docs)
    print(f"Entity resolution statistics:")
    for key, value in entity_stats.items():
        print(f"  {key}: {value}")
    
    # Relation Completion Demo  
    print("\n2. RELATION COMPLETION DEMO:")
    print("-" * 40)
    
    relation_completer = RelationCompleter(confidence_threshold=0.5)
    
    print("Finding missing relations...")
    missing_relations = relation_completer.find_missing_relations(docs)
    print(f"Missing relation statistics:")
    for key, value in missing_relations['statistics'].items():
        print(f"  {key}: {value}")
    
    print("\nApplying relation completion...")
    completed_docs, relation_stats = relation_completer.apply_relation_completion(docs)
    print(f"Relation completion statistics:")
    for key, value in relation_stats.items():
        print(f"  {key}: {value}")


def run_entity_resolution_comparison():
    """Compare different entity resolution approaches."""
    print("=" * 60)
    print("ENTITY RESOLUTION COMPARISON")
    print("=" * 60)
    
    # Run the comprehensive evaluation
    evaluator = EntityResolutionEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n✅ Entity resolution comparison completed!")
    print(f"Best approach: {results['best_approach']['name']} (F1: {results['best_approach']['f1_score']:.3f})")
    
    return results


def run_single_approach(approach_name: str):
    """Run a single entity resolution approach."""
    print(f"Running {approach_name} approach...")
    
    # Create test documents
    docs = create_demo_dataset()
    
    # Initialize the specified approach
    approaches = {
        'string_similarity': StringSimilarityResolver(threshold=0.7),
        'rule_based': RuleBasedResolver(),
        'graph_embedding': GraphEmbeddingResolver(embedding_dim=64),
        'rl_based': RLKnowledgeGraphHealer(),
        'original': EntityResolver()
    }
    
    if approach_name not in approaches:
        print(f"Unknown approach: {approach_name}")
        print(f"Available approaches: {list(approaches.keys())}")
        return None
    
    resolver = approaches[approach_name]
    
    # Apply entity resolution
    if hasattr(resolver, 'find_duplicate_entities'):
        # New interface (approaches have find_duplicate_entities method)
        results = resolver.find_duplicate_entities(docs)
        print(f"Found {len(results.get('duplicates', []))} duplicate pairs")
        
        # Show some examples
        for i, dup in enumerate(results.get('duplicates', [])[:3]):
            entity1 = dup['entity1']['name']
            entity2 = dup['entity2']['name']
            confidence = dup.get('confidence', dup.get('rl_confidence', 'N/A'))
            print(f"  {i+1}. {entity1} <-> {entity2} (confidence: {confidence})")
    else:
        # Original interface
        duplicates = resolver.find_duplicate_entities(docs)
        print(f"Found {len(duplicates)} potential duplicate surface forms")
        
        for surface_form, mentions in list(duplicates.items())[:3]:
            print(f"  '{surface_form}': {len(mentions)} mentions")
    
    return results if 'results' in locals() else duplicates


def run_benchmark_demo():
    """Demonstrate benchmarking capabilities."""
    print("=" * 60)
    print("BENCHMARKING DEMONSTRATION") 
    print("=" * 60)
    
    # Create different test datasets
    demo_docs = create_demo_dataset()
    
    # Split into different test sets
    test_datasets = [
        ("small_demo", demo_docs[:3]),
        ("full_demo", demo_docs)
    ]
    
    # Define different healing configurations
    healing_configs = [
        {
            'name': 'conservative',
            'entity_similarity_threshold': 0.9,
            'relation_confidence_threshold': 0.7,
            'verbose': False
        },
        {
            'name': 'balanced', 
            'entity_similarity_threshold': 0.8,
            'relation_confidence_threshold': 0.5,
            'verbose': False
        },
        {
            'name': 'aggressive',
            'entity_similarity_threshold': 0.6,
            'relation_confidence_threshold': 0.3,
            'verbose': False
        }
    ]
    
    # Run benchmark
    print("Running benchmark...")
    benchmark_runner = BenchmarkRunner()
    results = benchmark_runner.run_benchmark(
        test_datasets=test_datasets,
        healing_configs=healing_configs
    )
    
    # Show summary
    print("\nBenchmark Summary:")
    summary = results['summary']
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Show best configurations
    print("\nBest Configurations:")
    comparison = results['config_comparison']
    for metric, dataset_results in comparison.items():
        print(f"  {metric}:")
        for dataset, (config, score) in dataset_results.items():
            print(f"    {dataset}: {config} (score: {score:.3f})")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Healing for DocRED - Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo basic          # Run basic healing demo
  python main.py --demo components     # Show individual components  
  python main.py --demo benchmark      # Run benchmarking demo
  python main.py --demo comparison     # Compare entity resolution approaches
  python main.py --approach rule_based # Run specific approach
  python main.py --demo all           # Run all demonstrations
        """
    )
    
    parser.add_argument(
        '--demo',
        choices=['basic', 'components', 'benchmark', 'comparison', 'all'],
        default='basic',
        help='Type of demonstration to run'
    )
    
    parser.add_argument(
        '--approach',
        choices=['string_similarity', 'rule_based', 'graph_embedding', 'rl_based', 'original'],
        help='Run a specific entity resolution approach'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save results (JSON format)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        # Redirect output to reduce verbosity
        import io
        import contextlib
        
        @contextlib.contextmanager
        def suppress_stdout():
            with open('/dev/null', 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    yield
    
    try:
        # Handle specific approach request
        if args.approach:
            print(f"Running {args.approach} approach...")
            results = run_single_approach(args.approach)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to {args.output}")
        
        # Handle demo types
        if args.demo in ['basic', 'all']:
            print("Running basic demonstration...")
            healed_docs, stats = run_basic_demo()
            
            if args.output and not args.approach:
                from utils import save_json_file
                save_json_file({
                    'healed_documents': healed_docs,
                    'healing_statistics': stats
                }, args.output)
                print(f"\nResults saved to {args.output}")
        
        if args.demo in ['components', 'all']:
            print("\nRunning component demonstration...")
            run_component_demo()
        
        if args.demo in ['benchmark', 'all']:
            print("\nRunning benchmark demonstration...")
            run_benchmark_demo()
        
        if args.demo in ['comparison', 'all']:
            print("\nRunning entity resolution comparison...")
            comparison_results = run_entity_resolution_comparison()
            
            if args.output and not args.approach:
                with open(args.output.replace('.json', '_comparison.json'), 'w') as f:
                    json.dump(comparison_results, f, indent=2, default=str)
                print(f"\nComparison results saved to {args.output.replace('.json', '_comparison.json')}")
        
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
