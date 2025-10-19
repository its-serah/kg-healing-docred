#!/usr/bin/env python3
"""
Data preprocessing and EDA utilities extracted from final_thesisforgood.py
Contains analysis functions, visualization, and data preparation utilities
"""

import os
import json
import random
import warnings
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from thesis_components import setup_nltk, clean_tokens, build_doc_graph

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =====================================
# Data Loading and Basic Stats
# =====================================

def load_docred_data(data_dir="data"):
    """Load DocRED train/dev/test data"""
    def load_split(filepath):
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return []
        with open(filepath, 'r') as f:
            return json.load(f)
    
    train_data = load_split(os.path.join(data_dir, 'train_revised.json'))
    dev_data = load_split(os.path.join(data_dir, 'dev_revised.json'))
    test_data = load_split(os.path.join(data_dir, 'test_revised.json'))
    
    print(f"Loaded DocRED data:")
    print(f"  Train: {len(train_data)} documents")
    print(f"  Dev: {len(dev_data)} documents")  
    print(f"  Test: {len(test_data)} documents")
    
    return train_data, dev_data, test_data

def basic_dataset_stats(data_splits):
    """Print basic statistics for all splits"""
    stats = {}
    
    for split_name, data in data_splits.items():
        n_docs = len(data)
        n_entities = sum(len(doc['vertexSet']) for doc in data)
        n_relations = sum(len(doc.get('labels', [])) for doc in data)
        n_sentences = sum(len(doc.get('sents', [])) for doc in data)
        
        avg_entities = n_entities / max(1, n_docs)
        avg_relations = n_relations / max(1, n_docs)
        avg_sentences = n_sentences / max(1, n_docs)
        
        stats[split_name] = {
            'documents': n_docs,
            'entities': n_entities,
            'relations': n_relations,
            'sentences': n_sentences,
            'avg_entities_per_doc': avg_entities,
            'avg_relations_per_doc': avg_relations,
            'avg_sentences_per_doc': avg_sentences
        }
        
        print(f"\n=== {split_name.upper()} SPLIT STATISTICS ===")
        print(f"Documents: {n_docs}")
        print(f"Total entities: {n_entities} (avg: {avg_entities:.1f}/doc)")
        print(f"Total relations: {n_relations} (avg: {avg_relations:.1f}/doc)")
        print(f"Total sentences: {n_sentences} (avg: {avg_sentences:.1f}/doc)")
    
    return stats

# =====================================
# Exploratory Data Analysis
# =====================================

def explore_document_structure(doc, doc_idx=0):
    """Detailed exploration of a single document"""
    print(f"=== Document {doc_idx}: {doc.get('title', 'Untitled')} ===\n")
    
    # Basic info
    print(f"Entities: {len(doc.get('vertexSet', []))}")
    print(f"Relations: {len(doc.get('labels', []))}")
    print(f"Sentences: {len(doc.get('sents', []))}")
    
    # Show first few sentences
    print(f"\nFirst 3 sentences:")
    for i, sent in enumerate(doc.get('sents', [])[:3]):
        sent_text = " ".join(sent)
        print(f"[{i}] {sent_text}")
    
    # Show entity clusters
    print(f"\nFirst 5 entity clusters:")
    for i, cluster in enumerate(doc.get('vertexSet', [])[:5]):
        mentions = [m['name'] for m in cluster]
        types = {m['type'] for m in cluster}
        print(f"Entity {i} ({'/'.join(types)}): {mentions}")
    
    # Show relations
    print(f"\nFirst 5 relations:")
    for i, rel in enumerate(doc.get('labels', [])[:5]):
        try:
            head_name = doc['vertexSet'][rel['h']][0]['name']
            tail_name = doc['vertexSet'][rel['t']][0]['name']
            print(f"{head_name} --[{rel['r']}]--> {tail_name}")
        except (IndexError, KeyError):
            print(f"Entity {rel['h']} --[{rel['r']}]--> Entity {rel['t']}")

def analyze_entity_types(data_splits, plot=True):
    """Analyze entity type distributions across splits"""
    type_stats = {}
    
    for split_name, data in data_splits.items():
        type_counter = Counter()
        for doc in data:
            for cluster in doc.get('vertexSet', []):
                for mention in cluster:
                    type_counter[mention.get('type', 'UNKNOWN')] += 1
        
        type_stats[split_name] = dict(type_counter)
        
        print(f"\n=== {split_name.upper()} ENTITY TYPES ===")
        for entity_type, count in type_counter.most_common(10):
            print(f"{entity_type}: {count}")
    
    if plot:
        # Plot entity type distributions
        fig, axes = plt.subplots(1, len(data_splits), figsize=(5*len(data_splits), 6))
        if len(data_splits) == 1:
            axes = [axes]
        
        for i, (split_name, data) in enumerate(data_splits.items()):
            type_counter = Counter()
            for doc in data:
                for cluster in doc.get('vertexSet', []):
                    for mention in cluster:
                        type_counter[mention.get('type', 'UNKNOWN')] += 1
            
            types, counts = zip(*type_counter.most_common(15))
            axes[i].bar(range(len(types)), counts)
            axes[i].set_title(f'Entity Types ({split_name})')
            axes[i].set_xticks(range(len(types)))
            axes[i].set_xticklabels(types, rotation=45, ha='right')
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    return type_stats

def analyze_relation_types(data_splits, plot=True):
    """Analyze relation type distributions"""
    relation_stats = {}
    
    for split_name, data in data_splits.items():
        rel_counter = Counter()
        for doc in data:
            for rel in doc.get('labels', []):
                rel_counter[rel.get('r', 'UNKNOWN')] += 1
        
        relation_stats[split_name] = dict(rel_counter)
        
        print(f"\n=== {split_name.upper()} RELATION TYPES ===")
        for rel_type, count in rel_counter.most_common(10):
            print(f"{rel_type}: {count}")
    
    if plot:
        # Plot relation distributions
        fig, axes = plt.subplots(1, len(data_splits), figsize=(6*len(data_splits), 6))
        if len(data_splits) == 1:
            axes = [axes]
        
        for i, (split_name, data) in enumerate(data_splits.items()):
            rel_counter = Counter()
            for doc in data:
                for rel in doc.get('labels', []):
                    rel_counter[rel.get('r', 'UNKNOWN')] += 1
            
            relations, counts = zip(*rel_counter.most_common(20))
            axes[i].bar(range(len(relations)), counts)
            axes[i].set_title(f'Relation Types ({split_name})')
            axes[i].set_xticks(range(len(relations)))
            axes[i].set_xticklabels(relations, rotation=90, ha='right')
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    return relation_stats

def analyze_mention_clusters(data_splits, plot=True):
    """Analyze mention clustering patterns"""
    cluster_stats = {}
    
    for split_name, data in data_splits.items():
        cluster_sizes = []
        for doc in data:
            for cluster in doc.get('vertexSet', []):
                cluster_sizes.append(len(cluster))
        
        cluster_stats[split_name] = {
            'sizes': cluster_sizes,
            'mean_size': np.mean(cluster_sizes),
            'median_size': np.median(cluster_sizes),
            'max_size': max(cluster_sizes) if cluster_sizes else 0
        }
        
        print(f"\n=== {split_name.upper()} MENTION CLUSTERS ===")
        print(f"Mean cluster size: {cluster_stats[split_name]['mean_size']:.2f}")
        print(f"Median cluster size: {cluster_stats[split_name]['median_size']:.1f}")
        print(f"Max cluster size: {cluster_stats[split_name]['max_size']}")
    
    if plot:
        fig, axes = plt.subplots(1, len(data_splits), figsize=(5*len(data_splits), 4))
        if len(data_splits) == 1:
            axes = [axes]
        
        for i, (split_name, stats) in enumerate(cluster_stats.items()):
            axes[i].hist(stats['sizes'], bins=range(1, max(stats['sizes'])+2), 
                        alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Mention Cluster Sizes ({split_name})')
            axes[i].set_xlabel('Mentions per Entity')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    return cluster_stats

def find_healing_opportunities(data, split_name, max_docs=100):
    """Find specific opportunities for KG healing"""
    print(f"\n=== {split_name} HEALING OPPORTUNITIES ===")
    
    # Entity duplicate opportunities
    entity_surface_forms = defaultdict(list)
    
    # Relation chain opportunities  
    doc_relations = []
    
    # Relation type distribution
    relation_types = Counter()
    
    sample_data = data[:max_docs] if len(data) > max_docs else data
    
    for doc_idx, doc in enumerate(sample_data):
        # Analyze entities
        for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
            for mention in entity_cluster:
                surface_form = mention.get('name', '').lower().strip()
                entity_type = mention.get('type', 'UNKNOWN')
                entity_surface_forms[surface_form].append((doc_idx, ent_idx, entity_type))
        
        # Analyze relations
        for rel in doc.get('labels', []):
            head_idx = rel.get('h')
            tail_idx = rel.get('t')
            relation_type = rel.get('r')
            
            if head_idx is not None and tail_idx is not None and relation_type:
                doc_relations.append((doc_idx, head_idx, tail_idx, relation_type))
                relation_types[relation_type] += 1
    
    # Find opportunities
    print(f"Analyzed {len(sample_data)} documents")
    
    # Potential entity duplicates
    potential_duplicates = {sf: entities for sf, entities in entity_surface_forms.items()
                          if len(set(e[1] for e in entities)) > 1}
    
    print(f"\n1. ENTITY DUPLICATE OPPORTUNITIES:")
    print(f"   - Potential duplicate surface forms: {len(potential_duplicates)}")
    if potential_duplicates:
        for sf, entities in list(potential_duplicates.items())[:5]:
            print(f"   - '{sf}': appears in {len(entities)} different entity clusters")
    
    # Relation chain opportunities
    print(f"\n2. RELATION CHAIN OPPORTUNITIES:")
    doc_graphs = defaultdict(list)
    
    for doc_idx, head, tail, rel in doc_relations:
        doc_graphs[doc_idx].append((head, tail, rel))
    
    potential_chains = []
    for doc_idx, relations in doc_graphs.items():
        # Find 2-hop paths
        for r1 in relations:
            for r2 in relations:
                if r1[1] == r2[0] and r1[0] != r2[1]:  # A->B, B->C (not A->B, B->A)
                    # Check if direct A->C exists
                    direct_exists = any(r[0] == r1[0] and r[1] == r2[1] for r in relations)
                    potential_chains.append({
                        'doc_idx': doc_idx,
                        'chain': [r1[0], r1[1], r2[1]],
                        'relations': [r1[2], r2[2]],
                        'direct_exists': direct_exists
                    })
    
    print(f"   - Potential 2-hop chains found: {len(potential_chains)}")
    if potential_chains:
        example = potential_chains[0]
        print(f"   - Example: Entity {example['chain'][0]} -> {example['chain'][1]} -> {example['chain'][2]}")
        print(f"     Relations: {example['relations'][0]} -> {example['relations'][1]}")
        print(f"     Direct relation exists: {example['direct_exists']}")
    
    # Relation type analysis
    print(f"\n3. RELATION TYPE ANALYSIS:")
    print(f"   - Total unique relation types: {len(relation_types)}")
    print(f"   - Top 5 relation types: {dict(relation_types.most_common(5))}")
    
    return {
        'potential_duplicates': potential_duplicates,
        'potential_chains': potential_chains[:50],  # Sample
        'relation_types': relation_types,
        'total_relations': len(doc_relations)
    }

# =====================================
# Graph-level Analysis
# =====================================

def analyze_graph_properties(data_splits, sample_docs=200):
    """Analyze graph-theoretic properties of documents"""
    graph_stats = {}
    
    for split_name, data in data_splits.items():
        print(f"\nAnalyzing graph properties for {split_name} split...")
        
        sample_data = data[:sample_docs] if len(data) > sample_docs else data
        stats_list = []
        
        for doc_idx, doc in enumerate(tqdm(sample_data, desc=f"Processing {split_name}")):
            G = build_doc_graph(doc)
            
            stats = {
                'doc_idx': doc_idx,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G) if G.number_of_nodes() > 1 else 0.0,
                'components': nx.number_connected_components(G.to_undirected()),
            }
            
            # Degree statistics
            degrees = [d for _, d in G.degree()]
            if degrees:
                stats['avg_degree'] = np.mean(degrees)
                stats['max_degree'] = max(degrees)
            else:
                stats['avg_degree'] = 0.0
                stats['max_degree'] = 0
            
            # Clustering coefficient
            try:
                stats['clustering'] = nx.average_clustering(G.to_undirected())
            except:
                stats['clustering'] = 0.0
            
            stats_list.append(stats)
        
        graph_stats[split_name] = pd.DataFrame(stats_list)
        
        # Summary statistics
        df = graph_stats[split_name]
        print(f"\n{split_name.upper()} Graph Statistics:")
        print(f"Avg nodes per doc: {df['nodes'].mean():.1f} ± {df['nodes'].std():.1f}")
        print(f"Avg edges per doc: {df['edges'].mean():.1f} ± {df['edges'].std():.1f}")
        print(f"Avg density: {df['density'].mean():.3f} ± {df['density'].std():.3f}")
        print(f"Avg components: {df['components'].mean():.1f} ± {df['components'].std():.1f}")
        print(f"Avg clustering: {df['clustering'].mean():.3f} ± {df['clustering'].std():.3f}")
    
    return graph_stats

def plot_graph_distributions(graph_stats):
    """Plot distributions of graph properties"""
    n_splits = len(graph_stats)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['nodes', 'edges', 'density', 'components', 'avg_degree', 'clustering']
    titles = ['Nodes per Doc', 'Edges per Doc', 'Graph Density', 'Connected Components', 
              'Average Degree', 'Clustering Coefficient']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        for split_name, df in graph_stats.items():
            axes[i].hist(df[metric], alpha=0.6, label=split_name, bins=30)
        axes[i].set_title(title)
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def analyze_chain_redundancy(data_splits, max_docs=200):
    """Analyze 2-hop chain redundancy patterns"""
    redundancy_stats = {}
    
    for split_name, data in data_splits.items():
        print(f"\nAnalyzing chain redundancy for {split_name}...")
        
        sample_data = data[:max_docs] if len(data) > max_docs else data
        stats_list = []
        
        for doc_idx, doc in enumerate(tqdm(sample_data, desc=f"Processing {split_name}")):
            G = build_doc_graph(doc)
            
            two_hop = 0
            redundant = 0
            
            for a in G.nodes:
                for b in G.successors(a):
                    for c in G.successors(b):
                        two_hop += 1
                        if G.has_edge(a, c):
                            redundant += 1
            
            stats_list.append({
                'doc_idx': doc_idx,
                'two_hop_paths': two_hop,
                'redundant_paths': redundant,
                'redundancy_rate': redundant / max(1, two_hop)
            })
        
        redundancy_stats[split_name] = pd.DataFrame(stats_list)
        
        # Summary
        df = redundancy_stats[split_name]
        print(f"Avg redundancy rate: {df['redundancy_rate'].mean():.3f} ± {df['redundancy_rate'].std():.3f}")
        print(f"Total 2-hop paths: {df['two_hop_paths'].sum()}")
        print(f"Total redundant paths: {df['redundant_paths'].sum()}")
    
    return redundancy_stats

# =====================================
# Surface Form Analysis
# =====================================

def analyze_surface_form_duplicates(data_splits, top_n=10):
    """Analyze potential entity duplicates based on surface forms"""
    duplicate_stats = {}
    
    for split_name, data in data_splits.items():
        surface_forms = defaultdict(list)
        
        for doc_idx, doc in enumerate(data):
            for ent_idx, cluster in enumerate(doc.get('vertexSet', [])):
                for mention in cluster:
                    sf = mention.get('name', '').lower().strip()
                    surface_forms[sf].append((doc_idx, ent_idx, mention.get('type', 'UNKNOWN')))
        
        # Find duplicates (same surface form, different entity clusters)
        duplicates = {}
        for sf, entities in surface_forms.items():
            unique_entities = set((e[0], e[1]) for e in entities)  # (doc_idx, ent_idx) pairs
            if len(unique_entities) > 1:
                duplicates[sf] = entities
        
        duplicate_stats[split_name] = duplicates
        
        print(f"\n=== {split_name.upper()} SURFACE FORM ANALYSIS ===")
        print(f"Total unique surface forms: {len(surface_forms)}")
        print(f"Potential duplicates: {len(duplicates)}")
        
        # Show top duplicates
        print(f"\nTop {top_n} potential duplicates:")
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        for sf, entities in sorted_duplicates[:top_n]:
            n_docs = len(set(e[0] for e in entities))
            n_entities = len(set((e[0], e[1]) for e in entities))
            types = set(e[2] for e in entities)
            print(f"  '{sf}': {len(entities)} mentions, {n_entities} entities, {n_docs} docs, types: {types}")
    
    return duplicate_stats

# =====================================
# Visualization Utilities
# =====================================

def visualize_document_graph(doc, doc_idx=0, max_nodes=20, figsize=(10, 8)):
    """Visualize a document's graph structure"""
    G = build_doc_graph(doc)
    
    # Limit graph size for visualization
    if G.number_of_nodes() > max_nodes:
        nodes_to_keep = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes_to_keep)
        print(f"Showing subgraph with {len(nodes_to_keep)} nodes (out of {len(doc.get('vertexSet', []))})")
    
    plt.figure(figsize=figsize)
    
    # Node labels: entity names
    labels = {}
    node_colors = []
    
    for i in G.nodes():
        try:
            name = doc['vertexSet'][i][0]['name']
            entity_type = doc['vertexSet'][i][0].get('type', 'UNKNOWN')
            labels[i] = name[:15] + ('...' if len(name) > 15 else '')
            
            # Color by entity type
            type_colors = {'PER': 'lightblue', 'ORG': 'lightgreen', 'LOC': 'lightcoral', 
                          'MISC': 'lightyellow', 'NUM': 'lightpink'}
            node_colors.append(type_colors.get(entity_type, 'lightgray'))
        except (IndexError, KeyError):
            labels[i] = f"Entity_{i}"
            node_colors.append('lightgray')
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
            node_size=800, font_size=8, arrows=True, edge_color='gray', alpha=0.7)
    
    # Add edge labels for relations
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', '')
        edge_labels[(u, v)] = relation
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    plt.title(f"Document {doc_idx}: {doc.get('title', 'Untitled')}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_summary_report(data_splits, output_file="docred_analysis_report.txt"):
    """Create a comprehensive summary report"""
    with open(output_file, 'w') as f:
        f.write("DocRED Dataset Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic stats
        stats = basic_dataset_stats(data_splits)
        f.write("BASIC STATISTICS\n")
        f.write("-" * 20 + "\n")
        for split_name, split_stats in stats.items():
            f.write(f"{split_name.upper()}:\n")
            for key, value in split_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Entity types
        entity_types = analyze_entity_types(data_splits, plot=False)
        f.write("ENTITY TYPES\n")
        f.write("-" * 15 + "\n")
        for split_name, types in entity_types.items():
            f.write(f"{split_name.upper()}:\n")
            sorted_types = sorted(types.items(), key=lambda x: x[1], reverse=True)
            for entity_type, count in sorted_types[:10]:
                f.write(f"  {entity_type}: {count}\n")
            f.write("\n")
        
        # Relation types
        relation_types = analyze_relation_types(data_splits, plot=False)
        f.write("RELATION TYPES\n")
        f.write("-" * 15 + "\n")
        for split_name, rels in relation_types.items():
            f.write(f"{split_name.upper()}:\n")
            sorted_rels = sorted(rels.items(), key=lambda x: x[1], reverse=True)
            for rel_type, count in sorted_rels[:10]:
                f.write(f"  {rel_type}: {count}\n")
            f.write("\n")
        
        # Healing opportunities
        f.write("HEALING OPPORTUNITIES\n")
        f.write("-" * 25 + "\n")
        for split_name, data in data_splits.items():
            opportunities = find_healing_opportunities(data, split_name, max_docs=100)
            f.write(f"{split_name.upper()}:\n")
            f.write(f"  Potential duplicates: {len(opportunities['potential_duplicates'])}\n")
            f.write(f"  Potential chains: {len(opportunities['potential_chains'])}\n")
            f.write(f"  Total relations: {opportunities['total_relations']}\n")
            f.write("\n")
    
    print(f"Summary report saved to {output_file}")

# =====================================
# Main Analysis Function
# =====================================

def run_comprehensive_analysis(data_dir="data", max_docs=200, create_plots=True, save_report=True):
    """Run comprehensive EDA on DocRED dataset"""
    print("Loading DocRED dataset...")
    train_data, dev_data, test_data = load_docred_data(data_dir)
    
    data_splits = {
        'train': train_data,
        'dev': dev_data,  
        'test': test_data
    }
    
    print("\n" + "="*60)
    print("COMPREHENSIVE DOCRED ANALYSIS")
    print("="*60)
    
    # Basic statistics
    basic_dataset_stats(data_splits)
    
    # Example document exploration
    if train_data:
        print("\n" + "-"*60)
        print("EXAMPLE DOCUMENT STRUCTURE")
        print("-"*60)
        explore_document_structure(train_data[0], doc_idx=0)
    
    # Entity and relation analysis
    print("\n" + "-"*60)
    print("ENTITY TYPE ANALYSIS")
    print("-"*60)
    entity_stats = analyze_entity_types(data_splits, plot=create_plots)
    
    print("\n" + "-"*60)
    print("RELATION TYPE ANALYSIS")  
    print("-"*60)
    relation_stats = analyze_relation_types(data_splits, plot=create_plots)
    
    # Mention cluster analysis
    print("\n" + "-"*60)
    print("MENTION CLUSTER ANALYSIS")
    print("-"*60)
    cluster_stats = analyze_mention_clusters(data_splits, plot=create_plots)
    
    # Graph-level analysis
    print("\n" + "-"*60)
    print("GRAPH-LEVEL ANALYSIS")
    print("-"*60)
    graph_stats = analyze_graph_properties(data_splits, sample_docs=max_docs)
    
    if create_plots:
        plot_graph_distributions(graph_stats)
    
    # Chain redundancy analysis
    print("\n" + "-"*60)
    print("CHAIN REDUNDANCY ANALYSIS")
    print("-"*60)
    redundancy_stats = analyze_chain_redundancy(data_splits, max_docs=max_docs)
    
    # Surface form duplicates
    print("\n" + "-"*60)
    print("SURFACE FORM DUPLICATE ANALYSIS")
    print("-"*60)
    duplicate_stats = analyze_surface_form_duplicates(data_splits)
    
    # Healing opportunities
    print("\n" + "-"*60)
    print("HEALING OPPORTUNITY ANALYSIS")
    print("-"*60)
    healing_opportunities = {}
    for split_name, data in data_splits.items():
        healing_opportunities[split_name] = find_healing_opportunities(data, split_name, max_docs=max_docs)
    
    # Visualization
    if create_plots and train_data:
        print("\n" + "-"*60)
        print("GRAPH VISUALIZATION")
        print("-"*60)
        visualize_document_graph(train_data[0], doc_idx=0)
    
    # Create summary report
    if save_report:
        create_summary_report(data_splits)
    
    return {
        'basic_stats': basic_dataset_stats(data_splits),
        'entity_stats': entity_stats,
        'relation_stats': relation_stats,
        'cluster_stats': cluster_stats,
        'graph_stats': graph_stats,
        'redundancy_stats': redundancy_stats,
        'duplicate_stats': duplicate_stats,
        'healing_opportunities': healing_opportunities
    }

if __name__ == "__main__":
    # Setup NLTK if available
    setup_nltk()
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis(
        data_dir="data",
        max_docs=200,
        create_plots=True,
        save_report=True
    )
    
    print("\nAnalysis complete! Check docred_analysis_report.txt for detailed results.")
