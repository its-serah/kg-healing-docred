"""
Data loader for DocRED dataset with knowledge graph healing functionality.
Loads DocRED data and prepares it for entity resolution and relation completion.
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx


class DocREDLoader:
    """Loads and preprocesses DocRED dataset for KG healing."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing DocRED dataset files
        """
        self.data_dir = data_dir
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        
    def load_dataset(self, splits: List[str] = ['train', 'dev', 'test']) -> Dict[str, List[Dict]]:
        """Load DocRED dataset splits.
        
        Args:
            splits: List of splits to load ('train', 'dev', 'test')
            
        Returns:
            Dictionary mapping split names to loaded data
        """
        data = {}
        
        for split in splits:
            filepath = os.path.join(self.data_dir, f'{split}_annotated.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data[split] = json.load(f)
                print(f"Loaded {len(data[split])} documents from {split} split")
            else:
                print(f"Warning: {filepath} not found")
                
        return data
    
    def preprocess_documents(self, docs: List[Dict]) -> List[Dict]:
        """Preprocess documents for KG healing.
        
        Args:
            docs: List of raw DocRED documents
            
        Returns:
            List of preprocessed documents with additional metadata
        """
        processed_docs = []
        
        for i, doc in enumerate(docs):
            processed_doc = doc.copy()
            
            # Add document index
            processed_doc['doc_idx'] = i
            
            # Add entity and relation counts
            processed_doc['n_entities'] = len(doc.get('vertexSet', []))
            processed_doc['n_relations'] = len(doc.get('labels', []))
            processed_doc['n_sentences'] = len(doc.get('sents', []))
            
            # Extract entity mentions with additional metadata
            entity_mentions = []
            for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
                for mention in entity_cluster:
                    mention_data = mention.copy()
                    mention_data['entity_idx'] = ent_idx
                    entity_mentions.append(mention_data)
            
            processed_doc['entity_mentions'] = entity_mentions
            
            processed_docs.append(processed_doc)
            
        return processed_docs
    
    def build_document_graph(self, doc: Dict) -> nx.DiGraph:
        """Build a NetworkX graph for a single document.
        
        Args:
            doc: Document dictionary
            
        Returns:
            NetworkX directed graph representing the document's KG
        """
        graph = nx.DiGraph()
        
        # Add nodes (entities)
        n_entities = len(doc.get('vertexSet', []))
        graph.add_nodes_from(range(n_entities))
        
        # Add edges (relations)
        for rel in doc.get('labels', []):
            head_idx = rel['h']
            tail_idx = rel['t']
            relation_type = rel['r']
            evidence = rel.get('evidence', [])
            
            graph.add_edge(head_idx, tail_idx, 
                         relation=relation_type, 
                         evidence=evidence)
        
        return graph
    
    def extract_entity_surface_forms(self, docs: List[Dict]) -> Dict[str, List[Tuple]]:
        """Extract entity surface forms for duplicate detection.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary mapping surface forms to list of (doc_idx, entity_idx, type) tuples
        """
        surface_forms = defaultdict(list)
        
        for doc_idx, doc in enumerate(docs):
            for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
                for mention in entity_cluster:
                    surface_form = mention['name'].lower().strip()
                    entity_type = mention['type']
                    surface_forms[surface_form].append((doc_idx, ent_idx, entity_type))
                    
        return surface_forms
    
    def find_relation_chains(self, doc: Dict, max_length: int = 2) -> List[Dict]:
        """Find potential relation chains in a document.
        
        Args:
            doc: Document dictionary
            max_length: Maximum chain length to search for
            
        Returns:
            List of relation chain dictionaries
        """
        chains = []
        relations = doc.get('labels', [])
        
        # Build adjacency lists for efficient chain finding
        outgoing = defaultdict(list)
        for rel in relations:
            outgoing[rel['h']].append((rel['t'], rel['r']))
        
        # Find 2-hop chains (A -> B -> C)
        if max_length >= 2:
            for rel1 in relations:
                h1, t1, r1 = rel1['h'], rel1['t'], rel1['r']
                
                for t2, r2 in outgoing[t1]:
                    if h1 != t2:  # Avoid A -> B -> A patterns
                        # Check if direct relation exists
                        direct_exists = any(r['h'] == h1 and r['t'] == t2 for r in relations)
                        
                        chains.append({
                            'entities': [h1, t1, t2],
                            'relations': [r1, r2],
                            'direct_exists': direct_exists,
                            'confidence': 0.5  # Default confidence
                        })
        
        return chains
    
    def compute_document_statistics(self, docs: List[Dict]) -> Dict[str, Any]:
        """Compute statistics for a set of documents.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'n_documents': len(docs),
            'total_entities': 0,
            'total_relations': 0,
            'total_sentences': 0,
            'entity_types': Counter(),
            'relation_types': Counter(),
            'avg_entities_per_doc': 0,
            'avg_relations_per_doc': 0,
            'avg_sentences_per_doc': 0
        }
        
        for doc in docs:
            # Count entities and relations
            n_entities = len(doc.get('vertexSet', []))
            n_relations = len(doc.get('labels', []))
            n_sentences = len(doc.get('sents', []))
            
            stats['total_entities'] += n_entities
            stats['total_relations'] += n_relations
            stats['total_sentences'] += n_sentences
            
            # Count entity types
            for entity_cluster in doc.get('vertexSet', []):
                for mention in entity_cluster:
                    stats['entity_types'][mention['type']] += 1
            
            # Count relation types
            for rel in doc.get('labels', []):
                stats['relation_types'][rel['r']] += 1
        
        # Compute averages
        if stats['n_documents'] > 0:
            stats['avg_entities_per_doc'] = stats['total_entities'] / stats['n_documents']
            stats['avg_relations_per_doc'] = stats['total_relations'] / stats['n_documents']
            stats['avg_sentences_per_doc'] = stats['total_sentences'] / stats['n_documents']
        
        return stats
    
    def create_sample_dataset(self, docs: List[Dict], n_samples: int = 50) -> List[Dict]:
        """Create a smaller sample of the dataset for testing.
        
        Args:
            docs: Full dataset
            n_samples: Number of samples to extract
            
        Returns:
            Sample dataset
        """
        return docs[:min(n_samples, len(docs))]


def download_docred_sample():
    """Download a sample of DocRED dataset for demonstration purposes."""
    print("Note: This is a placeholder. In a real implementation, you would:")
    print("1. Download DocRED dataset from https://github.com/thunlp/DocRED")
    print("2. Extract the files to the data/ directory")
    print("3. Ensure files are named: train_annotated.json, dev.json, test.json")
    
    # Create sample data structure
    sample_doc = {
        "title": "Sample Document",
        "sents": [
            ["Apple", "Inc.", "is", "an", "American", "technology", "company"],
            ["Tim", "Cook", "is", "the", "CEO", "of", "Apple"]
        ],
        "vertexSet": [
            [{"name": "Apple Inc.", "sent_id": 0, "pos": [0, 2], "type": "ORG"}],
            [{"name": "American", "sent_id": 0, "pos": [4, 5], "type": "MISC"}],
            [{"name": "Tim Cook", "sent_id": 1, "pos": [0, 2], "type": "PER"}],
            [{"name": "Apple", "sent_id": 1, "pos": [6, 7], "type": "ORG"}]
        ],
        "labels": [
            {"h": 0, "t": 1, "r": "P17", "evidence": [0]},  # Apple -> American (country)
            {"h": 2, "t": 0, "r": "P108", "evidence": [1]}   # Tim Cook -> Apple (employer)
        ]
    }
    
    return [sample_doc]


if __name__ == "__main__":
    # Example usage
    loader = DocREDLoader()
    
    # For demonstration, create sample data
    sample_data = download_docred_sample()
    
    # Preprocess the data
    processed_docs = loader.preprocess_documents(sample_data)
    
    # Compute statistics
    stats = loader.compute_document_statistics(processed_docs)
    print("Dataset statistics:", stats)
    
    # Build graph for first document
    if processed_docs:
        graph = loader.build_document_graph(processed_docs[0])
        print(f"Document graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Find relation chains
        chains = loader.find_relation_chains(processed_docs[0])
        print(f"Found {len(chains)} potential relation chains")
