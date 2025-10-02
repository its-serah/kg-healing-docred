"""
Utility functions for knowledge graph healing.
Common helper functions used across modules.
"""

import json
import re
import os
import time
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import defaultdict, Counter
import networkx as nx


def load_json_file(file_path: str) -> Any:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_jsonlines_file(file_path: str) -> List[Dict]:
    """Load data from a JSON Lines file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data


def save_jsonlines_file(data: List[Dict], file_path: str) -> None:
    """Save data to a JSON Lines file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')


def normalize_text(text: str) -> str:
    """Normalize text for better comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two text strings.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    from difflib import SequenceMatcher
    
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Substring match
    if norm1 in norm2 or norm2 in norm1:
        return 0.9
    
    # Sequence matching
    return SequenceMatcher(None, norm1, norm2).ratio()


def extract_entity_mentions(doc: Dict) -> List[Dict]:
    """Extract all entity mentions from a document.
    
    Args:
        doc: Document dictionary
        
    Returns:
        List of entity mention dictionaries
    """
    mentions = []
    
    for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
        for mention in entity_cluster:
            mention_data = mention.copy()
            mention_data['entity_idx'] = ent_idx
            mention_data['cluster_size'] = len(entity_cluster)
            mentions.append(mention_data)
    
    return mentions


def extract_relations(doc: Dict) -> List[Dict]:
    """Extract all relations from a document.
    
    Args:
        doc: Document dictionary
        
    Returns:
        List of relation dictionaries
    """
    relations = []
    
    for rel in doc.get('labels', []):
        rel_data = rel.copy()
        relations.append(rel_data)
    
    return relations


def build_entity_type_distribution(docs: List[Dict]) -> Dict[str, int]:
    """Build distribution of entity types across documents.
    
    Args:
        docs: List of documents
        
    Returns:
        Dictionary mapping entity types to counts
    """
    type_counts = Counter()
    
    for doc in docs:
        for entity_cluster in doc.get('vertexSet', []):
            if entity_cluster:
                entity_type = entity_cluster[0].get('type', 'UNK')
                type_counts[entity_type] += 1
    
    return dict(type_counts)


def build_relation_type_distribution(docs: List[Dict]) -> Dict[str, int]:
    """Build distribution of relation types across documents.
    
    Args:
        docs: List of documents
        
    Returns:
        Dictionary mapping relation types to counts
    """
    type_counts = Counter()
    
    for doc in docs:
        for rel in doc.get('labels', []):
            rel_type = rel['r']
            type_counts[rel_type] += 1
    
    return dict(type_counts)


def compute_document_statistics(docs: List[Dict]) -> Dict[str, Any]:
    """Compute comprehensive statistics for a document collection.
    
    Args:
        docs: List of documents
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_documents': len(docs),
        'total_entities': 0,
        'total_relations': 0,
        'total_mentions': 0,
        'entity_types': {},
        'relation_types': {},
        'avg_entities_per_doc': 0,
        'avg_relations_per_doc': 0,
        'avg_mentions_per_doc': 0,
        'documents_with_no_entities': 0,
        'documents_with_no_relations': 0
    }
    
    all_entity_counts = []
    all_relation_counts = []
    all_mention_counts = []
    
    for doc in docs:
        entities = doc.get('vertexSet', [])
        relations = doc.get('labels', [])
        mentions = extract_entity_mentions(doc)
        
        entity_count = len(entities)
        relation_count = len(relations)
        mention_count = len(mentions)
        
        stats['total_entities'] += entity_count
        stats['total_relations'] += relation_count  
        stats['total_mentions'] += mention_count
        
        all_entity_counts.append(entity_count)
        all_relation_counts.append(relation_count)
        all_mention_counts.append(mention_count)
        
        if entity_count == 0:
            stats['documents_with_no_entities'] += 1
        if relation_count == 0:
            stats['documents_with_no_relations'] += 1
    
    # Compute averages
    if docs:
        stats['avg_entities_per_doc'] = stats['total_entities'] / len(docs)
        stats['avg_relations_per_doc'] = stats['total_relations'] / len(docs)
        stats['avg_mentions_per_doc'] = stats['total_mentions'] / len(docs)
    
    # Compute type distributions
    stats['entity_types'] = build_entity_type_distribution(docs)
    stats['relation_types'] = build_relation_type_distribution(docs)
    
    return stats


def filter_documents_by_criteria(docs: List[Dict],
                                  min_entities: int = 0,
                                  max_entities: int = float('inf'),
                                  min_relations: int = 0,
                                  max_relations: int = float('inf'),
                                  required_entity_types: Set[str] = None,
                                  required_relation_types: Set[str] = None) -> List[Dict]:
    """Filter documents based on various criteria.
    
    Args:
        docs: List of documents to filter
        min_entities: Minimum number of entities
        max_entities: Maximum number of entities
        min_relations: Minimum number of relations
        max_relations: Maximum number of relations
        required_entity_types: Set of required entity types
        required_relation_types: Set of required relation types
        
    Returns:
        Filtered list of documents
    """
    filtered_docs = []
    
    for doc in docs:
        entities = doc.get('vertexSet', [])
        relations = doc.get('labels', [])
        
        # Check entity count constraints
        if not (min_entities <= len(entities) <= max_entities):
            continue
        
        # Check relation count constraints
        if not (min_relations <= len(relations) <= max_relations):
            continue
        
        # Check required entity types
        if required_entity_types:
            doc_entity_types = set()
            for entity_cluster in entities:
                if entity_cluster:
                    doc_entity_types.add(entity_cluster[0].get('type', 'UNK'))
            
            if not required_entity_types.issubset(doc_entity_types):
                continue
        
        # Check required relation types
        if required_relation_types:
            doc_relation_types = set(rel['r'] for rel in relations)
            
            if not required_relation_types.issubset(doc_relation_types):
                continue
        
        filtered_docs.append(doc)
    
    return filtered_docs


def sample_documents(docs: List[Dict], 
                    n_samples: int, 
                    random_seed: int = 42,
                    stratify_by: str = None) -> List[Dict]:
    """Sample documents from a collection.
    
    Args:
        docs: List of documents to sample from
        n_samples: Number of documents to sample
        random_seed: Random seed for reproducibility
        stratify_by: Optional stratification criterion ('entity_types', 'relation_types')
        
    Returns:
        Sampled list of documents
    """
    import random
    
    if n_samples >= len(docs):
        return docs.copy()
    
    random.seed(random_seed)
    
    if stratify_by is None:
        return random.sample(docs, n_samples)
    
    # Stratified sampling
    if stratify_by == 'entity_types':
        # Group by dominant entity type
        groups = defaultdict(list)
        for doc in docs:
            entities = doc.get('vertexSet', [])
            if entities and entities[0]:
                dominant_type = entities[0][0].get('type', 'UNK')
                groups[dominant_type].append(doc)
        
        # Sample proportionally from each group
        sampled = []
        total_groups = len(groups)
        samples_per_group = n_samples // total_groups
        remaining = n_samples % total_groups
        
        for i, (group_type, group_docs) in enumerate(groups.items()):
            group_samples = samples_per_group + (1 if i < remaining else 0)
            group_samples = min(group_samples, len(group_docs))
            sampled.extend(random.sample(group_docs, group_samples))
        
        return sampled[:n_samples]
    
    else:
        # Default to random sampling
        return random.sample(docs, n_samples)


def create_document_index(docs: List[Dict]) -> Dict[str, Any]:
    """Create an index for efficient document lookup.
    
    Args:
        docs: List of documents
        
    Returns:
        Index dictionary with various lookup tables
    """
    index = {
        'entity_to_docs': defaultdict(list),      # entity_name -> [doc_indices]
        'relation_to_docs': defaultdict(list),    # relation_type -> [doc_indices]
        'type_to_entities': defaultdict(set),     # entity_type -> {entity_names}
        'doc_metadata': []                        # per-document metadata
    }
    
    for doc_idx, doc in enumerate(docs):
        doc_meta = {
            'doc_idx': doc_idx,
            'n_entities': len(doc.get('vertexSet', [])),
            'n_relations': len(doc.get('labels', [])),
            'entity_types': set(),
            'relation_types': set()
        }
        
        # Index entities
        for entity_cluster in doc.get('vertexSet', []):
            for mention in entity_cluster:
                entity_name = mention['name']
                entity_type = mention.get('type', 'UNK')
                
                index['entity_to_docs'][entity_name].append(doc_idx)
                index['type_to_entities'][entity_type].add(entity_name)
                doc_meta['entity_types'].add(entity_type)
        
        # Index relations
        for rel in doc.get('labels', []):
            rel_type = rel['r']
            index['relation_to_docs'][rel_type].append(doc_idx)
            doc_meta['relation_types'].add(rel_type)
        
        # Convert sets to lists for JSON serialization
        doc_meta['entity_types'] = list(doc_meta['entity_types'])
        doc_meta['relation_types'] = list(doc_meta['relation_types'])
        
        index['doc_metadata'].append(doc_meta)
    
    return dict(index)


def merge_document_collections(collections: List[Tuple[str, List[Dict]]]) -> Tuple[List[Dict], Dict[str, Any]]:
    """Merge multiple document collections.
    
    Args:
        collections: List of (collection_name, documents) tuples
        
    Returns:
        Tuple of (merged_documents, merge_metadata)
    """
    merged_docs = []
    merge_meta = {
        'source_collections': {},
        'total_documents': 0,
        'documents_per_collection': {}
    }
    
    for collection_name, docs in collections:
        start_idx = len(merged_docs)
        
        # Add source information to each document
        for doc in docs:
            doc_with_source = doc.copy()
            doc_with_source['_source_collection'] = collection_name
            doc_with_source['_original_idx'] = len(merged_docs) - start_idx
            merged_docs.append(doc_with_source)
        
        # Update metadata
        merge_meta['source_collections'][collection_name] = {
            'start_idx': start_idx,
            'end_idx': len(merged_docs),
            'count': len(docs)
        }
        merge_meta['documents_per_collection'][collection_name] = len(docs)
    
    merge_meta['total_documents'] = len(merged_docs)
    
    return merged_docs, merge_meta


def validate_document_format(doc: Dict) -> List[str]:
    """Validate that a document follows the expected format.
    
    Args:
        doc: Document to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    if 'vertexSet' not in doc:
        errors.append("Missing 'vertexSet' field")
    elif not isinstance(doc['vertexSet'], list):
        errors.append("'vertexSet' must be a list")
    
    if 'labels' not in doc:
        errors.append("Missing 'labels' field")
    elif not isinstance(doc['labels'], list):
        errors.append("'labels' must be a list")
    
    # Validate entities
    if 'vertexSet' in doc:
        for ent_idx, entity_cluster in enumerate(doc['vertexSet']):
            if not isinstance(entity_cluster, list):
                errors.append(f"Entity cluster {ent_idx} must be a list")
                continue
            
            for mention_idx, mention in enumerate(entity_cluster):
                if not isinstance(mention, dict):
                    errors.append(f"Entity mention {ent_idx}.{mention_idx} must be a dict")
                    continue
                
                if 'name' not in mention:
                    errors.append(f"Entity mention {ent_idx}.{mention_idx} missing 'name'")
                
                if 'type' not in mention:
                    errors.append(f"Entity mention {ent_idx}.{mention_idx} missing 'type'")
    
    # Validate relations
    if 'labels' in doc:
        n_entities = len(doc.get('vertexSet', []))
        
        for rel_idx, rel in enumerate(doc['labels']):
            if not isinstance(rel, dict):
                errors.append(f"Relation {rel_idx} must be a dict")
                continue
            
            # Check required fields
            for field in ['h', 't', 'r']:
                if field not in rel:
                    errors.append(f"Relation {rel_idx} missing '{field}'")
            
            # Check entity indices
            if 'h' in rel and not (0 <= rel['h'] < n_entities):
                errors.append(f"Relation {rel_idx} head index out of range")
            
            if 't' in rel and not (0 <= rel['t'] < n_entities):
                errors.append(f"Relation {rel_idx} tail index out of range")
    
    return errors


def create_sample_document() -> Dict:
    """Create a sample document for testing purposes.
    
    Returns:
        Sample document dictionary
    """
    return {
        'title': 'Sample Document',
        'vertexSet': [
            [{'name': 'Apple Inc.', 'type': 'ORG', 'sent_id': 0, 'pos': [0, 9]}],
            [{'name': 'Tim Cook', 'type': 'PER', 'sent_id': 1, 'pos': [0, 8]}],
            [{'name': 'California', 'type': 'LOC', 'sent_id': 1, 'pos': [20, 30]}],
        ],
        'labels': [
            {'h': 1, 't': 0, 'r': 'P108', 'evidence': [1]},  # Tim Cook works for Apple
            {'h': 0, 't': 2, 'r': 'P131', 'evidence': [1]},  # Apple located in California
        ],
        'sents': [
            ['Apple', 'Inc.', 'is', 'a', 'technology', 'company', '.'],
            ['Tim', 'Cook', 'is', 'the', 'CEO', 'and', 'works', 'in', 'California', '.']
        ]
    }


if __name__ == "__main__":
    # Example usage
    sample_doc = create_sample_document()
    
    # Validate document
    errors = validate_document_format(sample_doc)
    if errors:
        print("Validation errors:", errors)
    else:
        print("Document is valid")
    
    # Compute statistics
    stats = compute_document_statistics([sample_doc])
    print("Document statistics:", stats)
