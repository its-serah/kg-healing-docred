"""
Entity resolution module for knowledge graph healing.
Detects and merges duplicate entities based on surface forms and context similarity.
"""

import re
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import networkx as nx


class EntityResolver:
    """Entity resolution for duplicate detection and merging."""
    
    def __init__(self, similarity_threshold: float = 0.8, type_weight: float = 0.3):
        """Initialize the entity resolver.
        
        Args:
            similarity_threshold: Minimum similarity score for considering entities as duplicates
            type_weight: Weight given to entity type similarity in overall score
        """
        self.similarity_threshold = similarity_threshold
        self.type_weight = type_weight
        
    def normalize_surface_form(self, surface_form: str) -> str:
        """Normalize surface form for better matching.
        
        Args:
            surface_form: Raw entity mention text
            
        Returns:
            Normalized surface form
        """
        # Convert to lowercase
        normalized = surface_form.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common punctuation
        normalized = re.sub(r'[,\.\!\?\;\:]', '', normalized)
        
        # Handle common abbreviations and expansions
        normalized = normalized.replace('inc.', 'incorporated')
        normalized = normalized.replace('corp.', 'corporation') 
        normalized = normalized.replace('co.', 'company')
        normalized = normalized.replace('ltd.', 'limited')
        normalized = normalized.replace('&', 'and')
        
        return normalized
        
    def compute_string_similarity(self, str1: str, str2: str) -> float:
        """Compute string similarity between two surface forms.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Exact match
        if str1 == str2:
            return 1.0
            
        # Substring match
        if str1 in str2 or str2 in str1:
            return 0.9
            
        # Sequence matching
        return SequenceMatcher(None, str1, str2).ratio()
    
    def compute_type_similarity(self, type1: str, type2: str) -> float:
        """Compute similarity between entity types.
        
        Args:
            type1: First entity type
            type2: Second entity type
            
        Returns:
            Type similarity score (1.0 for exact match, 0.0 for different types)
        """
        return 1.0 if type1 == type2 else 0.0
    
    def compute_context_similarity(self, mention1: Dict, mention2: Dict) -> float:
        """Compute contextual similarity between two entity mentions.
        
        Args:
            mention1: First entity mention
            mention2: Second entity mention
            
        Returns:
            Context similarity score between 0 and 1
        """
        # For now, use a simple heuristic based on sentence similarity
        # In a full implementation, this could use sentence embeddings
        
        sent_id1 = mention1.get('sent_id', -1)
        sent_id2 = mention2.get('sent_id', -1)
        
        # Same sentence gets high similarity
        if sent_id1 == sent_id2:
            return 1.0
        
        # Adjacent sentences get moderate similarity
        if abs(sent_id1 - sent_id2) == 1:
            return 0.7
        
        # Different sentences get lower similarity
        return max(0.0, 1.0 - abs(sent_id1 - sent_id2) * 0.1)
    
    def compute_entity_similarity(self, mention1: Dict, mention2: Dict) -> float:
        """Compute overall similarity between two entity mentions.
        
        Args:
            mention1: First entity mention
            mention2: Second entity mention
            
        Returns:
            Overall similarity score between 0 and 1
        """
        # Normalize surface forms
        norm1 = self.normalize_surface_form(mention1['name'])
        norm2 = self.normalize_surface_form(mention2['name'])
        
        # Compute component similarities
        string_sim = self.compute_string_similarity(norm1, norm2)
        type_sim = self.compute_type_similarity(mention1['type'], mention2['type'])
        context_sim = self.compute_context_similarity(mention1, mention2)
        
        # Weighted combination
        overall_sim = (
            (1 - self.type_weight) * string_sim + 
            self.type_weight * type_sim
        ) * (0.5 + 0.5 * context_sim)  # Context as a multiplier
        
        return overall_sim
    
    def find_duplicate_entities(self, docs: List[Dict]) -> Dict[str, List[Dict]]:
        """Find potential duplicate entities across documents.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary mapping surface forms to lists of duplicate entity mentions
        """
        # Group mentions by normalized surface form
        surface_form_groups = defaultdict(list)
        
        for doc_idx, doc in enumerate(docs):
            for ent_idx, entity_cluster in enumerate(doc.get('vertexSet', [])):
                for mention in entity_cluster:
                    mention_data = mention.copy()
                    mention_data['doc_idx'] = doc_idx
                    mention_data['entity_idx'] = ent_idx
                    mention_data['cluster_size'] = len(entity_cluster)
                    
                    # Use normalized surface form as key
                    norm_form = self.normalize_surface_form(mention['name'])
                    surface_form_groups[norm_form].append(mention_data)
        
        # Find groups with multiple entities (potential duplicates)
        duplicates = {}
        for norm_form, mentions in surface_form_groups.items():
            if len(mentions) > 1:
                # Check if mentions refer to different entity clusters
                unique_entities = set()
                for mention in mentions:
                    entity_id = (mention['doc_idx'], mention['entity_idx'])
                    unique_entities.add(entity_id)
                
                if len(unique_entities) > 1:
                    duplicates[norm_form] = mentions
        
        return duplicates
    
    def resolve_entity_cluster(self, mentions: List[Dict]) -> Dict[str, Any]:
        """Resolve a cluster of potentially duplicate entity mentions.
        
        Args:
            mentions: List of entity mentions that may refer to the same entity
            
        Returns:
            Resolution result with merged entity information
        """
        if not mentions:
            return {}
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                sim = self.compute_entity_similarity(mentions[i], mentions[j])
                similarities.append((i, j, sim))
        
        # Find groups of similar mentions
        # For simplicity, use a greedy clustering approach
        clusters = []
        used = set()
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, sim in similarities:
            if sim >= self.similarity_threshold:
                # Find or create cluster
                cluster_found = False
                for cluster in clusters:
                    if i in cluster or j in cluster:
                        cluster.update([i, j])
                        cluster_found = True
                        break
                
                if not cluster_found:
                    clusters.append(set([i, j]))
                    
                used.update([i, j])
        
        # Add singleton clusters for unused mentions
        for i in range(len(mentions)):
            if i not in used:
                clusters.append(set([i]))
        
        # Create resolution result
        resolution = {
            'original_mentions': mentions,
            'clusters': [],
            'n_clusters': len(clusters),
            'merge_confidence': 0.0
        }
        
        for cluster_indices in clusters:
            cluster_mentions = [mentions[i] for i in cluster_indices]
            
            # Choose canonical mention (e.g., most frequent or complete form)
            canonical = max(cluster_mentions, key=lambda m: len(m['name']))
            
            cluster_info = {
                'mentions': cluster_mentions,
                'canonical': canonical,
                'size': len(cluster_mentions),
                'confidence': sum(sim for _, _, sim in similarities if sim >= self.similarity_threshold) / max(1, len(similarities))
            }
            
            resolution['clusters'].append(cluster_info)
        
        # Compute overall merge confidence
        if len(clusters) < len(mentions):
            resolution['merge_confidence'] = 1.0 - (len(clusters) / len(mentions))
        
        return resolution
    
    def apply_entity_resolution(self, docs: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Apply entity resolution to a set of documents.
        
        Args:
            docs: List of documents to process
            
        Returns:
            Tuple of (resolved_docs, resolution_statistics)
        """
        # Find potential duplicates
        duplicates = self.find_duplicate_entities(docs)
        
        # Resolve each duplicate cluster
        resolutions = {}
        for surface_form, mentions in duplicates.items():
            resolution = self.resolve_entity_cluster(mentions)
            if resolution.get('n_clusters', 0) < len(mentions):
                resolutions[surface_form] = resolution
        
        # Create statistics
        stats = {
            'total_surface_forms_analyzed': len(duplicates),
            'surface_forms_with_merges': len(resolutions),
            'total_mentions_analyzed': sum(len(mentions) for mentions in duplicates.values()),
            'total_clusters_created': sum(res['n_clusters'] for res in resolutions.values()),
            'entities_merged': sum(len(res['original_mentions']) - res['n_clusters'] for res in resolutions.values()),
            'avg_merge_confidence': sum(res['merge_confidence'] for res in resolutions.values()) / max(1, len(resolutions))
        }
        
        # For demonstration, return original docs with resolution metadata
        # In a full implementation, you would modify the documents to merge entities
        resolved_docs = []
        for doc in docs:
            resolved_doc = doc.copy()
            resolved_doc['entity_resolutions'] = []
            
            # Add resolution information for entities in this document
            for surface_form, resolution in resolutions.items():
                doc_mentions = [m for m in resolution['original_mentions'] 
                              if m['doc_idx'] == resolved_doc.get('doc_idx', -1)]
                if doc_mentions:
                    resolved_doc['entity_resolutions'].append({
                        'surface_form': surface_form,
                        'mentions_in_doc': len(doc_mentions),
                        'resolution': resolution
                    })
            
            resolved_docs.append(resolved_doc)
        
        return resolved_docs, stats


class EntityLinker:
    """Links entities to external knowledge bases."""
    
    def __init__(self):
        """Initialize the entity linker."""
        # Placeholder for external KB connections
        self.external_kbs = ['wikidata', 'freebase', 'dbpedia']
    
    def link_entity(self, entity_mention: Dict) -> List[Dict]:
        """Link an entity mention to external knowledge bases.
        
        Args:
            entity_mention: Entity mention to link
            
        Returns:
            List of potential KB links with confidence scores
        """
        # Placeholder implementation
        # In practice, this would query external KBs
        links = []
        
        entity_name = entity_mention['name']
        entity_type = entity_mention['type']
        
        # Mock some links based on simple heuristics
        if entity_type == 'PER':
            links.append({
                'kb': 'wikidata',
                'entity_id': f'Q{hash(entity_name) % 1000000}',
                'entity_label': entity_name,
                'confidence': 0.8,
                'description': f'Person named {entity_name}'
            })
        elif entity_type == 'ORG':
            links.append({
                'kb': 'wikidata', 
                'entity_id': f'Q{hash(entity_name) % 1000000}',
                'entity_label': entity_name,
                'confidence': 0.7,
                'description': f'Organization named {entity_name}'
            })
        
        return links


if __name__ == "__main__":
    # Example usage
    resolver = EntityResolver()
    
    # Create sample data with potential duplicates
    sample_docs = [
        {
            'doc_idx': 0,
            'vertexSet': [
                [{'name': 'Apple Inc.', 'type': 'ORG', 'sent_id': 0}],
                [{'name': 'Tim Cook', 'type': 'PER', 'sent_id': 1}]
            ]
        },
        {
            'doc_idx': 1,
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG', 'sent_id': 0}],
                [{'name': 'Timothy Cook', 'type': 'PER', 'sent_id': 1}]
            ]
        }
    ]
    
    # Find duplicates
    duplicates = resolver.find_duplicate_entities(sample_docs)
    print(f"Found {len(duplicates)} potential duplicate surface forms")
    
    # Apply resolution
    resolved_docs, stats = resolver.apply_entity_resolution(sample_docs)
    print("Resolution statistics:", stats)
