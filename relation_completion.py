"""
Relation completion module for knowledge graph healing.
Discovers missing relations between entities using graph patterns and inference rules.
"""

import re
import json
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
import networkx as nx
from itertools import combinations


class RelationCompleter:
    """Relation completion for missing relation discovery."""
    
    def __init__(self, confidence_threshold: float = 0.5, max_path_length: int = 3):
        """Initialize the relation completer.
        
        Args:
            confidence_threshold: Minimum confidence for accepting new relations
            max_path_length: Maximum path length for relation inference
        """
        self.confidence_threshold = confidence_threshold
        self.max_path_length = max_path_length
        self.relation_patterns = {}
        self.transitivity_rules = self._load_transitivity_rules()
        
    def _load_transitivity_rules(self) -> Dict[Tuple[str, str], str]:
        """Load transitivity rules for relation inference.
        
        Returns:
            Dictionary mapping relation pairs to inferred relations
        """
        # Common transitivity patterns in DocRED
        rules = {
            # Geographic relations
            ('P131', 'P131'): 'P131',  # located_in -> located_in = located_in
            ('P17', 'P131'): 'P17',   # country -> located_in = country
            
            # Organizational relations  
            ('P108', 'P749'): 'P108',  # employer -> parent_org = employer
            ('P463', 'P463'): 'P463',  # member_of -> member_of = member_of
            
            # Family relations
            ('P22', 'P22'): 'P1038',   # father -> father = relative
            ('P25', 'P25'): 'P1038',   # mother -> mother = relative
            ('P3373', 'P3373'): 'P1038', # sibling -> sibling = relative
            
            # Educational relations
            ('P69', 'P749'): 'P69',    # educated_at -> parent_org = educated_at
            
            # Temporal relations
            ('P155', 'P155'): 'P155',  # follows -> follows = follows (transitivity)
            ('P156', 'P156'): 'P156',  # followed_by -> followed_by = followed_by
        }
        
        return rules
    
    def extract_relation_patterns(self, docs: List[Dict]) -> Dict[str, Any]:
        """Extract common relation patterns from documents.
        
        Args:
            docs: List of documents to analyze
            
        Returns:
            Dictionary of extracted patterns and statistics
        """
        patterns = {
            'entity_type_pairs': defaultdict(Counter),  # (ent1_type, rel, ent2_type)
            'relation_chains': defaultdict(list),       # relation -> [chain_patterns]
            'co_occurrence': defaultdict(Counter),      # entity_pair -> relations
            'relation_stats': Counter()                 # relation -> frequency
        }
        
        for doc in docs:
            entities = doc.get('vertexSet', [])
            relations = doc.get('labels', [])
            
            # Build entity type mapping
            entity_types = {}
            for i, entity_cluster in enumerate(entities):
                if entity_cluster:
                    entity_types[i] = entity_cluster[0].get('type', 'UNK')
            
            # Process relations
            for rel in relations:
                rel_type = rel['r']
                head_idx = rel['h']
                tail_idx = rel['t']
                
                patterns['relation_stats'][rel_type] += 1
                
                # Type pattern
                head_type = entity_types.get(head_idx, 'UNK')
                tail_type = entity_types.get(tail_idx, 'UNK')
                type_pattern = (head_type, rel_type, tail_type)
                patterns['entity_type_pairs'][rel_type][type_pattern] += 1
                
                # Co-occurrence pattern
                entity_pair = tuple(sorted([head_idx, tail_idx]))
                patterns['co_occurrence'][entity_pair][rel_type] += 1
        
        # Find relation chains (paths of length 2)
        for doc in docs:
            graph = self._build_document_graph(doc)
            patterns['relation_chains'].update(
                self._find_relation_chains(graph, max_length=2)
            )
        
        return patterns
    
    def _build_document_graph(self, doc: Dict) -> nx.MultiDiGraph:
        """Build a directed multigraph from document relations.
        
        Args:
            doc: Document dictionary
            
        Returns:
            NetworkX directed multigraph
        """
        graph = nx.MultiDiGraph()
        
        # Add entity nodes
        entities = doc.get('vertexSet', [])
        for i, entity_cluster in enumerate(entities):
            if entity_cluster:
                entity_type = entity_cluster[0].get('type', 'UNK')
                entity_name = entity_cluster[0].get('name', f'entity_{i}')
                graph.add_node(i, type=entity_type, name=entity_name)
        
        # Add relation edges
        relations = doc.get('labels', [])
        for rel in relations:
            graph.add_edge(
                rel['h'], 
                rel['t'], 
                relation=rel['r'],
                evidence=rel.get('evidence', [])
            )
        
        return graph
    
    def _find_relation_chains(self, graph: nx.MultiDiGraph, max_length: int = 2) -> Dict[str, List[Tuple]]:
        """Find relation chains in the graph.
        
        Args:
            graph: Document graph
            max_length: Maximum chain length
            
        Returns:
            Dictionary mapping relation types to chain patterns
        """
        chains = defaultdict(list)
        
        # Find paths of length 2 (A -> B -> C)
        for node in graph.nodes():
            for path in nx.single_source_shortest_path(graph, node, cutoff=max_length):
                if len(path) == 3:  # Path length 2
                    source, intermediate, target = path
                    
                    # Get relations on the path
                    if graph.has_edge(source, intermediate) and graph.has_edge(intermediate, target):
                        rel1_data = list(graph[source][intermediate].values())[0]
                        rel2_data = list(graph[intermediate][target].values())[0]
                        
                        rel1 = rel1_data['relation']
                        rel2 = rel2_data['relation']
                        
                        # Check for potential transitivity
                        if (rel1, rel2) in self.transitivity_rules:
                            inferred_rel = self.transitivity_rules[(rel1, rel2)]
                            chains[inferred_rel].append((source, intermediate, target, rel1, rel2))
        
        return chains
    
    def find_missing_relations(self, docs: List[Dict]) -> Dict[str, Any]:
        """Find missing relations in documents using various strategies.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary of missing relations and completion statistics
        """
        # Extract patterns first
        patterns = self.extract_relation_patterns(docs)
        
        missing_relations = {
            'transitivity_based': [],
            'pattern_based': [],
            'type_consistency': [],
            'statistics': {
                'total_candidates': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        }
        
        for doc_idx, doc in enumerate(docs):
            doc_graph = self._build_document_graph(doc)
            
            # Strategy 1: Transitivity-based completion
            transitivity_candidates = self._find_transitivity_candidates(doc_graph, doc_idx)
            missing_relations['transitivity_based'].extend(transitivity_candidates)
            
            # Strategy 2: Pattern-based completion
            pattern_candidates = self._find_pattern_candidates(doc_graph, patterns, doc_idx)
            missing_relations['pattern_based'].extend(pattern_candidates)
            
            # Strategy 3: Type consistency completion
            type_candidates = self._find_type_consistency_candidates(doc_graph, patterns, doc_idx)
            missing_relations['type_consistency'].extend(type_candidates)
        
        # Compute statistics
        all_candidates = (
            missing_relations['transitivity_based'] + 
            missing_relations['pattern_based'] + 
            missing_relations['type_consistency']
        )
        
        missing_relations['statistics']['total_candidates'] = len(all_candidates)
        
        for candidate in all_candidates:
            confidence = candidate.get('confidence', 0.0)
            if confidence >= 0.8:
                missing_relations['statistics']['high_confidence'] += 1
            elif confidence >= 0.5:
                missing_relations['statistics']['medium_confidence'] += 1
            else:
                missing_relations['statistics']['low_confidence'] += 1
        
        return missing_relations
    
    def _find_transitivity_candidates(self, graph: nx.MultiDiGraph, doc_idx: int) -> List[Dict]:
        """Find relation candidates based on transitivity rules.
        
        Args:
            graph: Document graph
            doc_idx: Document index
            
        Returns:
            List of candidate relations
        """
        candidates = []
        
        # Look for 2-hop paths that could be completed
        for source in graph.nodes():
            for target in graph.nodes():
                if source == target:
                    continue
                
                # Check if direct relation already exists
                if graph.has_edge(source, target):
                    continue
                
                # Find intermediate nodes
                for intermediate in graph.nodes():
                    if (intermediate == source or intermediate == target):
                        continue
                    
                    if (graph.has_edge(source, intermediate) and 
                        graph.has_edge(intermediate, target)):
                        
                        # Get relations
                        rel1_data = list(graph[source][intermediate].values())[0]
                        rel2_data = list(graph[intermediate][target].values())[0]
                        
                        rel1 = rel1_data['relation']
                        rel2 = rel2_data['relation']
                        
                        # Check transitivity rule
                        if (rel1, rel2) in self.transitivity_rules:
                            inferred_rel = self.transitivity_rules[(rel1, rel2)]
                            
                            # Compute confidence based on rule reliability
                            confidence = 0.8 if (rel1, rel2) in [
                                ('P131', 'P131'), ('P17', 'P131'), ('P108', 'P749')
                            ] else 0.6
                            
                            candidate = {
                                'doc_idx': doc_idx,
                                'head': source,
                                'tail': target,
                                'relation': inferred_rel,
                                'confidence': confidence,
                                'method': 'transitivity',
                                'evidence': {
                                    'intermediate': intermediate,
                                    'path_relations': [rel1, rel2],
                                    'rule': f"{rel1} + {rel2} → {inferred_rel}"
                                }
                            }
                            candidates.append(candidate)
        
        return candidates
    
    def _find_pattern_candidates(self, graph: nx.MultiDiGraph, patterns: Dict, doc_idx: int) -> List[Dict]:
        """Find relation candidates based on learned patterns.
        
        Args:
            graph: Document graph
            patterns: Extracted relation patterns
            doc_idx: Document index
            
        Returns:
            List of candidate relations
        """
        candidates = []
        
        # Look for entity pairs that match common type patterns
        for source in graph.nodes():
            for target in graph.nodes():
                if source == target or graph.has_edge(source, target):
                    continue
                
                source_type = graph.nodes[source].get('type', 'UNK')
                target_type = graph.nodes[target].get('type', 'UNK')
                
                # Check patterns for this type pair
                for relation, type_patterns in patterns['entity_type_pairs'].items():
                    for (h_type, r_type, t_type), count in type_patterns.items():
                        if (h_type == source_type and t_type == target_type and 
                            r_type == relation and count >= 2):  # Minimum support
                            
                            # Compute confidence based on pattern frequency
                            total_patterns = sum(type_patterns.values())
                            pattern_confidence = count / total_patterns
                            
                            if pattern_confidence >= 0.3:  # Minimum confidence
                                candidate = {
                                    'doc_idx': doc_idx,
                                    'head': source,
                                    'tail': target,
                                    'relation': relation,
                                    'confidence': min(0.7, pattern_confidence),
                                    'method': 'pattern_matching',
                                    'evidence': {
                                        'pattern': (h_type, r_type, t_type),
                                        'pattern_count': count,
                                        'pattern_confidence': pattern_confidence
                                    }
                                }
                                candidates.append(candidate)
        
        return candidates
    
    def _find_type_consistency_candidates(self, graph: nx.MultiDiGraph, patterns: Dict, doc_idx: int) -> List[Dict]:
        """Find candidates based on entity type consistency.
        
        Args:
            graph: Document graph
            patterns: Extracted patterns
            doc_idx: Document index
            
        Returns:
            List of candidate relations
        """
        candidates = []
        
        # Simple heuristic: if two entities of specific types appear together
        # and commonly have certain relations, suggest them
        type_relation_rules = {
            ('PER', 'ORG'): ['P108', 'P463'],  # person-org: works_for, member_of
            ('PER', 'LOC'): ['P19', 'P20', 'P27'],  # person-location: birth_place, death_place, citizenship
            ('ORG', 'LOC'): ['P131', 'P159'],  # org-location: located_in, headquarters
            ('PER', 'PER'): ['P26', 'P22', 'P25', 'P3373'],  # person-person: spouse, father, mother, sibling
        }
        
        for source in graph.nodes():
            for target in graph.nodes():
                if source == target or graph.has_edge(source, target):
                    continue
                
                source_type = graph.nodes[source].get('type', 'UNK')
                target_type = graph.nodes[target].get('type', 'UNK')
                
                # Check if this type pair has common relations
                type_pair = (source_type, target_type)
                if type_pair in type_relation_rules:
                    for relation in type_relation_rules[type_pair]:
                        # Check if this relation is common for this type pair
                        if relation in patterns['relation_stats']:
                            rel_frequency = patterns['relation_stats'][relation]
                            if rel_frequency >= 5:  # Minimum frequency
                                
                                confidence = min(0.5, rel_frequency / 100.0)
                                
                                candidate = {
                                    'doc_idx': doc_idx,
                                    'head': source,
                                    'tail': target,
                                    'relation': relation,
                                    'confidence': confidence,
                                    'method': 'type_consistency',
                                    'evidence': {
                                        'type_pair': type_pair,
                                        'relation_frequency': rel_frequency,
                                        'rule': f"{source_type}-{target_type} commonly have {relation}"
                                    }
                                }
                                candidates.append(candidate)
        
        return candidates
    
    def apply_relation_completion(self, docs: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Apply relation completion to documents.
        
        Args:
            docs: List of documents
            
        Returns:
            Tuple of (completed_docs, completion_statistics)
        """
        # Find missing relations
        missing_relations = self.find_missing_relations(docs)
        
        # Filter by confidence threshold
        high_confidence_relations = []
        for category in ['transitivity_based', 'pattern_based', 'type_consistency']:
            for candidate in missing_relations[category]:
                if candidate['confidence'] >= self.confidence_threshold:
                    high_confidence_relations.append(candidate)
        
        # Apply completions to documents
        completed_docs = []
        relations_added = 0
        
        for doc_idx, doc in enumerate(docs):
            completed_doc = doc.copy()
            
            # Get relations to add for this document
            doc_relations = [r for r in high_confidence_relations if r['doc_idx'] == doc_idx]
            
            if doc_relations:
                # Add new relations to labels
                if 'labels' not in completed_doc:
                    completed_doc['labels'] = []
                
                original_labels = completed_doc['labels'].copy()
                
                for rel_candidate in doc_relations:
                    new_relation = {
                        'h': rel_candidate['head'],
                        't': rel_candidate['tail'],
                        'r': rel_candidate['relation'],
                        'evidence': [],  # Could be improved with evidence detection
                        'confidence': rel_candidate['confidence'],
                        'completion_method': rel_candidate['method'],
                        'completion_evidence': rel_candidate['evidence']
                    }
                    completed_doc['labels'].append(new_relation)
                    relations_added += 1
                
                # Add metadata about completion
                completed_doc['completion_metadata'] = {
                    'original_relations': len(original_labels),
                    'added_relations': len(doc_relations),
                    'completion_methods': list(set(r['method'] for r in doc_relations))
                }
            
            completed_docs.append(completed_doc)
        
        # Create completion statistics
        stats = {
            'documents_processed': len(docs),
            'total_candidates_found': missing_relations['statistics']['total_candidates'],
            'high_confidence_candidates': missing_relations['statistics']['high_confidence'],
            'relations_added': relations_added,
            'completion_rate': relations_added / max(1, missing_relations['statistics']['total_candidates']),
            'methods_used': {
                'transitivity_based': len([r for r in high_confidence_relations if r['method'] == 'transitivity']),
                'pattern_based': len([r for r in high_confidence_relations if r['method'] == 'pattern_matching']),
                'type_consistency': len([r for r in high_confidence_relations if r['method'] == 'type_consistency'])
            }
        }
        
        return completed_docs, stats


if __name__ == "__main__":
    # Example usage
    completer = RelationCompleter()
    
    # Create sample data
    sample_docs = [
        {
            'vertexSet': [
                [{'name': 'John Doe', 'type': 'PER'}],      # 0
                [{'name': 'Apple Inc.', 'type': 'ORG'}],    # 1  
                [{'name': 'California', 'type': 'LOC'}],    # 2
                [{'name': 'United States', 'type': 'LOC'}]  # 3
            ],
            'labels': [
                {'h': 0, 't': 1, 'r': 'P108'},  # John works at Apple
                {'h': 1, 't': 2, 'r': 'P131'},  # Apple located in California
                {'h': 2, 't': 3, 'r': 'P131'},  # California located in US
            ]
        }
    ]
    
    # Find missing relations
    missing = completer.find_missing_relations(sample_docs)
    print(f"Found {missing['statistics']['total_candidates']} relation candidates")
    
    # Apply completion
    completed_docs, stats = completer.apply_relation_completion(sample_docs)
    print("Completion statistics:", stats)
