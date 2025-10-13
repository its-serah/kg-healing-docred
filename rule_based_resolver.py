"""
Rule-Based Entity Resolution.
Uses hand-crafted rules for detecting duplicate entities based on patterns,
abbreviations, and domain-specific knowledge.
"""

import re
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, Counter
from difflib import SequenceMatcher


class RuleBasedResolver:
    """Rule-based entity resolution using hand-crafted patterns."""
    
    def __init__(self):
        """Initialize the rule-based resolver."""
        self.company_suffixes = {
            'inc.', 'incorporated', 'corp.', 'corporation', 'company', 'co.',
            'ltd.', 'limited', 'llc', 'plc', 'gmbh', 'ag', 's.a.', 'spa'
        }
        
        self.person_titles = {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'president', 'ceo', 'cto', 'cfo'
        }
        
        self.location_suffixes = {
            'city', 'town', 'county', 'state', 'province', 'country', 'region'
        }
        
        # Common abbreviation patterns
        self.abbreviation_rules = {
            # Organizations
            'international business machines': 'ibm',
            'general electric': 'ge',
            'american telephone and telegraph': 'at&t',
            'united states': 'us',
            'united states of america': 'usa',
            'united kingdom': 'uk',
            'european union': 'eu',
            
            # Common words
            'university': 'univ',
            'department': 'dept',
            'association': 'assoc',
            'foundation': 'found',
            'institute': 'inst',
            'technology': 'tech',
            'international': 'intl'
        }
        
        # Reverse abbreviation mapping
        self.expansion_rules = {v: k for k, v in self.abbreviation_rules.items()}
        
    def normalize_entity_name(self, name: str, entity_type: str = None) -> str:
        """Normalize entity name using rules.
        
        Args:
            name: Original entity name
            entity_type: Entity type (ORG, PER, LOC, etc.)
            
        Returns:
            Normalized entity name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation
        normalized = re.sub(r'[,\.\!\?\;\:\(\)\[\]\"\']+', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Apply entity-type specific rules
        if entity_type == 'ORG':
            normalized = self._normalize_organization(normalized)
        elif entity_type == 'PER':
            normalized = self._normalize_person(normalized)
        elif entity_type == 'LOC':
            normalized = self._normalize_location(normalized)
        
        # Apply general abbreviation rules
        words = normalized.split()
        normalized_words = []
        
        for word in words:
            # Check for abbreviation expansions
            if word in self.expansion_rules:
                normalized_words.append(self.expansion_rules[word])
            elif word in self.abbreviation_rules:
                normalized_words.append(self.abbreviation_rules[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _normalize_organization(self, name: str) -> str:
        """Normalize organization names."""
        words = name.split()
        filtered_words = []
        
        for word in words:
            # Remove common company suffixes
            if word not in self.company_suffixes:
                filtered_words.append(word)
        
        # Handle specific patterns
        result = ' '.join(filtered_words)
        
        # Handle "&" vs "and"
        result = result.replace('&', 'and')
        
        return result
    
    def _normalize_person(self, name: str) -> str:
        """Normalize person names."""
        words = name.split()
        filtered_words = []
        
        for word in words:
            # Remove titles
            if word not in self.person_titles:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _normalize_location(self, name: str) -> str:
        """Normalize location names."""
        words = name.split()
        filtered_words = []
        
        for word in words:
            # Remove location suffixes
            if word not in self.location_suffixes:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def apply_exact_match_rules(self, entity1: Dict, entity2: Dict) -> bool:
        """Apply exact match rules.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities match exactly
        """
        name1 = self.normalize_entity_name(entity1['name'], entity1.get('type'))
        name2 = self.normalize_entity_name(entity2['name'], entity2.get('type'))
        
        return name1 == name2 and entity1.get('type') == entity2.get('type')
    
    def apply_substring_rules(self, entity1: Dict, entity2: Dict) -> bool:
        """Apply substring matching rules.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if one name is substring of another
        """
        name1 = self.normalize_entity_name(entity1['name'], entity1.get('type'))
        name2 = self.normalize_entity_name(entity2['name'], entity2.get('type'))
        
        # Same type required
        if entity1.get('type') != entity2.get('type'):
            return False
        
        # One name contains the other (with minimum length)
        if len(name1) >= 3 and len(name2) >= 3:
            return name1 in name2 or name2 in name1
        
        return False
    
    def apply_abbreviation_rules(self, entity1: Dict, entity2: Dict) -> bool:
        """Apply abbreviation matching rules.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities match via abbreviation
        """
        name1 = entity1['name'].lower().strip()
        name2 = entity2['name'].lower().strip()
        
        # Same type required
        if entity1.get('type') != entity2.get('type'):
            return False
        
        # Check if one is abbreviation of the other
        if self._is_abbreviation(name1, name2) or self._is_abbreviation(name2, name1):
            return True
        
        return False
    
    def _is_abbreviation(self, short: str, long: str) -> bool:
        """Check if short is an abbreviation of long.
        
        Args:
            short: Potential abbreviation
            long: Full form
            
        Returns:
            True if short is abbreviation of long
        """
        # Remove punctuation and split
        short_clean = re.sub(r'[^\w\s]', '', short).lower()
        long_clean = re.sub(r'[^\w\s]', '', long).lower()
        
        short_words = short_clean.split()
        long_words = long_clean.split()
        
        # Simple acronym check
        if len(short_words) == 1 and len(short_words[0]) <= 5:
            acronym = ''.join(word[0] for word in long_words if word)
            if short_words[0] == acronym:
                return True
        
        # Initial matching
        if len(short_words) <= len(long_words):
            matches = 0
            for short_word in short_words:
                for long_word in long_words:
                    if long_word.startswith(short_word) or short_word in long_word:
                        matches += 1
                        break
            
            return matches == len(short_words)
        
        return False
    
    def apply_fuzzy_rules(self, entity1: Dict, entity2: Dict, threshold: float = 0.8) -> bool:
        """Apply fuzzy string matching rules.
        
        Args:
            entity1: First entity
            entity2: Second entity
            threshold: Similarity threshold
            
        Returns:
            True if entities match via fuzzy matching
        """
        name1 = self.normalize_entity_name(entity1['name'], entity1.get('type'))
        name2 = self.normalize_entity_name(entity2['name'], entity2.get('type'))
        
        # Same type required
        if entity1.get('type') != entity2.get('type'):
            return False
        
        # Compute similarity
        similarity = SequenceMatcher(None, name1, name2).ratio()
        
        return similarity >= threshold
    
    def apply_domain_specific_rules(self, entity1: Dict, entity2: Dict) -> bool:
        """Apply domain-specific matching rules.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities match via domain rules
        """
        name1 = entity1['name'].lower()
        name2 = entity2['name'].lower()
        entity_type = entity1.get('type')
        
        # Same type required
        if entity1.get('type') != entity2.get('type'):
            return False
        
        if entity_type == 'ORG':
            # Company variations
            variations = [
                ('apple inc', 'apple'),
                ('microsoft corp', 'microsoft'),
                ('google inc', 'google'),
                ('facebook inc', 'facebook'),
                ('meta platforms', 'facebook'),
                ('alphabet inc', 'google'),
                ('international business machines', 'ibm'),
            ]
            
            for var1, var2 in variations:
                if (var1 in name1 and var2 in name2) or (var2 in name1 and var1 in name2):
                    return True
        
        elif entity_type == 'PER':
            # Person name variations
            # Remove middle initials/names for comparison
            name1_parts = name1.split()
            name2_parts = name2.split()
            
            if len(name1_parts) >= 2 and len(name2_parts) >= 2:
                # Compare first and last names
                first1, last1 = name1_parts[0], name1_parts[-1]
                first2, last2 = name2_parts[0], name2_parts[-1]
                
                if first1 == first2 and last1 == last2:
                    return True
                
                # Handle nicknames/variations
                nickname_rules = {
                    'william': ['bill', 'will'],
                    'robert': ['bob', 'rob'],
                    'richard': ['rick', 'dick'],
                    'james': ['jim', 'jimmy'],
                    'michael': ['mike', 'mick'],
                    'timothy': ['tim', 'timmy'],
                    'christopher': ['chris'],
                    'alexander': ['alex'],
                    'benjamin': ['ben'],
                    'samuel': ['sam'],
                }
                
                for full_name, nicknames in nickname_rules.items():
                    if ((first1 == full_name and first2 in nicknames) or 
                        (first2 == full_name and first1 in nicknames)):
                        if last1 == last2:
                            return True
        
        elif entity_type == 'LOC':
            # Location variations
            location_rules = [
                ('united states', 'usa'),
                ('united states', 'us'),
                ('united kingdom', 'uk'),
                ('new york city', 'nyc'),
                ('san francisco', 'sf'),
                ('los angeles', 'la'),
            ]
            
            for loc1, loc2 in location_rules:
                if (loc1 in name1 and loc2 in name2) or (loc2 in name1 and loc1 in name2):
                    return True
        
        return False
    
    def find_duplicate_entities(self, docs: List[Dict]) -> Dict[str, Any]:
        """Find duplicate entities using rule-based approach.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary with duplicate detection results
        """
        # Extract all entities
        all_entities = []
        for doc_idx, doc in enumerate(docs):
            entities = doc.get('vertexSet', [])
            for ent_idx, entity_cluster in enumerate(entities):
                if entity_cluster:
                    entity_data = entity_cluster[0].copy()
                    entity_data['doc_idx'] = doc_idx
                    entity_data['ent_idx'] = ent_idx
                    entity_data['global_idx'] = len(all_entities)
                    all_entities.append(entity_data)
        
        print(f"Analyzing {len(all_entities)} entities with rule-based approach...")
        
        # Apply rules to find duplicates
        duplicates = []
        rule_stats = {
            'exact_match': 0,
            'substring_match': 0,
            'abbreviation_match': 0,
            'fuzzy_match': 0,
            'domain_specific': 0
        }
        
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                entity1 = all_entities[i]
                entity2 = all_entities[j]
                
                matched = False
                match_rule = None
                confidence = 0.0
                
                # Apply rules in order of precedence
                if self.apply_exact_match_rules(entity1, entity2):
                    matched = True
                    match_rule = 'exact_match'
                    confidence = 1.0
                    rule_stats['exact_match'] += 1
                
                elif self.apply_domain_specific_rules(entity1, entity2):
                    matched = True
                    match_rule = 'domain_specific'
                    confidence = 0.95
                    rule_stats['domain_specific'] += 1
                
                elif self.apply_abbreviation_rules(entity1, entity2):
                    matched = True
                    match_rule = 'abbreviation_match'
                    confidence = 0.9
                    rule_stats['abbreviation_match'] += 1
                
                elif self.apply_substring_rules(entity1, entity2):
                    matched = True
                    match_rule = 'substring_match'
                    confidence = 0.85
                    rule_stats['substring_match'] += 1
                
                elif self.apply_fuzzy_rules(entity1, entity2, threshold=0.85):
                    matched = True
                    match_rule = 'fuzzy_match'
                    confidence = 0.8
                    rule_stats['fuzzy_match'] += 1
                
                if matched:
                    duplicates.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'rule': match_rule,
                        'confidence': confidence,
                        'method': 'rule_based'
                    })
        
        print(f"Found {len(duplicates)} duplicate pairs using rules:")
        for rule, count in rule_stats.items():
            print(f"  {rule}: {count}")
        
        return {
            'duplicates': duplicates,
            'entities': all_entities,
            'rule_statistics': rule_stats,
            'method': 'rule_based'
        }
    
    def resolve_entities(self, docs: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Complete entity resolution pipeline.
        
        Args:
            docs: List of documents
            
        Returns:
            Tuple of (resolved_docs, resolution_stats)
        """
        results = self.find_duplicate_entities(docs)
        
        # Create resolution statistics
        stats = {
            'method': 'rule_based',
            'total_entities': len(results['entities']),
            'duplicate_pairs_found': len(results['duplicates']),
            'rule_breakdown': results['rule_statistics'],
            'avg_confidence': sum(d['confidence'] for d in results['duplicates']) / max(1, len(results['duplicates']))
        }
        
        # Apply merging (simplified - just add metadata)
        resolved_docs = []
        for doc in docs:
            resolved_doc = doc.copy()
            resolved_doc['rule_based_resolution'] = {
                'duplicates_found': len([
                    d for d in results['duplicates']
                    if d['entity1']['doc_idx'] == docs.index(doc) or 
                       d['entity2']['doc_idx'] == docs.index(doc)
                ])
            }
            resolved_docs.append(resolved_doc)
        
        return resolved_docs, stats


if __name__ == "__main__":
    # Test the rule-based resolver
    from utils import create_sample_document
    
    # Create test documents with various duplicate patterns
    test_docs = [
        # Document 1: Original
        create_sample_document(),
        
        # Document 2: Exact duplicates
        {
            'vertexSet': [
                [{'name': 'Apple Inc.', 'type': 'ORG'}],  # Exact match
                [{'name': 'Tim Cook', 'type': 'PER'}],    # Exact match
                [{'name': 'Cupertino', 'type': 'LOC'}]
            ],
            'labels': [{'h': 1, 't': 0, 'r': 'P108'}]
        },
        
        # Document 3: Substring and abbreviation matches
        {
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG'}],           # Substring of "Apple Inc."
                [{'name': 'Timothy Cook', 'type': 'PER'}],     # Domain rule for Tim Cook
                [{'name': 'IBM', 'type': 'ORG'}]              # Abbreviation
            ],
            'labels': [{'h': 1, 't': 0, 'r': 'P108'}]
        },
        
        # Document 4: Domain-specific matches
        {
            'vertexSet': [
                [{'name': 'International Business Machines', 'type': 'ORG'}],  # Should match IBM
                [{'name': 'Bill Gates', 'type': 'PER'}],                       # Different person
                [{'name': 'USA', 'type': 'LOC'}]                               # Should match "United States"
            ],
            'labels': [{'h': 1, 't': 0, 'r': 'P108'}]
        }
    ]
    
    # Add document indices for tracking
    for i, doc in enumerate(test_docs):
        doc['doc_idx'] = i
    
    # Run rule-based resolution
    resolver = RuleBasedResolver()
    resolved_docs, stats = resolver.resolve_entities(test_docs)
    
    print("\nRule-Based Entity Resolution Results:")
    print(f"Statistics: {stats}")
    
    # Show found duplicates
    results = resolver.find_duplicate_entities(test_docs)
    print("\nFound duplicates:")
    for dup in results['duplicates']:
        print(f"- {dup['entity1']['name']} <-> {dup['entity2']['name']} "
              f"(rule: {dup['rule']}, confidence: {dup['confidence']:.2f})")
