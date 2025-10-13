"""
String Similarity Baseline for Entity Resolution.
Uses basic string similarity metrics (edit distance, Jaccard, etc.) 
to detect duplicate entities. This serves as a simple baseline approach.
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict
from difflib import SequenceMatcher
import re


class StringSimilarityResolver:
    """String similarity-based entity resolution."""
    
    def __init__(self, threshold: float = 0.8):
        """Initialize string similarity resolver.
        
        Args:
            threshold: Similarity threshold for duplicate detection
        """
        self.threshold = threshold
    
    def preprocess_string(self, text: str) -> str:
        """Preprocess string for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """Compute Jaccard similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaccard similarity score
        """
        # Character-level n-grams (bigrams)
        def get_bigrams(s):
            if len(s) < 2:
                return set([s])
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)
        
        intersection = bigrams1 & bigrams2
        union = bigrams1 | bigrams2
        
        if not union:
            return 1.0 if str1 == str2 else 0.0
        
        return len(intersection) / len(union)
    
    def edit_distance_similarity(self, str1: str, str2: str) -> float:
        """Compute edit distance-based similarity.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Normalized edit distance similarity (0-1)
        """
        if not str1 and not str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Use SequenceMatcher for edit distance-like similarity
        return SequenceMatcher(None, str1, str2).ratio()
    
    def longest_common_substring_similarity(self, str1: str, str2: str) -> float:
        """Compute longest common substring similarity.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            LCS-based similarity score
        """
        if not str1 or not str2:
            return 0.0
        
        # Dynamic programming to find LCS length
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])
                else:
                    dp[i][j] = 0
        
        # Normalize by average length
        avg_len = (len(str1) + len(str2)) / 2
        return max_len / avg_len if avg_len > 0 else 0.0
    
    def word_overlap_similarity(self, str1: str, str2: str) -> float:
        """Compute word-level overlap similarity.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Word overlap similarity score
        """
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 1.0 if str1 == str2 else 0.0
        
        return len(intersection) / len(union)
    
    def compute_combined_similarity(self, str1: str, str2: str) -> float:
        """Compute combined similarity using multiple metrics.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Combined similarity score
        """
        # Preprocess strings
        proc1 = self.preprocess_string(str1)
        proc2 = self.preprocess_string(str2)
        
        # Compute individual similarities
        jaccard = self.jaccard_similarity(proc1, proc2)
        edit_dist = self.edit_distance_similarity(proc1, proc2)
        lcs = self.longest_common_substring_similarity(proc1, proc2)
        word_overlap = self.word_overlap_similarity(proc1, proc2)
        
        # Weighted combination
        combined = (
            0.3 * jaccard +
            0.3 * edit_dist +
            0.2 * lcs +
            0.2 * word_overlap
        )
        
        return combined
    
    def find_duplicate_entities(self, docs: List[Dict]) -> Dict[str, Any]:
        """Find duplicate entities using string similarity.
        
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
        
        print(f"Computing string similarities for {len(all_entities)} entities...")
        
        # Find duplicates using string similarity
        duplicates = []
        similarity_metrics = {
            'jaccard': [],
            'edit_distance': [],
            'lcs': [],
            'word_overlap': [],
            'combined': []
        }
        
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                entity1 = all_entities[i]
                entity2 = all_entities[j]
                
                # Only compare entities of the same type
                if entity1.get('type') != entity2.get('type'):
                    continue
                
                name1 = entity1['name']
                name2 = entity2['name']
                
                # Compute similarities
                proc1 = self.preprocess_string(name1)
                proc2 = self.preprocess_string(name2)
                
                jaccard = self.jaccard_similarity(proc1, proc2)
                edit_dist = self.edit_distance_similarity(proc1, proc2)
                lcs = self.longest_common_substring_similarity(proc1, proc2)
                word_overlap = self.word_overlap_similarity(proc1, proc2)
                combined = self.compute_combined_similarity(name1, name2)
                
                # Store metrics for analysis
                similarity_metrics['jaccard'].append(jaccard)
                similarity_metrics['edit_distance'].append(edit_dist)
                similarity_metrics['lcs'].append(lcs)
                similarity_metrics['word_overlap'].append(word_overlap)
                similarity_metrics['combined'].append(combined)
                
                # Check if similarity exceeds threshold
                if combined >= self.threshold:
                    duplicates.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'similarities': {
                            'jaccard': jaccard,
                            'edit_distance': edit_dist,
                            'lcs': lcs,
                            'word_overlap': word_overlap,
                            'combined': combined
                        },
                        'method': 'string_similarity'
                    })
        
        print(f"Found {len(duplicates)} duplicate pairs above threshold {self.threshold}")
        
        # Compute average similarities
        avg_similarities = {}
        for metric, values in similarity_metrics.items():
            avg_similarities[metric] = sum(values) / len(values) if values else 0.0
        
        return {
            'duplicates': duplicates,
            'entities': all_entities,
            'average_similarities': avg_similarities,
            'threshold': self.threshold,
            'method': 'string_similarity'
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
            'method': 'string_similarity',
            'total_entities': len(results['entities']),
            'duplicate_pairs_found': len(results['duplicates']),
            'threshold_used': self.threshold,
            'average_similarities': results['average_similarities'],
            'avg_combined_similarity': sum(d['similarities']['combined'] for d in results['duplicates']) / max(1, len(results['duplicates']))
        }
        
        # Apply merging (simplified - just add metadata)
        resolved_docs = []
        for doc in docs:
            resolved_doc = doc.copy()
            resolved_doc['string_similarity_resolution'] = {
                'duplicates_found': len([
                    d for d in results['duplicates']
                    if d['entity1']['doc_idx'] == docs.index(doc) or 
                       d['entity2']['doc_idx'] == docs.index(doc)
                ])
            }
            resolved_docs.append(resolved_doc)
        
        return resolved_docs, stats
    
    def analyze_similarity_distribution(self, docs: List[Dict]) -> Dict[str, Any]:
        """Analyze the distribution of similarity scores.
        
        Args:
            docs: List of documents
            
        Returns:
            Analysis of similarity distributions
        """
        results = self.find_duplicate_entities(docs)
        
        # Collect all similarity scores
        all_similarities = {
            'jaccard': [],
            'edit_distance': [],
            'lcs': [],
            'word_overlap': [],
            'combined': []
        }
        
        for dup in results['duplicates']:
            for metric, score in dup['similarities'].items():
                all_similarities[metric].append(score)
        
        # Compute statistics for each metric
        analysis = {}
        for metric, scores in all_similarities.items():
            if scores:
                analysis[metric] = {
                    'min': min(scores),
                    'max': max(scores),
                    'mean': sum(scores) / len(scores),
                    'count': len(scores),
                    'above_threshold': sum(1 for s in scores if s >= self.threshold)
                }
            else:
                analysis[metric] = {
                    'min': 0, 'max': 0, 'mean': 0, 'count': 0, 'above_threshold': 0
                }
        
        return analysis


if __name__ == "__main__":
    # Test the string similarity resolver
    from utils import create_sample_document
    
    # Create test documents with various similarity patterns
    test_docs = [
        # Document 1: Original
        create_sample_document(),
        
        # Document 2: Exact and near-exact matches
        {
            'vertexSet': [
                [{'name': 'Apple Inc.', 'type': 'ORG'}],     # Exact match
                [{'name': 'Tim Cook', 'type': 'PER'}],       # Exact match
                [{'name': 'california', 'type': 'LOC'}]      # Case difference
            ],
            'labels': [{'h': 1, 't': 0, 'r': 'P108'}]
        },
        
        # Document 3: Partial matches
        {
            'vertexSet': [
                [{'name': 'Apple Corporation', 'type': 'ORG'}],  # Similar to Apple Inc.
                [{'name': 'Timothy Cook', 'type': 'PER'}],       # Similar to Tim Cook
                [{'name': 'San Francisco', 'type': 'LOC'}]       # Different location
            ],
            'labels': [{'h': 1, 't': 0, 'r': 'P108'}]
        },
        
        # Document 4: Low similarity
        {
            'vertexSet': [
                [{'name': 'Microsoft', 'type': 'ORG'}],       # Different company
                [{'name': 'Bill Gates', 'type': 'PER'}],      # Different person
                [{'name': 'Seattle', 'type': 'LOC'}]          # Different location
            ],
            'labels': [{'h': 1, 't': 0, 'r': 'P108'}]
        }
    ]
    
    # Add document indices for tracking
    for i, doc in enumerate(test_docs):
        doc['doc_idx'] = i
    
    # Test with different thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\n{'='*50}")
        print(f"Testing with threshold {threshold}")
        print('='*50)
        
        resolver = StringSimilarityResolver(threshold=threshold)
        resolved_docs, stats = resolver.resolve_entities(test_docs)
        
        print(f"Statistics: {stats}")
        
        # Show found duplicates
        results = resolver.find_duplicate_entities(test_docs)
        print(f"\nFound {len(results['duplicates'])} duplicates:")
        for dup in results['duplicates']:
            print(f"- {dup['entity1']['name']} <-> {dup['entity2']['name']}")
            print(f"  Combined similarity: {dup['similarities']['combined']:.3f}")
            print(f"  Breakdown: Jaccard={dup['similarities']['jaccard']:.3f}, "
                  f"EditDist={dup['similarities']['edit_distance']:.3f}, "
                  f"LCS={dup['similarities']['lcs']:.3f}, "
                  f"WordOverlap={dup['similarities']['word_overlap']:.3f}")
    
    # Analyze similarity distribution
    print(f"\n{'='*50}")
    print("Similarity Distribution Analysis")
    print('='*50)
    
    resolver = StringSimilarityResolver(threshold=0.5)  # Low threshold to capture all
    analysis = resolver.analyze_similarity_distribution(test_docs)
    
    for metric, stats in analysis.items():
        print(f"{metric.upper()}:")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Count: {stats['count']}")
        print(f"  Above 0.8 threshold: {stats['above_threshold']}")
        print()
