"""
Wrapper for the original entity resolution approach to make it compatible with evaluation.
"""

from typing import Dict, List, Tuple, Any
from entity_resolution import EntityResolver


class OriginalApproachWrapper:
    """Wrapper for your original entity resolution approach."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize wrapper.
        
        Args:
            similarity_threshold: Similarity threshold for duplicate detection
        """
        self.resolver = EntityResolver(similarity_threshold=similarity_threshold)
    
    def find_duplicate_entities(self, docs: List[Dict]) -> Dict[str, Any]:
        """Find duplicate entities using your original approach.
        
        Args:
            docs: List of documents
            
        Returns:
            Results in standard format
        """
        # Add doc_idx to documents if not present
        for i, doc in enumerate(docs):
            if 'doc_idx' not in doc:
                doc['doc_idx'] = i
        
        # Use your original approach
        duplicates_dict = self.resolver.find_duplicate_entities(docs)
        
        # Convert to standard format
        duplicates = []
        
        for surface_form, mentions in duplicates_dict.items():
            if len(mentions) >= 2:
                # Create pairs from mentions
                for i in range(len(mentions)):
                    for j in range(i + 1, len(mentions)):
                        mention1 = mentions[i]
                        mention2 = mentions[j]
                        
                        duplicates.append({
                            'entity1': {
                                'name': mention1['name'],
                                'doc_idx': mention1['doc_idx'],
                                'ent_idx': mention1['entity_idx'],
                                'type': mention1.get('type', 'UNK')
                            },
                            'entity2': {
                                'name': mention2['name'],
                                'doc_idx': mention2['doc_idx'],
                                'ent_idx': mention2['entity_idx'],
                                'type': mention2.get('type', 'UNK')
                            },
                            'surface_form': surface_form,
                            'method': 'original_approach'
                        })
        
        return {
            'duplicates': duplicates,
            'method': 'original_approach',
            'surface_forms_analyzed': len(duplicates_dict),
            'total_mentions': sum(len(mentions) for mentions in duplicates_dict.values())
        }


if __name__ == "__main__":
    # Test the wrapper
    from utils import create_sample_document
    
    # Create test documents
    test_docs = [
        create_sample_document(),
        {
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG'}],  # Should match "Apple Inc."
                [{'name': 'Timothy Cook', 'type': 'PER'}],  # Should match "Tim Cook"
                [{'name': 'California', 'type': 'LOC'}]  # Case match
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},
                {'h': 0, 't': 2, 'r': 'P131'}
            ]
        }
    ]
    
    # Test wrapper
    wrapper = OriginalApproachWrapper()
    results = wrapper.find_duplicate_entities(test_docs)
    
    print("Original Approach Wrapper Results:")
    print(f"Found {len(results['duplicates'])} duplicate pairs")
    
    for dup in results['duplicates']:
        print(f"- {dup['entity1']['name']} <-> {dup['entity2']['name']} "
              f"(surface_form: {dup['surface_form']})")
