"""
Main knowledge graph healer orchestrator.
Coordinates the entire healing pipeline including entity resolution and relation completion.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from data_loader import DocREDLoader
from entity_resolution import EntityResolver
from relation_completion import RelationCompleter


class KGHealer:
    """Main orchestrator for knowledge graph healing operations."""
    
    def __init__(self, 
                 entity_similarity_threshold: float = 0.8,
                 relation_confidence_threshold: float = 0.5,
                 enable_entity_resolution: bool = True,
                 enable_relation_completion: bool = True,
                 verbose: bool = True):
        """Initialize the KG healer.
        
        Args:
            entity_similarity_threshold: Threshold for entity duplicate detection
            relation_confidence_threshold: Threshold for relation completion
            enable_entity_resolution: Whether to perform entity resolution
            enable_relation_completion: Whether to perform relation completion
            verbose: Whether to print progress information
        """
        self.enable_entity_resolution = enable_entity_resolution
        self.enable_relation_completion = enable_relation_completion
        self.verbose = verbose
        
        # Initialize components
        self.entity_resolver = EntityResolver(
            similarity_threshold=entity_similarity_threshold
        ) if enable_entity_resolution else None
        
        self.relation_completer = RelationCompleter(
            confidence_threshold=relation_confidence_threshold
        ) if enable_relation_completion else None
        
        # Statistics tracking
        self.healing_stats = {}
        
    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[KGHealer] {message}")
    
    def analyze_document_quality(self, docs: List[Dict]) -> Dict[str, Any]:
        """Analyze the quality of documents before healing.
        
        Args:
            docs: List of documents to analyze
            
        Returns:
            Quality analysis results
        """
        analysis = {
            'total_documents': len(docs),
            'total_entities': 0,
            'total_relations': 0,
            'entity_types': {},
            'relation_types': {},
            'avg_entities_per_doc': 0,
            'avg_relations_per_doc': 0,
            'potential_issues': []
        }
        
        entity_type_counts = {}
        relation_type_counts = {}
        
        for doc in docs:
            # Count entities
            entities = doc.get('vertexSet', [])
            doc_entity_count = len(entities)
            analysis['total_entities'] += doc_entity_count
            
            # Count entity types
            for entity_cluster in entities:
                if entity_cluster:
                    entity_type = entity_cluster[0].get('type', 'UNK')
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            
            # Count relations
            relations = doc.get('labels', [])
            doc_relation_count = len(relations)
            analysis['total_relations'] += doc_relation_count
            
            # Count relation types
            for rel in relations:
                rel_type = rel['r']
                relation_type_counts[rel_type] = relation_type_counts.get(rel_type, 0) + 1
            
            # Check for potential issues
            if doc_entity_count == 0:
                analysis['potential_issues'].append(f"Document has no entities")
            if doc_relation_count == 0:
                analysis['potential_issues'].append(f"Document has no relations")
            if doc_relation_count > doc_entity_count * (doc_entity_count - 1):
                analysis['potential_issues'].append(f"Document may have too many relations")
        
        # Compute averages and distributions
        if analysis['total_documents'] > 0:
            analysis['avg_entities_per_doc'] = analysis['total_entities'] / analysis['total_documents']
            analysis['avg_relations_per_doc'] = analysis['total_relations'] / analysis['total_documents']
        
        analysis['entity_types'] = entity_type_counts
        analysis['relation_types'] = relation_type_counts
        
        return analysis
    
    def find_healing_opportunities(self, docs: List[Dict]) -> Dict[str, Any]:
        """Identify healing opportunities in the document set.
        
        Args:
            docs: List of documents to analyze
            
        Returns:
            Dictionary of healing opportunities
        """
        opportunities = {
            'entity_opportunities': {},
            'relation_opportunities': {},
            'quality_issues': []
        }
        
        # Entity resolution opportunities
        if self.enable_entity_resolution and self.entity_resolver:
            self._log("Analyzing entity resolution opportunities...")
            duplicates = self.entity_resolver.find_duplicate_entities(docs)
            opportunities['entity_opportunities'] = {
                'potential_duplicates': len(duplicates),
                'duplicate_surface_forms': list(duplicates.keys())[:10],  # Sample
                'total_duplicate_mentions': sum(len(mentions) for mentions in duplicates.values())
            }
        
        # Relation completion opportunities
        if self.enable_relation_completion and self.relation_completer:
            self._log("Analyzing relation completion opportunities...")
            missing_relations = self.relation_completer.find_missing_relations(docs)
            opportunities['relation_opportunities'] = {
                'total_candidates': missing_relations['statistics']['total_candidates'],
                'high_confidence_candidates': missing_relations['statistics']['high_confidence'],
                'completion_methods': {
                    'transitivity': len(missing_relations['transitivity_based']),
                    'pattern_matching': len(missing_relations['pattern_based']),
                    'type_consistency': len(missing_relations['type_consistency'])
                }
            }
        
        return opportunities
    
    def heal_documents(self, docs: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Apply complete healing pipeline to documents.
        
        Args:
            docs: List of documents to heal
            
        Returns:
            Tuple of (healed_documents, healing_statistics)
        """
        start_time = time.time()
        
        self._log(f"Starting healing pipeline for {len(docs)} documents...")
        
        # Initial quality analysis
        self._log("Analyzing initial document quality...")
        initial_quality = self.analyze_document_quality(docs)
        
        # Find healing opportunities
        opportunities = self.find_healing_opportunities(docs)
        
        # Initialize healing statistics
        healing_stats = {
            'initial_quality': initial_quality,
            'opportunities': opportunities,
            'entity_resolution': {},
            'relation_completion': {},
            'final_quality': {},
            'processing_time': 0,
            'improvements': {}
        }
        
        healed_docs = docs.copy()
        
        # Step 1: Entity Resolution
        if self.enable_entity_resolution and self.entity_resolver:
            self._log("Applying entity resolution...")
            entity_start = time.time()
            
            healed_docs, entity_stats = self.entity_resolver.apply_entity_resolution(healed_docs)
            
            healing_stats['entity_resolution'] = entity_stats
            healing_stats['entity_resolution']['processing_time'] = time.time() - entity_start
            
            self._log(f"Entity resolution completed: {entity_stats.get('entities_merged', 0)} entities merged")
        
        # Step 2: Relation Completion  
        if self.enable_relation_completion and self.relation_completer:
            self._log("Applying relation completion...")
            relation_start = time.time()
            
            healed_docs, relation_stats = self.relation_completer.apply_relation_completion(healed_docs)
            
            healing_stats['relation_completion'] = relation_stats
            healing_stats['relation_completion']['processing_time'] = time.time() - relation_start
            
            self._log(f"Relation completion completed: {relation_stats.get('relations_added', 0)} relations added")
        
        # Final quality analysis
        self._log("Analyzing final document quality...")
        final_quality = self.analyze_document_quality(healed_docs)
        healing_stats['final_quality'] = final_quality
        
        # Compute improvements
        improvements = self._compute_improvements(initial_quality, final_quality, healing_stats)
        healing_stats['improvements'] = improvements
        
        # Total processing time
        healing_stats['processing_time'] = time.time() - start_time
        
        self._log(f"Healing pipeline completed in {healing_stats['processing_time']:.2f} seconds")
        self._log(f"Improvements: {improvements}")
        
        return healed_docs, healing_stats
    
    def _compute_improvements(self, initial: Dict, final: Dict, healing_stats: Dict) -> Dict[str, Any]:
        """Compute improvements made during healing.
        
        Args:
            initial: Initial quality metrics
            final: Final quality metrics
            healing_stats: Healing statistics
            
        Returns:
            Dictionary of improvements
        """
        improvements = {
            'entities_merged': healing_stats.get('entity_resolution', {}).get('entities_merged', 0),
            'relations_added': healing_stats.get('relation_completion', {}).get('relations_added', 0),
            'relation_increase_pct': 0,
            'avg_relations_per_doc_increase': 0,
            'quality_score_improvement': 0
        }
        
        # Relation improvements
        initial_relations = initial.get('total_relations', 0)
        final_relations = final.get('total_relations', 0)
        
        if initial_relations > 0:
            improvements['relation_increase_pct'] = ((final_relations - initial_relations) / initial_relations) * 100
        
        improvements['avg_relations_per_doc_increase'] = (
            final.get('avg_relations_per_doc', 0) - initial.get('avg_relations_per_doc', 0)
        )
        
        # Simple quality score (relations per entity ratio)
        initial_score = initial.get('avg_relations_per_doc', 0) / max(1, initial.get('avg_entities_per_doc', 1))
        final_score = final.get('avg_relations_per_doc', 0) / max(1, final.get('avg_entities_per_doc', 1))
        improvements['quality_score_improvement'] = final_score - initial_score
        
        return improvements
    
    def generate_healing_report(self, healing_stats: Dict[str, Any]) -> str:
        """Generate a comprehensive healing report.
        
        Args:
            healing_stats: Healing statistics from heal_documents
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("KNOWLEDGE GRAPH HEALING REPORT")
        report.append("=" * 60)
        
        # Initial Quality
        initial = healing_stats.get('initial_quality', {})
        report.append(f"\nINITIAL QUALITY:")
        report.append(f"  Documents: {initial.get('total_documents', 0)}")
        report.append(f"  Entities: {initial.get('total_entities', 0)}")
        report.append(f"  Relations: {initial.get('total_relations', 0)}")
        report.append(f"  Avg entities/doc: {initial.get('avg_entities_per_doc', 0):.2f}")
        report.append(f"  Avg relations/doc: {initial.get('avg_relations_per_doc', 0):.2f}")
        
        # Opportunities
        opportunities = healing_stats.get('opportunities', {})
        report.append(f"\nHEALING OPPORTUNITIES:")
        
        entity_opps = opportunities.get('entity_opportunities', {})
        if entity_opps:
            report.append(f"  Entity duplicates found: {entity_opps.get('potential_duplicates', 0)}")
            report.append(f"  Duplicate mentions: {entity_opps.get('total_duplicate_mentions', 0)}")
        
        relation_opps = opportunities.get('relation_opportunities', {})
        if relation_opps:
            report.append(f"  Relation candidates: {relation_opps.get('total_candidates', 0)}")
            report.append(f"  High confidence: {relation_opps.get('high_confidence_candidates', 0)}")
        
        # Healing Results
        report.append(f"\nHEALING RESULTS:")
        
        entity_results = healing_stats.get('entity_resolution', {})
        if entity_results:
            report.append(f"  Entities merged: {entity_results.get('entities_merged', 0)}")
            report.append(f"  Entity resolution time: {entity_results.get('processing_time', 0):.2f}s")
        
        relation_results = healing_stats.get('relation_completion', {})
        if relation_results:
            report.append(f"  Relations added: {relation_results.get('relations_added', 0)}")
            report.append(f"  Relation completion time: {relation_results.get('processing_time', 0):.2f}s")
        
        # Final Quality
        final = healing_stats.get('final_quality', {})
        report.append(f"\nFINAL QUALITY:")
        report.append(f"  Entities: {final.get('total_entities', 0)}")
        report.append(f"  Relations: {final.get('total_relations', 0)}")
        report.append(f"  Avg entities/doc: {final.get('avg_entities_per_doc', 0):.2f}")
        report.append(f"  Avg relations/doc: {final.get('avg_relations_per_doc', 0):.2f}")
        
        # Improvements
        improvements = healing_stats.get('improvements', {})
        report.append(f"\nIMPROVEMENTS:")
        report.append(f"  Relations increased by: {improvements.get('relation_increase_pct', 0):.1f}%")
        report.append(f"  Avg relations/doc increased by: {improvements.get('avg_relations_per_doc_increase', 0):.2f}")
        report.append(f"  Quality score improvement: {improvements.get('quality_score_improvement', 0):.3f}")
        
        # Performance
        report.append(f"\nPERFORMANCE:")
        report.append(f"  Total processing time: {healing_stats.get('processing_time', 0):.2f}s")
        
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    healer = KGHealer(verbose=True)
    
    # Create sample data
    sample_docs = [
        {
            'vertexSet': [
                [{'name': 'Apple Inc.', 'type': 'ORG'}],
                [{'name': 'Tim Cook', 'type': 'PER'}],
                [{'name': 'California', 'type': 'LOC'}]
            ],
            'labels': [
                {'h': 1, 't': 0, 'r': 'P108'},  # Tim works at Apple
                {'h': 0, 't': 2, 'r': 'P131'},  # Apple located in California
            ]
        },
        {
            'vertexSet': [
                [{'name': 'Apple', 'type': 'ORG'}],  # Potential duplicate
                [{'name': 'iPhone', 'type': 'MISC'}],
                [{'name': 'United States', 'type': 'LOC'}]
            ],
            'labels': [
                {'h': 0, 't': 1, 'r': 'P176'},  # Apple manufacturer of iPhone
                {'h': 0, 't': 2, 'r': 'P17'},   # Apple country US
            ]
        }
    ]
    
    # Apply healing
    healed_docs, stats = healer.heal_documents(sample_docs)
    
    # Generate report
    report = healer.generate_healing_report(stats)
    print(report)
