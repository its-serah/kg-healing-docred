#!/usr/bin/env python3
"""
Knowledge Graph Healing using Ollama and Llama 3 - Enhanced Version
=================================================================

This script provides a comprehensive framework for knowledge graph healing
using the Re-DocRED dataset and local LLMs via Ollama.

Features:
- Full dataset processing with progress tracking
- Comprehensive statistics and analysis
- Before/after visualizations
- Benchmarking capabilities for research comparison
- Export functionality for further analysis

Author: Serah
Repository: https://github.com/its-serah/kg-healing-docred
"""

import json
import requests
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    type: str
    mentions: List[Dict]

@dataclass
class Relation:
    """Represents a relation triple"""
    head: int
    relation: str
    tail: int
    evidence: List[int]
    confidence: float = 1.0  # For predicted relations

@dataclass
class Document:
    """Represents a document with entities, relations, and text"""
    title: str
    entities: List[Entity]
    relations: List[Relation]
    sentences: List[List[str]]
    
@dataclass
class HealingStats:
    """Statistics for healing process"""
    total_documents: int
    total_entities: int
    original_relations: int
    predicted_relations: int
    healing_rate: float
    avg_entities_per_doc: float
    avg_relations_per_doc: float
    processing_time: float

class OllamaClient:
    """Enhanced client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout
        self.request_count = 0
        self.failed_requests = 0
        
    def generate(self, model: str, prompt: str, max_tokens: int = 100) -> Tuple[str, float]:
        """Generate text using Ollama API with confidence estimation"""
        self.request_count += 1
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.1,  # Low temperature for consistency
                        "top_p": 0.9
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            processing_time = time.time() - start_time
            result = response.json()["response"]
            
            # Simple confidence based on response length and processing time
            confidence = min(1.0, len(result.strip()) / 10.0)
            
            return result, confidence
            
        except requests.exceptions.RequestException as e:
            self.failed_requests += 1
            logger.error(f"Error calling Ollama API: {e}")
            return "", 0.0
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            "total_requests": self.request_count,
            "failed_requests": self.failed_requests,
            "success_rate": (self.request_count - self.failed_requests) / max(1, self.request_count)
        }

class KnowledgeGraphHealer:
    """Enhanced main class for healing knowledge graphs"""
    
    def __init__(self, model_name: str = "llama3", enable_caching: bool = True):
        self.ollama = OllamaClient()
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.relation_cache = {}
        
        # Comprehensive relation mappings
        self.relation_descriptions = {
            "P580": "start time", "P582": "end time", "P276": "location",
            "P1344": "participant", "P27": "country of citizenship", "P569": "date of birth",
            "P570": "date of death", "P19": "place of birth", "P131": "located in administrative division",
            "P17": "country", "P150": "contains administrative division", "P161": "cast member",
            "P57": "director", "P577": "publication date", "P175": "performer",
            "P495": "country of origin", "P31": "instance of", "P279": "subclass of",
            "P106": "occupation", "P108": "employer", "P69": "educated at",
            "P463": "member of", "P39": "position held", "P26": "spouse",
            "P22": "father", "P25": "mother", "P40": "child", "P3373": "sibling",
            "P571": "inception", "P740": "location of formation", "P159": "headquarters location",
            "P112": "founder", "P127": "owned by", "P137": "operator", "P1411": "nominated for"
        }
        
        # Common relation patterns by entity type
        self.relation_patterns = {
            ("PER", "LOC"): ["P27", "P19", "P20", "P108", "P69"],
            ("PER", "PER"): ["P26", "P22", "P25", "P40", "P3373", "P463"],
            ("PER", "ORG"): ["P108", "P463", "P69", "P39"],
            ("LOC", "LOC"): ["P131", "P150", "P17", "P47"],
            ("ORG", "LOC"): ["P159", "P17", "P131", "P740"],
            ("PER", "TIME"): ["P569", "P570", "P580", "P582"],
            ("MISC", "PER"): ["P57", "P161", "P175", "P170"]
        }
    
    def load_dataset(self, file_path: str, limit: Optional[int] = None) -> List[Document]:
        """Load Re-DocRED dataset with progress tracking"""
        logger.info(f"Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
        
        documents = []
        for item in tqdm(data, desc="Loading documents"):
            # Parse entities
            entities = []
            for idx, vertex_cluster in enumerate(item["vertexSet"]):
                first_mention = vertex_cluster[0]
                entity = Entity(
                    id=str(idx),
                    name=first_mention["name"],
                    type=first_mention["type"],
                    mentions=vertex_cluster
                )
                entities.append(entity)
            
            # Parse relations
            relations = []
            for label in item["labels"]:
                relation = Relation(
                    head=label["h"],
                    relation=label["r"], 
                    tail=label["t"],
                    evidence=label.get("evidence", [])
                )
                relations.append(relation)
            
            documents.append(Document(
                title=item["title"],
                entities=entities,
                relations=relations,
                sentences=item["sents"]
            ))
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def get_entity_context(self, doc: Document, entity_idx: int, max_sentences: int = 2) -> str:
        """Get textual context for an entity"""
        entity = doc.entities[entity_idx]
        
        relevant_sentences = []
        for mention in entity.mentions[:3]:  # Limit mentions for efficiency
            sent_id = mention["sent_id"]
            if sent_id < len(doc.sentences):
                sentence = " ".join(doc.sentences[sent_id])
                if sentence not in relevant_sentences:
                    relevant_sentences.append(sentence)
        
        return " ".join(relevant_sentences[:max_sentences])
    
    def generate_healing_prompt(self, doc: Document, head_idx: int, tail_idx: int) -> str:
        """Generate optimized prompt for relation prediction"""
        head_entity = doc.entities[head_idx]
        tail_entity = doc.entities[tail_idx]
        
        # Get context
        head_context = self.get_entity_context(doc, head_idx)
        tail_context = self.get_entity_context(doc, tail_idx)
        
        # Get suggested relations based on entity types
        entity_pair = (head_entity.type, tail_entity.type)
        suggested_relations = self.relation_patterns.get(entity_pair, [])
        
        # Limit context length for efficiency
        if len(head_context) > 200:
            head_context = head_context[:200] + "..."
        if len(tail_context) > 200:
            tail_context = tail_context[:200] + "..."
        
        prompt = f"""Task: Identify missing Wikidata relations between two entities.

Entity 1: "{head_entity.name}" (Type: {head_entity.type})
Context: {head_context}

Entity 2: "{tail_entity.name}" (Type: {tail_entity.type})
Context: {tail_context}

Suggested relations for {head_entity.type}-{tail_entity.type}: {", ".join(suggested_relations[:5])}

Based on the context, what Wikidata relation exists between these entities?
Respond with ONLY the relation code (e.g., "P27") or "NONE" if no clear relation exists.

Answer:"""
        
        return prompt
    
    def predict_relations(self, doc: Document, max_pairs: int = 20) -> List[Tuple[int, int, str, float]]:
        """Predict missing relations for a document"""
        # Get existing relations
        existing_pairs = set()
        for rel in doc.relations:
            existing_pairs.add((min(rel.head, rel.tail), max(rel.head, rel.tail)))
        
        # Generate candidate pairs
        candidate_pairs = []
        for i in range(len(doc.entities)):
            for j in range(i + 1, len(doc.entities)):
                if (i, j) not in existing_pairs:
                    candidate_pairs.append((i, j))
        
        # Prioritize pairs by entity type compatibility
        def pair_priority(pair):
            i, j = pair
            type_pair = (doc.entities[i].type, doc.entities[j].type)
            return len(self.relation_patterns.get(type_pair, []))
        
        candidate_pairs.sort(key=pair_priority, reverse=True)
        candidate_pairs = candidate_pairs[:max_pairs]
        
        predicted_relations = []
        
        for head_idx, tail_idx in candidate_pairs:
            # Check cache first
            cache_key = f"{doc.title}_{head_idx}_{tail_idx}"
            if self.enable_caching and cache_key in self.relation_cache:
                result, confidence = self.relation_cache[cache_key]
            else:
                prompt = self.generate_healing_prompt(doc, head_idx, tail_idx)
                result, confidence = self.ollama.generate(self.model_name, prompt, max_tokens=20)
                
                if self.enable_caching:
                    self.relation_cache[cache_key] = (result, confidence)
            
            result = result.strip().upper()
            
            # Validate result
            if result != "NONE" and result.startswith("P") and result[1:].isdigit():
                predicted_relations.append((head_idx, tail_idx, result, confidence))
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        return predicted_relations
    
    def heal_knowledge_graphs(self, documents: List[Document], output_dir: str = "results") -> HealingStats:
        """Process entire dataset and heal knowledge graphs"""
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Starting knowledge graph healing...")
        
        all_results = []
        total_original_relations = 0
        total_predicted_relations = 0
        total_entities = 0
        
        for doc in tqdm(documents, desc="Healing knowledge graphs"):
            original_relations = len(doc.relations)
            predicted_relations = self.predict_relations(doc)
            
            result = {
                "title": doc.title,
                "entities": len(doc.entities),
                "original_relations": original_relations,
                "predicted_relations": len(predicted_relations),
                "entity_types": Counter([e.type for e in doc.entities]),
                "relation_types": Counter([r.relation for r in doc.relations]),
                "predictions": [
                    {
                        "head": doc.entities[h].name,
                        "head_type": doc.entities[h].type,
                        "tail": doc.entities[t].name, 
                        "tail_type": doc.entities[t].type,
                        "relation": rel,
                        "relation_description": self.relation_descriptions.get(rel, "Unknown"),
                        "confidence": conf
                    }
                    for h, t, rel, conf in predicted_relations
                ]
            }
            
            all_results.append(result)
            total_original_relations += original_relations
            total_predicted_relations += len(predicted_relations)
            total_entities += len(doc.entities)
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        stats = HealingStats(
            total_documents=len(documents),
            total_entities=total_entities,
            original_relations=total_original_relations,
            predicted_relations=total_predicted_relations,
            healing_rate=total_predicted_relations / max(1, total_original_relations),
            avg_entities_per_doc=total_entities / len(documents),
            avg_relations_per_doc=total_original_relations / len(documents),
            processing_time=processing_time
        )
        
        # Save detailed results
        results_file = os.path.join(output_dir, "healing_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = os.path.join(output_dir, "healing_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        return stats, all_results

class Visualizer:
    """Class for creating visualizations and analysis"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_healing_overview(self, stats: HealingStats, results: List[Dict]):
        """Create overview plots of the healing process"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Knowledge Graph Healing Overview', fontsize=16)
        
        # 1. Before/After Relations Count
        ax1 = axes[0, 0]
        categories = ['Original Relations', 'Predicted Relations']
        values = [stats.original_relations, stats.predicted_relations]
        bars = ax1.bar(categories, values, color=['#3498db', '#e74c3c'])
        ax1.set_title('Relations: Before vs After')
        ax1.set_ylabel('Number of Relations')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    str(value), ha='center', va='bottom')
        
        # 2. Entity Type Distribution
        ax2 = axes[0, 1]
        entity_types = defaultdict(int)
        for result in results:
            for entity_type, count in result['entity_types'].items():
                entity_types[entity_type] += count
        
        types, counts = zip(*sorted(entity_types.items(), key=lambda x: x[1], reverse=True))
        ax2.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Entity Type Distribution')
        
        # 3. Relation Type Distribution
        ax3 = axes[1, 0]
        relation_types = defaultdict(int)
        for result in results:
            for rel_type, count in result['relation_types'].items():
                relation_types[rel_type] += count
        
        top_relations = sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:15]
        relations, rel_counts = zip(*top_relations)
        ax3.barh(range(len(relations)), rel_counts)
        ax3.set_yticks(range(len(relations)))
        ax3.set_yticklabels(relations)
        ax3.set_xlabel('Count')
        ax3.set_title('Top 15 Original Relation Types')
        
        # 4. Healing Rate by Document
        ax4 = axes[1, 1]
        healing_rates = [r['predicted_relations'] / max(1, r['original_relations']) for r in results]
        ax4.hist(healing_rates, bins=20, alpha=0.7, color='#2ecc71')
        ax4.set_xlabel('Healing Rate (Predicted/Original)')
        ax4.set_ylabel('Number of Documents')
        ax4.set_title('Distribution of Healing Rates')
        ax4.axvline(np.mean(healing_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(healing_rates):.2f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'healing_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_analysis(self, results: List[Dict]):
        """Analyze confidence scores of predictions"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract confidence scores
        confidences = []
        for result in results:
            for pred in result['predictions']:
                confidences.append(pred['confidence'])
        
        if not confidences:
            logger.warning("No predictions found for confidence analysis")
            return
        
        # 1. Confidence distribution
        ax1 = axes[0]
        ax1.hist(confidences, bins=20, alpha=0.7, color='#9b59b6')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Number of Predictions')
        ax1.set_title('Distribution of Prediction Confidence')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.2f}')
        ax1.legend()
        
        # 2. Confidence by relation type
        ax2 = axes[1]
        relation_confidences = defaultdict(list)
        for result in results:
            for pred in result['predictions']:
                relation_confidences[pred['relation']].append(pred['confidence'])
        
        # Get top 10 most frequent predicted relations
        top_relations = sorted(relation_confidences.items(), 
                             key=lambda x: len(x[1]), reverse=True)[:10]
        
        relations = [r[0] for r in top_relations]
        conf_data = [r[1] for r in top_relations]
        
        ax2.boxplot(conf_data, labels=relations)
        ax2.set_xlabel('Relation Type')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence by Predicted Relation Type')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confidence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, stats: HealingStats, results: List[Dict]):
        """Create a comprehensive summary report"""
        report = f"""
# Knowledge Graph Healing Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Documents Processed:** {stats.total_documents:,}
- **Total Entities:** {stats.total_entities:,}
- **Original Relations:** {stats.original_relations:,}
- **Predicted Relations:** {stats.predicted_relations:,}
- **Healing Rate:** {stats.healing_rate:.2%}
- **Average Entities per Document:** {stats.avg_entities_per_doc:.1f}
- **Average Relations per Document:** {stats.avg_relations_per_doc:.1f}
- **Processing Time:** {stats.processing_time:.1f} seconds

## Key Insights

### Entity Types
"""
        
        # Aggregate entity types
        entity_types = defaultdict(int)
        for result in results:
            for entity_type, count in result['entity_types'].items():
                entity_types[entity_type] += count
        
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats.total_entities) * 100
            report += f"- **{entity_type}:** {count:,} ({percentage:.1f}%)\n"
        
        report += "\n### Most Frequent Predicted Relations\n"
        
        # Aggregate predicted relations
        predicted_relations = defaultdict(int)
        for result in results:
            for pred in result['predictions']:
                predicted_relations[pred['relation']] += 1
        
        for rel, count in sorted(predicted_relations.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"- **{rel}:** {count:,} predictions\n"
        
        report += f"""
## Performance Metrics

- **Documents with Predictions:** {sum(1 for r in results if r['predicted_relations'] > 0):,}
- **Average Predictions per Document:** {stats.predicted_relations / stats.total_documents:.1f}
- **Documents with High Healing Rate (>0.5):** {sum(1 for r in results if r['predicted_relations'] / max(1, r['original_relations']) > 0.5):,}

## Methodology

This analysis used Ollama with Llama 3 to identify potentially missing relations in the Re-DocRED dataset.
The approach focuses on entity pairs without existing relations and uses contextual information to predict
likely missing connections.

### Limitations

- Predictions are based on local context and may miss global knowledge
- No manual validation of predicted relations
- Limited to relations present in Wikidata taxonomy
- Processing time scales with dataset size

"""
        
        # Save report
        with open(os.path.join(self.output_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Knowledge Graph Healing with Ollama')
    parser.add_argument('--dataset', default='Re-DocRED/data/dev_revised.json',
                       help='Path to Re-DocRED dataset file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of documents to process')
    parser.add_argument('--output', default='results',
                       help='Output directory for results')
    parser.add_argument('--model', default='llama3',
                       help='Ollama model name')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching of LLM responses')
    
    args = parser.parse_args()
    
    # Initialize healer
    healer = KnowledgeGraphHealer(model_name=args.model, enable_caching=not args.no_cache)
    
    # Load dataset
    documents = healer.load_dataset(args.dataset, limit=args.limit)
    
    # Process dataset
    stats, results = healer.heal_knowledge_graphs(documents, args.output)
    
    # Create visualizations
    visualizer = Visualizer(args.output)
    visualizer.plot_healing_overview(stats, results)
    visualizer.plot_confidence_analysis(results)
    visualizer.create_summary_report(stats, results)
    
    # Print final statistics
    logger.info("\n" + "="*60)
    logger.info("KNOWLEDGE GRAPH HEALING COMPLETED")
    logger.info("="*60)
    logger.info(f"Documents processed: {stats.total_documents:,}")
    logger.info(f"Original relations: {stats.original_relations:,}")
    logger.info(f"Predicted relations: {stats.predicted_relations:,}")
    logger.info(f"Healing rate: {stats.healing_rate:.2%}")
    logger.info(f"Processing time: {stats.processing_time:.1f}s")
    logger.info(f"Results saved to: {args.output}")
    
    # API statistics
    api_stats = healer.ollama.get_stats()
    logger.info(f"API requests: {api_stats['total_requests']}")
    logger.info(f"Success rate: {api_stats['success_rate']:.2%}")

if __name__ == "__main__":
    main()
