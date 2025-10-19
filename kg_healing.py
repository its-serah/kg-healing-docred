#!/usr/bin/env python3
"""
Knowledge Graph Healing using Ollama and Llama 3
=====================================

This script uses the Re-DocRED dataset to identify and heal missing relations
in knowledge graphs using a local Llama 3 model via Ollama.

Dataset Structure:
- vertexSet: List of entity clusters, each containing entity mentions
- labels: List of relation triples with evidence
- sents: List of sentences (tokenized)

Relation Properties (Wikidata format):
- P580: start time
- P582: end time  
- P276: location
- P1344: participant
- P27: country of citizenship
- P569: date of birth
- P19: place of birth
- etc.
"""

import json
import requests
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import time

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

@dataclass
class Document:
    """Represents a document with entities, relations, and text"""
    title: str
    entities: List[Entity]
    relations: List[Relation]
    sentences: List[List[str]]
    
class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        
    def generate(self, model: str, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return ""

class KnowledgeGraphHealer:
    """Main class for healing knowledge graphs"""
    
    def __init__(self, model_name: str = "llama3"):
        self.ollama = OllamaClient()
        self.model_name = model_name
        
        # Common relation mappings for better understanding
        self.relation_descriptions = {
            "P580": "start time",
            "P582": "end time", 
            "P276": "location",
            "P1344": "participant",
            "P27": "country of citizenship",
            "P569": "date of birth",
            "P570": "date of death",
            "P19": "place of birth",
            "P131": "located in administrative division",
            "P17": "country",
            "P150": "contains administrative division",
            "P161": "cast member",
            "P57": "director",
            "P577": "publication date",
            "P175": "performer"
        }
    
    def load_dataset(self, file_path: str, limit: Optional[int] = None) -> List[Document]:
        """Load Re-DocRED dataset"""
        print(f"Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data):
            if limit and i >= limit:
                break
                
            # Parse entities
            entities = []
            for idx, vertex_cluster in enumerate(item["vertexSet"]):
                # Use the first mention as canonical
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
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def get_entity_context(self, doc: Document, entity_idx: int, max_sentences: int = 3) -> str:
        """Get textual context for an entity"""
        entity = doc.entities[entity_idx]
        
        # Find sentences containing this entity
        relevant_sentences = []
        for mention in entity.mentions:
            sent_id = mention["sent_id"]
            if sent_id < len(doc.sentences):
                sentence = " ".join(doc.sentences[sent_id])
                if sentence not in relevant_sentences:
                    relevant_sentences.append(sentence)
        
        # Limit to max_sentences
        return " ".join(relevant_sentences[:max_sentences])
    
    def generate_healing_prompt(self, doc: Document, head_idx: int, tail_idx: int) -> str:
        """Generate prompt for identifying missing relations"""
        head_entity = doc.entities[head_idx]
        tail_entity = doc.entities[tail_idx]
        
        # Get context for both entities
        head_context = self.get_entity_context(doc, head_idx)
        tail_context = self.get_entity_context(doc, tail_idx)
        
        # Get existing relations for context
        existing_relations = []
        for rel in doc.relations:
            if (rel.head == head_idx and rel.tail == tail_idx) or \
               (rel.head == tail_idx and rel.tail == head_idx):
                rel_desc = self.relation_descriptions.get(rel.relation, rel.relation)
                existing_relations.append(f"{rel.relation} ({rel_desc})")
        
        existing_text = ", ".join(existing_relations) if existing_relations else "None"
        
        prompt = f"""
You are an expert in knowledge graph construction and relation extraction.

Document: "{doc.title}"

Entity 1: {head_entity.name} (Type: {head_entity.type})
Context: {head_context}

Entity 2: {tail_entity.name} (Type: {tail_entity.type})  
Context: {tail_context}

Existing relations between these entities: {existing_text}

Based on the context provided, what additional Wikidata relations might exist between "{head_entity.name}" and "{tail_entity.name}"?

Consider common relation types like:
- P27 (country of citizenship)
- P19 (place of birth) 
- P131 (located in administrative division)
- P17 (country)
- P276 (location)
- P580/P582 (start/end time)
- P1344 (participant)

Respond with ONLY the relation code(s) if you find missing relations, or "NONE" if no additional relations are evident.
Examples: "P27" or "P19, P131" or "NONE"

Answer:"""

        return prompt
    
    def identify_missing_relations(self, doc: Document, sample_pairs: int = 10) -> List[Tuple[int, int, str]]:
        """Identify potentially missing relations in a document"""
        print(f"Analyzing document: {doc.title}")
        
        # Get all existing entity pairs that have relations
        existing_pairs = set()
        for rel in doc.relations:
            existing_pairs.add((min(rel.head, rel.tail), max(rel.head, rel.tail)))
        
        # Sample some entity pairs that don't have relations
        all_pairs = []
        for i in range(len(doc.entities)):
            for j in range(i + 1, len(doc.entities)):
                if (i, j) not in existing_pairs:
                    all_pairs.append((i, j))
        
        # Randomly sample pairs to check
        if len(all_pairs) > sample_pairs:
            pairs_to_check = random.sample(all_pairs, sample_pairs)
        else:
            pairs_to_check = all_pairs
        
        missing_relations = []
        
        for head_idx, tail_idx in pairs_to_check:
            prompt = self.generate_healing_prompt(doc, head_idx, tail_idx)
            
            print(f"Checking pair: {doc.entities[head_idx].name} - {doc.entities[tail_idx].name}")
            
            response = self.ollama.generate(self.model_name, prompt, max_tokens=50)
            response = response.strip().upper()
            
            if response != "NONE" and response:
                # Parse multiple relations if present
                relations = [r.strip() for r in response.split(",")]
                for rel in relations:
                    if rel.startswith("P") and rel[1:].isdigit():
                        missing_relations.append((head_idx, tail_idx, rel))
                        print(f"  -> Found potential missing relation: {rel}")
            
            # Small delay to avoid overloading
            time.sleep(0.5)
        
        return missing_relations
    
    def heal_knowledge_graph(self, documents: List[Document], output_file: str):
        """Heal knowledge graphs and save results"""
        print("Starting knowledge graph healing...")
        
        healing_results = []
        
        for doc in documents:
            print(f"\n{'='*60}")
            
            original_relations = len(doc.relations)
            missing_relations = self.identify_missing_relations(doc, sample_pairs=5)
            
            result = {
                "title": doc.title,
                "original_relations": original_relations,
                "entities": [{"name": e.name, "type": e.type} for e in doc.entities],
                "missing_relations": []
            }
            
            for head_idx, tail_idx, relation in missing_relations:
                result["missing_relations"].append({
                    "head": doc.entities[head_idx].name,
                    "tail": doc.entities[tail_idx].name,
                    "relation": relation,
                    "relation_description": self.relation_descriptions.get(relation, "Unknown relation")
                })
            
            healing_results.append(result)
            
            print(f"Document: {doc.title}")
            print(f"Original relations: {original_relations}")
            print(f"Potential missing relations found: {len(missing_relations)}")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(healing_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nHealing results saved to: {output_file}")
        
        # Summary statistics
        total_docs = len(healing_results)
        total_missing = sum(len(r["missing_relations"]) for r in healing_results)
        print(f"\nSummary:")
        print(f"Documents processed: {total_docs}")
        print(f"Total potential missing relations: {total_missing}")
        print(f"Average missing relations per document: {total_missing/total_docs:.2f}")

def main():
    """Main function to run knowledge graph healing"""
    # Initialize healer
    healer = KnowledgeGraphHealer(model_name="llama3")
    
    # Load a small subset for testing
    data_path = "/home/serah/Downloads/kg-healing-docred/Re-DocRED/data/dev_revised.json"
    documents = healer.load_dataset(data_path, limit=5)  # Start with 5 documents
    
    # Heal knowledge graphs
    output_file = "/home/serah/Downloads/kg-healing-docred/healing_results.json"
    healer.heal_knowledge_graph(documents, output_file)

if __name__ == "__main__":
    main()
