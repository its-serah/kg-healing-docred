#!/usr/bin/env python3
"""
Core ML components extracted from final_thesisforgood.py
Contains R-GCN models, embedding generation, RotatE training, and utilities
"""

import os
import glob
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, Set2Set
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Optional dependencies
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Entity embedding generation will not work.")

try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    print("Warning: pykeen not available. RotatE training will not work.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not available. Text cleaning will not work.")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Text Processing Utilities
# =====================================

def setup_nltk():
    """Download required NLTK data"""
    if not NLTK_AVAILABLE:
        return False
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        print(f"Failed to download NLTK data: {e}")
        return False

def clean_tokens(text):
    """Clean and normalize text"""
    if not NLTK_AVAILABLE:
        return text.lower().split()
    
    import string
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = nltk.word_tokenize(text.lower())
    cleaned = []
    for tok in tokens:
        if tok in string.punctuation:
            continue
        if tok in stop_words:
            continue
        lemma = lemmatizer.lemmatize(tok)
        cleaned.append(lemma)
    return cleaned

# =====================================
# Entity Embedding Generation
# =====================================

class EntityEmbedder:
    """Handles entity embedding generation using transformer models"""
    
    def __init__(self, model_name="distilroberta-base", max_length=512):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for entity embedding")
            
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def get_embedding(self, text):
        """Return mean pooled transformer embedding for a text string."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pool across tokens
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    
    @torch.no_grad()
    def get_embeddings_batch(self, text_list, batch_size=32):
        """Batch embedding computation for efficiency"""
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=self.max_length)
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            outputs = self.model(**inputs)
            # CLS token for each batch item
            batch_embs = outputs.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(batch_embs)
        embeddings = np.vstack(embeddings)
        return embeddings

# =====================================
# R-GCN Models
# =====================================

class RGCNGraphNode(nn.Module):
    """R-GCN that returns both node and graph embeddings"""
    
    def __init__(self, in_dim, hidden_dim, num_relations, num_layers=3, pool_method='set2set'):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations))
        for _ in range(num_layers-1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            
        if pool_method == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=3)
            self.pool_out_dim = 2*hidden_dim
        else:
            self.pool = global_mean_pool
            self.pool_out_dim = hidden_dim
            
        self.pool_method = pool_method
    
    def forward(self, x, edge_index, edge_type, batch):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_type))
        
        if self.pool_method == 'set2set':
            graph_emb = self.pool(h, batch)
        else:
            graph_emb = self.pool(h, batch)
            
        return h, graph_emb

class RelDecoder(nn.Module):
    """Relation-aware MLP decoder for link prediction"""
    
    def __init__(self, node_dim, rel_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + rel_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, h_emb, r_emb, t_emb):
        inp = torch.cat([h_emb, r_emb, t_emb], dim=1)
        return torch.sigmoid(self.net(inp)).squeeze(1)

# =====================================
# Graph Construction Utilities
# =====================================

def build_doc_graph(doc):
    """Build a NetworkX graph for one document."""
    G = nx.DiGraph()
    n_entities = len(doc['vertexSet'])
    G.add_nodes_from(range(n_entities))
    for rel in doc.get("labels", []):
        G.add_edge(rel['h'], rel['t'], relation=rel['r'], evidence=rel['evidence'])
    return G

def build_pyg_graph(doc_data, entity_doc_dir, rotatE_edge_embeds=None, rel_feat_dim=200):
    """Build PyTorch Geometric graph from document data"""
    # Load node features
    entity_files = sorted(glob.glob(os.path.join(entity_doc_dir, "entity_*.npy")))
    if len(entity_files) == 0:
        raise FileNotFoundError(f"No entity files found in {entity_doc_dir}")
    x = torch.tensor(np.stack([np.load(f) for f in entity_files]), dtype=torch.float)
    
    # Build relation mapping
    if rotatE_edge_embeds is None:
        rotatE_edge_embeds = {}
    
    # Create relation2idx mapping
    all_relations = set()
    for rel in doc_data.get('labels', []):
        all_relations.add(rel['r'])
    relation2idx = {rel: idx for idx, rel in enumerate(sorted(all_relations))}
    
    # Build edges
    edge_index = []
    edge_type = []
    edge_attr_list = []
    
    for rel in doc_data.get('labels', []):
        h, t = rel['h'], rel['t']
        r_label = rel['r']
        r_idx = relation2idx.get(r_label, 0)
        
        edge_index.append([h, t])
        edge_type.append(r_idx)
        
        if r_label in rotatE_edge_embeds:
            edge_attr_list.append(rotatE_edge_embeds[r_label])
        else:
            edge_attr_list.append(np.zeros(rel_feat_dim, dtype=np.float32))
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = torch.tensor(np.stack(edge_attr_list), dtype=torch.float)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_attr = torch.empty((0, rel_feat_dim), dtype=torch.float)
    
    batch = torch.zeros(x.size(0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, 
                edge_attr=edge_attr, batch=batch)

# =====================================
# Candidate Generation (from thesis)
# =====================================

def generate_candidate_actions(
    doc_data,
    entity_emb_dir,
    relation2idx=None,
    rotatE_edge_embeds=None,
    merge_threshold=0.9,
    top_k_merge_per_entity=5,
    max_merge_candidates=200,
    refine_threshold=0.8,
    top_k_refines_per_edge=3
):
    """Generate candidate healing actions from thesis implementation"""
    # Load entity embeddings
    entity_files = sorted(glob.glob(os.path.join(entity_emb_dir, "entity_*.npy")))
    n_entities = len(entity_files)
    if n_entities == 0:
        return {"merge_candidates": [], "chain_candidates": [], "refine_candidates": []}
    entity_embs = np.stack([np.load(f) for f in entity_files])
    
    # Normalize embeddings
    norms = np.linalg.norm(entity_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    entity_embs_norm = entity_embs / norms
    
    # MERGE CANDIDATES
    nbrs = NearestNeighbors(n_neighbors=min(n_entities, top_k_merge_per_entity+1), 
                           metric='cosine').fit(entity_embs_norm)
    distances, indices = nbrs.kneighbors(entity_embs_norm)
    merge_candidates = []
    for i in range(n_entities):
        for nbr_idx, dist in zip(indices[i,1:], distances[i,1:]):
            j = int(nbr_idx)
            if i == j:
                continue
            score = 1.0 - float(dist)
            if score >= merge_threshold:
                head_type = None
                tail_type = None
                try:
                    head_type = doc_data['vertexSet'][i][0].get('type')
                    tail_type = doc_data['vertexSet'][j][0].get('type')
                except Exception:
                    pass
                if head_type != tail_type:
                    score *= 0.5
                merge_candidates.append((i, j, score, head_type, tail_type))
    
    # Deduplicate and sort
    seen_pairs = {}
    for a,b,score,ht,tt in merge_candidates:
        key = tuple(sorted((a,b)))
        if key not in seen_pairs or score > seen_pairs[key][2]:
            seen_pairs[key] = (a,b,score,ht,tt)
    merge_candidates = list(seen_pairs.values())
    merge_candidates.sort(key=lambda x: x[2], reverse=True)
    merge_candidates = merge_candidates[:max_merge_candidates]
    
    # CHAIN CANDIDATES
    adjacency = {}
    for rel in doc_data.get('labels', []):
        h, t = rel['h'], rel['t']
        adjacency.setdefault(h, set()).add(t)
    
    chain_candidates = []
    for a in list(adjacency.keys()):
        for b in list(adjacency.get(a, [])):
            for c in list(adjacency.get(b, [])):
                if c in adjacency.get(a, set()):
                    chain_candidates.append((int(a), int(b), int(c)))
    chain_candidates = list(dict.fromkeys(chain_candidates))
    
    # REFINE CANDIDATES
    refine_candidates = []
    if rotatE_edge_embeds:
        rel_labels = sorted(list(rotatE_edge_embeds.keys()))
        rel_mat = np.stack([rotatE_edge_embeds[r] for r in rel_labels])
        rel_norms = np.linalg.norm(rel_mat, axis=1, keepdims=True)
        rel_norms[rel_norms==0] = 1.0
        rel_mat_norm = rel_mat / rel_norms
        
        for edge_idx, rel in enumerate(doc_data.get('labels', [])):
            old_label = rel['r']
            if old_label not in rotatE_edge_embeds:
                continue
            
            old_vec = rotatE_edge_embeds[old_label]
            old_vec_norm = old_vec / (np.linalg.norm(old_vec) + 1e-12)
            sims = rel_mat_norm.dot(old_vec_norm)
            
            sorted_idx = np.argsort(-sims)
            proposals = []
            for idx in sorted_idx:
                candidate_label = rel_labels[idx]
                if candidate_label == old_label:
                    continue
                score = float(sims[idx])
                proposals.append((candidate_label, score))
                if len(proposals) >= top_k_refines_per_edge:
                    break
            
            if proposals:
                refine_candidates.append({
                    "edge_idx": edge_idx,
                    "head": int(rel['h']),
                    "tail": int(rel['t']),
                    "old_rel": old_label,
                    "proposals": proposals
                })
    
    return {
        "merge_candidates": merge_candidates,
        "chain_candidates": chain_candidates,
        "refine_candidates": refine_candidates
    }
