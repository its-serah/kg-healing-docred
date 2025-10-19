# Thesis Integration Guide

This repository now contains the complete thesis implementation integrated with the original 3-layer evaluation system.

## What's New

### Core Thesis Components (`thesis_components.py`)
- **EntityEmbedder**: RoBERTa-based entity embedding generation
- **RGCNGraphNode**: R-GCN models for graph neural networks  
- **RelDecoder**: Relation-aware MLP decoder
- **RotatE Training**: Knowledge graph embedding training
- **Candidate Generation**: Healing action candidate algorithms

### Data Analysis (`preprocessing.py`)
- Comprehensive EDA functions from thesis
- Graph property analysis
- Healing opportunity detection
- Visualization utilities

### Complete Pipeline (`run_thesis_pipeline.py`)
- **Stage 1**: Exploratory Data Analysis
- **Stage 2**: Entity embedding generation (RoBERTa)
- **Stage 3**: RotatE relation embedding training
- **Stage 4**: R-GCN model training
- **Stage 5**: RL agent training (PPO)
- **Stage 6**: 3-layer evaluation system

### Enhanced Main Runner (`main.py`)
Now supports two modes:
1. **Evaluation mode**: Original evaluation-only pipeline (backward compatible)
2. **Thesis mode**: Complete ML training + evaluation workflow

## Usage

### Quick Start - Evaluation Only (Original)
```bash
python main.py --n_docs 25 --split dev
```

### Complete Thesis Workflow
```bash
python main.py --mode thesis --n_docs 25 --max_docs_train 100
```

### Direct Pipeline Runner
```bash
python run_thesis_pipeline.py --quick
```

### Stage Control
```bash
# Skip training stages, run evaluation only
python main.py --mode thesis --skip_training

# Skip EDA, run training and evaluation
python main.py --mode thesis --skip_eda
```

## Installation

Install additional dependencies for thesis components:
```bash
pip install -r requirements.txt
```

Key new dependencies:
- `transformers>=4.12.0` (for entity embeddings)
- `torch-geometric>=2.0.0` (for R-GCN)
- `pykeen>=1.7.0` (for RotatE)
- `stable-baselines3>=1.6.0` (for RL training)
- `gymnasium>=0.26.0` (for RL environment)

## Output Structure

### Thesis Pipeline Output
```
experiments/YYYYMMDD_HHMMSS_thesis_pipeline/
├── pipeline_config.json          # Configuration used
├── pipeline_results.json         # Summary of all results
├── docred_analysis_report.txt    # EDA report
└── evaluation_results/           # 3-layer evaluation results
    ├── graphs/                   # Before/after graphs
    ├── sanity_table.csv         # Layer 1: Structural metrics
    ├── semantic_gain.csv        # Layer 2: Semantic gains
    ├── action_efficiency.csv    # Layer 2: Action efficiency
    ├── error_correction_*.csv   # Layer 3: Precision/recall
    └── summary_snapshot.json    # Final summary
```

### Generated Artifacts
```
embeddings/
├── transformer/                 # Entity embeddings
│   ├── train/doc_*/entity_*.npy
│   └── dev/doc_*/entity_*.npy
└── rgcn_nodes/                 # R-GCN node embeddings
    └── doc_*/entity_*.npy

rotatE_embeddings/
└── train/*.npy                 # Relation embeddings

models/
├── rgcn_thesis.pt             # Trained R-GCN model
├── decoder_thesis.pt          # Relation decoder
└── ppo_thesis_healer.zip      # RL agent
```

## Key Features

### Preserved Evaluation System
- **Layer 1**: Structural metrics (nodes, edges, density, components)
- **Layer 2**: Healing-specific metrics (semantic gain, action efficiency)  
- **Layer 3**: Error-correction metrics (precision/recall vs gold standard)

### New ML Training
- Contextual entity embeddings with RoBERTa
- RotatE knowledge graph embeddings for relations
- R-GCN graph neural network for node representations
- PPO reinforcement learning for healing policy

### Modular Design
- Each component can be used independently
- Stages can be skipped/enabled as needed
- Backward compatibility with original evaluation system

## Examples

### Run Full Pipeline (Quick Mode)
```bash
python run_thesis_pipeline.py --quick
# Uses: 100 docs, 3 RGCN epochs, 10k RL timesteps
```

### Custom Training Configuration  
```bash
python run_thesis_pipeline.py \
    --max_docs 500 \
    --rgcn_epochs 6 \
    --rl_timesteps 50000 \
    --eval_n_docs 25
```

### Evaluation with Thesis Models
```bash
# After training, run evaluation using trained components
python main.py --mode thesis --skip_training --n_docs 50
```

The integration maintains the original 3-layer evaluation framework while adding the complete ML training pipeline from your thesis. You can now run the thesis workflow and then evaluate it using the established metrics system!
