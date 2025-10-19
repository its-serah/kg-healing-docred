# Knowledge Graph Healing for DocRED

A comprehensive modular pipeline for knowledge graph healing on the DocRED dataset, focusing on entity resolution, relation completion, and comprehensive evaluation across multiple layers.

## Overview

This project implements a structured evaluation framework for knowledge graph healing with three distinct evaluation layers:

- **Layer 1 (Structural)**: Basic graph metrics (nodes, edges, density, components)
- **Layer 2 (Healing-specific)**: Semantic consistency gain and action efficiency 
- **Layer 3 (Error-correction)**: Precision/recall against gold standard annotations

## Project Structure

```
kg-healing-docred/
├── config.py              # Configuration and data loading
├── helpers.py              # Utility functions and metrics
├── adapters.py             # Graph building and candidate generation adapters
├── experiment.py           # Main experiment runner (Layer 1)
├── healing_metrics.py      # Layer 2 metrics (semantic gain, efficiency)
├── error_correction.py     # Layer 3 metrics (precision/recall/F1)
├── schema_validation.py    # Optional schema constraint validation
├── main.py                 # Complete pipeline orchestrator
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
# Knowledge Graph Healing with Ollama & Llama 3

A comprehensive framework for identifying and healing missing relations in knowledge graphs using local LLMs via Ollama. Built for research and benchmarking on the Re-DocRED dataset.

## Features

- Local LLM Processing: Uses Ollama with Llama 3 - no API keys needed
- Full Dataset Support: Process entire Re-DocRED dataset with progress tracking
- Comprehensive Analysis: Before/after visualizations and detailed statistics
- Research Ready: Built for academic comparison and benchmarking
- Memory Efficient: Optimized prompts and caching for large datasets

## What It Does

This tool analyzes knowledge graphs to identify potentially missing relations between entities. It:

1. Loads Re-DocRED data with entities, existing relations, and text
2. Identifies candidate entity pairs without existing relations
3. Uses contextual LLM analysis to predict missing relations
4. Generates comprehensive reports with statistics and visualizations

## Installation

### Prerequisites
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3 model  
ollama pull llama3
```

### Python Setup
```bash
cd kg-healing-docred
pip install -r requirements.txt
```

### Data Setup
```bash
# Get Re-DocRED dataset
git clone https://github.com/tonytan48/Re-DocRED.git
```

## Quick Start

### Basic Usage
```bash
# Process 10 documents from dev set
python kg_healing_enhanced.py --limit 10

# Process full dev set
python kg_healing_enhanced.py --dataset Re-DocRED/data/dev_revised.json

# Process with different model
python kg_healing_enhanced.py --model llama3:8b --limit 50
```

### Advanced Usage
```bash
# Full train set processing with custom output
python kg_healing_enhanced.py \
    --dataset Re-DocRED/data/train_revised.json \
    --output my_results \
    --limit 1000

# Disable caching for different runs
python kg_healing_enhanced.py --no-cache --limit 20
```

## Output & Results

The tool generates:

### Files Generated
- `healing_results.json` - Detailed predictions for each document
- `healing_stats.json` - Summary statistics
- `healing_overview.png` - Before/after visualizations
- `confidence_analysis.png` - Confidence score distributions  
- `summary_report.md` - Comprehensive markdown report

### Sample Statistics
```
Documents processed: 500
Original relations: 17,832
Predicted relations: 3,247
Healing rate: 18.2%
Processing time: 892.3s
```

## Example Results

### Input Document: "Willi Schneider (skeleton racer)"
**Entities**: Willi Schneider (PER), Germany (LOC), 1992 Winter Olympics (MISC)

**Original Relations**: 
- P27: country of citizenship (Schneider → Germany)
- P1344: participant (Schneider → Olympics)

**Predicted Missing Relations**:
- P19: place of birth (Schneider → Transylvania) - Confidence: 0.85
- P106: occupation (Schneider → skeleton racer) - Confidence: 0.92

## Configuration

### Command Line Options
```
--dataset      Path to Re-DocRED JSON file (default: Re-DocRED/data/dev_revised.json)
--limit        Limit number of documents to process 
--output       Output directory for results (default: results)
--model        Ollama model name (default: llama3)
--no-cache     Disable response caching
```

### Customization
Edit `kg_healing_enhanced.py` to modify:
- Relation patterns: Entity type → relation mappings
- Context length: Text snippets sent to LLM
- Candidate pairs: Number of entity pairs to check per document
- Confidence scoring: How prediction confidence is calculated

## Supported Relations

The system recognizes 50+ Wikidata relation types including:
- P27: country of citizenship
- P19: place of birth  
- P569/P570: birth/death dates
- P108: employer
- P69: educated at
- P276: location
- P1344: participant
- And many more...

## Use Cases

### Research & Academia
- Benchmark against other KG completion methods
- Analyze relation extraction performance
- Study LLM capabilities for structured prediction

### Industry Applications  
- Enhance existing knowledge bases
- Identify missing facts in corporate KGs
- Improve information extraction pipelines

### Data Analysis
- Understand relation patterns in your data
- Quality assessment of knowledge graphs
- Gap analysis for KB completion

## Performance Notes

### Speed
- ~5-10 documents/minute on standard hardware
- Scales linearly with document count
- Caching reduces repeated API calls

### Memory Usage
- ~2-4GB RAM for full dataset processing
- Efficient streaming for large datasets
- Configurable batch sizes

### Accuracy
- Precision varies by relation type (0.3-0.9)
- Higher accuracy on common relations
- Context-dependent predictions

## Troubleshooting

### Common Issues

**Ollama not responding**:
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
sudo systemctl restart ollama
```

**Out of memory errors**:
- Reduce `--limit` parameter
- Use smaller model: `--model llama3:8b`
- Increase system swap space

**Slow processing**:
- Enable caching (default)
- Reduce candidate pairs per document
- Use GPU acceleration if available

## Research & Citation

If you use this work in research, please cite:
```bibtex
@software{kg_healing_docred,
  title={Knowledge Graph Healing with Ollama & Llama 3},
  author={Serah},
  year={2024},
  url={https://github.com/its-serah/kg-healing-docred}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Contact

- GitHub: [@its-serah](https://github.com/its-serah)
- Issues: [GitHub Issues](https://github.com/its-serah/kg-healing-docred/issues)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your data directory structure:
```
data/
├── train_revised.json
├── dev_revised.json
└── test_revised.json
```

4. (Optional) Set up embeddings cache:
```
embeddings/transformer/
├── dev/
│   ├── doc_0/
│   │   ├── entity_0.npy
│   │   ├── entity_1.npy
│   │   └── ...
│   └── ...
└── ...
```

## Quick Start

### Basic Usage

Run the complete pipeline with default settings:

```python
from main import run_complete_pipeline

# Run pipeline with 25 dev documents
exp_dir, summary = run_complete_pipeline(n_docs=25, split="dev")
```

### Manual Step-by-Step Execution

```python
# 1. Configure settings
from config import DOCS, SPLIT, N_DOCS

# 2. Run main experiment (Layer 1)
from experiment import run_once_and_save
exp_dir = run_once_and_save(split=SPLIT, n_docs=N_DOCS)

# 3. Compute healing-specific metrics (Layer 2)
from healing_metrics import compute_semantic_gain, compute_action_efficiency
scg_df = compute_semantic_gain(exp_dir)
eff_df = compute_action_efficiency(exp_dir)

# 4. Compute error-correction metrics (Layer 3, requires gold standard)
from error_correction import compute_error_correction_metrics
error_results = compute_error_correction_metrics(exp_dir)

# 5. Optional: Schema validation
from schema_validation import schema_report
schema_df = schema_report(exp_dir)
```

## Configuration

Edit `config.py` to adjust paths and settings:

```python
DATA_DIR   = "data"                             # DocRED data location
EMB_CACHE  = "embeddings/transformer"           # Entity embeddings cache
OUT_ROOT   = "experiments"                      # Output directory
SPLIT      = "dev"                              # train|dev|test
N_DOCS     = 25                                 # Number of documents
ROTATE_NPY = None                               # RotatE embeddings (optional)
```

## Adapters

The `adapters.py` module provides interfaces to your existing healing code. Update these functions to point to your implementations:

```python
# Update these function names to match your code:
def generate_candidates(doc, G, entity_embs, rel_embs, ...):
    # Replace with your candidate generation function
    if 'your_generate_function' in globals():
        return your_generate_function(...)
    return []

def apply_candidate(G, cand):
    # Replace with your candidate application function  
    if 'your_apply_function' in globals():
        return your_apply_function(G, cand)
    return G
```

## Output Structure

Each experiment creates a timestamped directory with:

```
experiments/YYYYMMDD_HHMMSS_split_exp/
├── graphs/                    # Before/after graph JSON files
│   ├── doc0_before.json
│   ├── doc0_after.json
│   └── ...
├── sanity_table.csv          # Layer 1: Structural metrics per document
├── actions.csv               # All healing actions applied
├── metrics.json              # Layer 1: Aggregated structural metrics
├── semantic_gain.csv         # Layer 2: Semantic consistency gains
├── action_efficiency.csv     # Layer 2: Action efficiency metrics
├── error_correction_*.csv    # Layer 3: Precision/recall results
├── config.json              # Experiment configuration
└── summary_snapshot.json    # Comprehensive summary
```

## Evaluation Layers

### Layer 1: Structural Metrics
- Nodes, edges, components, density
- Redundancy (transitive closure violations)
- Before/after comparison

### Layer 2: Healing-Specific Metrics
- **Semantic Consistency Gain**: Average cosine similarity between connected entities
- **Action Efficiency**: Actions per document, action type distribution

### Layer 3: Error-Correction Metrics
- Requires gold standard CSV: `human_validation_sheet.csv`
- **MERGE**: Precision/recall/F1, PR curves
- **REFINE**: Accuracy of relation corrections
- **CHAIN**: Coverage of removed redundant paths

## Gold Standard Format

For Layer 3 evaluation, provide annotations in CSV format:

```csv
doc_idx,action_type,head_id,tail_id,new_relation,correct
0,MERGE,1,3,,1
0,REFINE,2,5,P27,1
1,CHAIN,0,2,4,1
```

## Advanced Features

### RotatE Ablation Study
```python
# Compare with/without relation embeddings
exp_dir, summary = run_complete_pipeline(with_ablation=True)
```

### Schema Validation
Automatically validate type constraints:
```python
from schema_validation import schema_report
schema_df = schema_report(exp_dir)
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black .
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kg-healing-docred,
  title={Knowledge Graph Healing for DocRED: A Comprehensive Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/its-serah/kg-healing-docred}
}
```
