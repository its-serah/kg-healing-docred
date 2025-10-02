# Knowledge Graph Healing for DocRED

This project implements a comprehensive knowledge graph healing pipeline for the DocRED dataset, focusing on entity resolution, relation completion, and knowledge graph quality improvement.

## Overview

Knowledge graphs often contain incomplete or inconsistent information. This project provides tools to "heal" knowledge graphs by:

1. **Entity Resolution**: Detecting and merging duplicate entities based on surface forms and contextual similarity
2. **Relation Completion**: Finding missing relations between entities using graph patterns and machine learning
3. **Quality Assessment**: Evaluating the effectiveness of healing operations with comprehensive metrics

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for each healing component
- **DocRED Integration**: Optimized for the Document-level Relation Extraction Dataset format
- **Configurable Parameters**: Adjustable similarity thresholds and healing strategies
- **Comprehensive Evaluation**: Detailed metrics and statistics for healing operations
- **Extensible Framework**: Easy to add new healing algorithms and evaluation methods

## Installation

```bash
git clone https://github.com/your-username/kg-healing-docred.git
cd kg-healing-docred
pip install -r requirements.txt
```

## Quick Start

```python
from data_loader import DocREDLoader
from entity_resolution import EntityResolver
from kg_healer import KGHealer

# Load data
loader = DocREDLoader()
docs = loader.load_docred_subset(num_docs=100)

# Initialize healer
healer = KGHealer()

# Apply healing
healed_docs, stats = healer.heal_documents(docs)

print(f"Healed {len(healed_docs)} documents")
print(f"Entity merges: {stats['entities_merged']}")
print(f"Relations completed: {stats['relations_completed']}")
```

## Modules

- **`data_loader.py`**: DocRED dataset loading and preprocessing
- **`entity_resolution.py`**: Entity duplicate detection and merging
- **`relation_completion.py`**: Missing relation discovery and completion
- **`kg_healer.py`**: Main orchestrator for the healing pipeline
- **`evaluation.py`**: Metrics and evaluation tools
- **`utils.py`**: Helper functions and utilities
- **`main.py`**: Example usage and demonstration script

## Architecture

The healing pipeline follows this workflow:

1. **Data Loading**: Load DocRED documents and build initial knowledge graphs
2. **Entity Resolution**: Identify and merge duplicate entities across documents
3. **Relation Completion**: Discover missing relations using various strategies
4. **Quality Assessment**: Evaluate the improvements made to the knowledge graph
5. **Output**: Generate healed knowledge graph with detailed statistics

## Algorithms

### Entity Resolution
- Surface form normalization with abbreviation handling
- Multi-factor similarity scoring (string, type, context)
- Greedy clustering for entity merging
- Configurable similarity thresholds

### Relation Completion
- Graph pattern mining for relation discovery
- Transitivity-based relation inference
- Type consistency checking
- Confidence scoring for new relations

## Evaluation Metrics

- Entity resolution precision, recall, and F1
- Relation completion accuracy and coverage
- Knowledge graph completeness improvements
- Processing time and scalability metrics

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kg-healing-docred,
  title={Knowledge Graph Healing for DocRED},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/kg-healing-docred}
}
```
