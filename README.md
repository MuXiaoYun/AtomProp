# AtomProp

**AtomProp** is a molecular property prediction framework powered by self-supervised pretraining on large-scale molecular datasets. It leverages **Graph Edge Attention Transformer (GeAT)** to learn rich molecular representations through a multi-task learning paradigm.

## Purpose

AtomProp aims to learn universal molecular representations that capture chemical structure and properties, enabling efficient transfer learning on downstream tasks such as molecular property prediction, drug discovery, and chemical analysis.

## Method: GeAT + Multi-Task Pretraining

### GeAT Architecture

GeAT (Graph Edge Attention Transformer) is a novel graph neural network architecture that incorporates explicit edge attention mechanisms:

- **GeATLayer**: Uses multi-head edge attention parallel across bond types to aggregate information from neighboring atoms, explicitly modeling chemical bonds
- **GeATNet**: Stacks multiple GeAT layers with residual connections and layer normalization for robust feature learning. Applies global multi-head self-attention within each graph to aggregate atomic-level representations into graph-level embeddings. Uses MoE (Mixture of Experts) as final output layer.

See [atomprop/models/GeAT.py](atomprop/models/GeAT.py) for implementation details.

### Pretraining Tasks

AtomProp employs a multi-task pretraining strategy with the following tasks:

| Task                        | Description                                              | Type           |
| --------------------------- | -------------------------------------------------------- | -------------- |
| Node Attribute Prediction   | Predict atomic features from graph embeddings            | Regression     |
| Masked Node Prediction      | Recover masked atom types                                | Classification |
| Graph Mask Contrast         | Contrast between unmasked/less-masked/more-masked graphs | Contrastive    |
| Batch Contrast              | InfoNCE loss within batch                                | Contrastive    |
| Functional Group Prediction | Detect functional groups in molecules                    | Classification |
| Scaffold Contrast           | Contrast learning based on molecular scaffolds           | Contrastive    |

See [atomprop/tasks/tasks.py](atomprop/tasks/tasks.py) for task implementations.

### Training Strategy

- **Multi-Task Learning**: Simultaneously optimizes multiple pretraining tasks
- **Uncertainty Weighting**: Automatically adjusts task weights based on task uncertainty (Supports other weighting stratergies as well)
- **One-Cycle LR**: Uses One-CycleLR scheduler for efficient training
- **Chunked Data Loading**: Supports large-scale datasets with memory-efficient chunking

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate atomprop

# Or install with pip
pip install -r requirements.txt
```

## Quick Start

### Pretraining

```bash
python pretrain.py
```

Configure training parameters in [configs/config.py](configs/config.py).

### Fine-tuning on Downstream Tasks

```bash
python finetune.py
```

Example: See [examples/molecule_mass/train.py](examples/molecule_mass/train.py) for a complete fine-tuning example.

### Data Preprocessing

Generate xyzs for SMILES file.

```bash
python preprocess.py
```

## Citation

If you use AtomProp in your research, please cite:

```bibtex
@software{atomprop2024,
  title={AtomProp: Molecular Property Prediction with GeAT},
  author={MuXiaoYun},
  year={2024},
  url={https://github.com/MuXiaoYun/AtomProp}
}
```

## AI Wiki

English: [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MuXiaoYun/AtomProp)

Mandarin: [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/MuXiaoYun/AtomProp)
