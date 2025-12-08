# **25fall-ee641-group9-final**
Group members: Vivin Thiyagarajan , Yi Fan

# COLD Attack for LLM Product Ranking Manipulation

This repository contains our implementation of COLD (Continuous Optimization for LLM-based Deception), a novel adversarial attack method for manipulating product rankings in LLM-powered recommendation systems.

## What This Does

COLD generates adversarial text suffixes that, when added to product descriptions, cause language models to rank those products higher in recommendations. Unlike discrete token-based attacks (like GCG), COLD optimizes in continuous embedding space for better performance.

## Quick Start

### Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate ee641-final

# Or use pip
pip install -r requirements.txt

# Authenticate
export HF_TOKEN='your_huggingface_token'
wandb login
```

### Run an Experiment

```bash
# COLD attack
python scripts/run_cold_experiment.py --catalog cameras --product_idx 1

# GCG baseline
python scripts/run_gcg_baseline.py --catalog cameras --product_idx 1

# Quick test (10 iterations)
python scripts/run_cold_experiment.py --catalog cameras --product_idx 1 --test
```

## Repository Structure

```
├── src/
│   ├── main.py              # Main training loop
│   ├── attack.py            # COLD optimization
│   ├── get.py               # Model/data loading
│   └── process.py           # Text processing
├── scripts/
│   ├── run_cold_experiment.py
│   ├── run_gcg_baseline.py
│   └── run_transferability.py
├── configs/
│   ├── cold_config.yaml     # Hyperparameters
│   └── gcg_config.yaml
└── data/catalogs/           # Product datasets (JSONL)
```

## How It Works

1. **Initialize**: Start with random token embeddings for the adversarial suffix
2. **Optimize**: Use gradient descent to minimize a multi-objective loss:
   - Target loss: Make model output target product name
   - Fluency loss: Keep suffix natural-sounding
   - N-gram loss: Avoid banned words
3. **Evaluate**: Test if the suffix makes the target product rank #1

Key hyperparameters (in `configs/cold_config.yaml`):
- Learning rate: 0.03
- Suffix length: 30 tokens
- Iterations: 1000
- Loss weights: fluency=1, ngram=10, target=50

## Data Format

Product catalogs are JSONL files with this structure:

```json
{"Name": "Canon EOS R5", "Natural": "Professional mirrorless camera with 45MP sensor"}
{"Name": "Sony A7 IV", "Natural": "Versatile full-frame camera for photo and video"}
```

## Requirements

- Python 3.10+
- GPU with 16GB+ VRAM (for LLaMA models)
- HuggingFace account with LLaMA access
- W&B account for logging

## Reproducibility

All experiments use seed=9 by default. Main hyperparameters are in config files. Results should be similar (±5% ASR) when using the same setup, though exact numbers vary due to hardware differences.

## Citation

If you use this code, please cite our project report (available upon request).
