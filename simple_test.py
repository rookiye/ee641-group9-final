"""
Simple Smoke Test - Basic Verification

Tests core functionality without heavy dependencies.
"""

import sys
import os
from pathlib import Path

print("="*60)
print("SIMPLE SMOKE TEST")
print("="*60)

# Test 1: Python version
print("\n[1/7] Python version...")
print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Basic imports
print("\n[2/7] Basic dependencies...")
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except:
    print("[ERROR] PyTorch missing")
    sys.exit(1)

try:
    import yaml
    print("[OK] PyYAML")
except:
    print("[ERROR] PyYAML missing")

# Test 3: Data files
print("\n[3/7] Data files...")
catalogs = ['cameras', 'shampoo', 'coffee_machines', 'books']
for cat in catalogs:
    path = Path(f'data/catalogs/{cat}.jsonl')
    if path.exists():
        print(f"[OK] {cat}.jsonl")
    else:
        print(f"[ERROR] {cat}.jsonl MISSING")

# Test 4: Config files
print("\n[4/7] Config files...")
for cfg in ['cold_config.yaml', 'gcg_config.yaml']:
    path = Path(f'configs/{cfg}')
    if path.exists():
        print(f"[OK] {cfg}")
    else:
        print(f"[ERROR] {cfg} MISSING")

# Test 5: Source files
print("\n[5/7] Source files...")
src_files = ['main.py', 'attack.py', 'get.py', 'process.py']
for f in src_files:
    if Path(f'src/{f}').exists():
        print(f"[OK] src/{f}")
    else:
        print(f"[ERROR] src/{f} MISSING")

# Test 6: Scripts
print("\n[6/7] Scripts...")
scripts = ['run_cold_experiment.py', 'run_gcg_baseline.py']
for s in scripts:
    if Path(f'scripts/{s}').exists():
        print(f"[OK] {s}")
    else:
        print(f"[ERROR] {s} MISSING")

# Test 7: Authentication
print("\n[7/7] Authentication...")
hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
if hf_token:
    print(f"[OK] HuggingFace token set")
else:
    print("[WARNING] HuggingFace token NOT set")
    print("  Set with: export HF_TOKEN='your_token'")

print("\n" + "="*60)
print("BASIC CHECKS COMPLETE")
print("="*60)
print("\nNext: Set HF_TOKEN and run:")
print("  python scripts/run_cold_experiment.py --catalog cameras --product_idx 1 --test")
print("="*60)
