"""GCG Training Script - Wrapper around scripts/run_gcg_baseline.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from scripts.run_gcg_baseline import main

if __name__ == "__main__":
    main()
