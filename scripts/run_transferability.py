"""Transferability Evaluation Script"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation.evaluate_transfer import evaluate_transfer
from src.data.load_catalog import load_catalog

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--product_idx", type=int, required=True)
    parser.add_argument("--suffix_file", required=True)
    parser.add_argument("--source_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()
    
    products = load_catalog(args.catalog)
    with open(args.suffix_file, 'r') as f:
        suffix = f.read().strip()
    
    rank, success = evaluate_transfer(
        args.source_model, args.target_model, products, args.product_idx - 1, suffix)
    
    print(f"Transfer Rank: {rank}, Success: {success}")

if __name__ == "__main__":
    main()
