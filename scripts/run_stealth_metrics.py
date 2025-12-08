"""Stealth Metrics Calculation Script"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.evaluate_stealth import calculate_perplexity, calculate_embedding_similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix_file", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map='auto').eval()
    
    with open(args.suffix_file, 'r') as f:
        suffix = f.read().strip()
    
    ppl = calculate_perplexity(suffix, model, tokenizer)
    baseline = "This is a high-quality product with excellent features."
    sim = calculate_embedding_similarity(suffix, baseline, model, tokenizer)
    
    print(f"Perplexity: {ppl:.2f}")
    print(f"Embedding Similarity: {sim:.3f}")

if __name__ == "__main__":
    main()
