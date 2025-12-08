"""Transferability Evaluation"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .evaluate_ranking import evaluate_suffix

def evaluate_transfer(source_model_name, target_model_name, products, target_idx, suffix):
    """Test if suffix transfers across models."""
    # Load target model
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        target_model_name, torch_dtype=torch.float16, device_map='auto').eval()
    
    # Evaluate on target model
    rank = evaluate_suffix(model, tokenizer, products, target_idx, suffix)
    return rank, rank == 1
