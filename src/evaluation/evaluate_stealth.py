"""Stealth Metrics Evaluation"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(text, model, tokenizer):
    """Calculate perplexity of text."""
    if not text or len(text.strip()) == 0:
        return float('inf')
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
        return torch.exp(outputs.loss).item()

def calculate_embedding_similarity(text1, text2, model, tokenizer):
    """Calculate cosine similarity between embeddings."""
    if not text1 or not text2:
        return 0.0
    embed_layer = model.get_input_embeddings()
    enc1 = tokenizer(text1, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    enc2 = tokenizer(text2, return_tensors='pt', truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        emb1 = embed_layer(enc1.input_ids).mean(dim=1)
        emb2 = embed_layer(enc2.input_ids).mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        return similarity.item()
