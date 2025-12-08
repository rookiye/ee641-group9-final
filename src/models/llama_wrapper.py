"""LLaMA Model Wrapper"""
import torch, transformers

def load_llama_model(model_path, precision=16, device='cuda'):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=eval(f'torch.float{precision}'),
        trust_remote_code=True, low_cpu_mem_usage=True, use_cache=False
    ).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    if 'llama' in model_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    return model, tokenizer
