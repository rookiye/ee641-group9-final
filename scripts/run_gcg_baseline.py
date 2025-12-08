"""
GCG Baseline Experiment Runner

This script runs GCG (Greedy Coordinate Gradient) baseline experiments for comparison with COLD.
GCG uses discrete token-level optimization instead of continuous embeddings.

Usage:
    python run_gcg_baseline.py --catalog cameras --product_idx 1 --num_iter 500

Arguments:
    --catalog: Product catalog
    --product_idx: Target product index (1-based)
    --num_iter: Number of iterations (default: 500)
    --top_k: Top-k tokens for candidate generation (default: 256)
    --num_samples: Number of candidate sequences (default: 64)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import random
import wandb
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def rank_products(text, product_names):
    """Rank products based on their position in generated text."""
    position_dict = {}
    for name in product_names:
        position = text.find(name)
        if position != -1:
            position_dict[name] = position
        else:
            position_dict[name] = float('inf')
    
    sorted_products = sorted(position_dict, key=position_dict.get)
    
    ranks = {}
    for i, prod in enumerate(sorted_products):
        if position_dict[prod] != float('inf'):
            ranks[prod] = i + 1
        else:
            ranks[prod] = len(sorted_products) + 1
    
    return ranks


def prompt_generator(target_product_idx, product_list, tokenizer, device, suffix_tokens):
    """Generate prompt with adversarial suffix."""
    # Determine catalog type from product list
    catalog_types = {
        'camera': 'camera',
        'shampoo': 'shampoo',
        'coffee': 'coffee machine',
        'book': 'book'
    }
    
    # Try to infer catalog type
    first_product_name = product_list[0]['Name'].lower()
    catalog_type = 'product'
    for key, value in catalog_types.items():
        if key in first_product_name:
            catalog_type = value
            break
    
    user_msg = f"I am looking for a {catalog_type}. Can I get some recommendations?"
    
    system_prompt = f"User: {user_msg}\\n\\nProducts:\\n"
    
    head = system_prompt
    tail = ''
    
    # Insert suffix after target product
    for i, product in enumerate(product_list):
        if i < target_product_idx:
            head += json.dumps(product) + "\\n"
        elif i == target_product_idx:
            head += json.dumps(product) + "\\n"
            tail += head[-3:]
            head = head[:-3]
        else:
            tail += json.dumps(product) + "\\n"
    
    tail += "\\n\\nAssistant: Based on your needs, here are my recommendations ranked from most to least suitable:\\n1."
    
    # Tokenization
    head_tokens = tokenizer(head, return_tensors="pt")["input_ids"].to(device)
    suffix_tokens = suffix_tokens.to(device)
    head_suffix = torch.cat((head_tokens, suffix_tokens), dim=1)
    suffix_idxs = torch.arange(head_suffix.shape[1] - suffix_tokens.shape[1], head_suffix.shape[1], device=device)
    tail_tokens = tokenizer(tail, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_prompt_ids = torch.cat((head_suffix, tail_tokens), dim=1)
    
    return full_prompt_ids, suffix_idxs


def gcg_step(input_sequence, suffix_idxs, model, loss_function, forbidden_tokens, 
             top_k=256, num_samples=32, batch_size=4):
    """Single GCG optimization step."""
    device = model.device
    embed_layer = model.get_input_embeddings()
    embedding_matrix = embed_layer.weight.data
    
    # Get embeddings and compute gradients
    input_embeddings = embed_layer(input_sequence).clone().detach()
    input_embeddings.requires_grad = True
    
    loss = loss_function(input_embeddings, model, input_sequence)
    loss.backward()
    
    gradients = input_embeddings.grad
    
    # Compute token scores
    dot_prod = torch.matmul(gradients[0], embedding_matrix.T)
    dot_prod[:, forbidden_tokens] = float('-inf')
    
    # Get top-k candidates for suffix positions
    top_k_candidates = (torch.topk(dot_prod, top_k).indices)[suffix_idxs]
    
    # Generate candidate sequences
    new_sequences = []
    for _ in range(num_samples):
        new_seq = input_sequence.clone()
        for i, idx in enumerate(suffix_idxs):
            if random.random() < 0.3:  # 30% modification probability
                new_seq[0, idx] = top_k_candidates[i][random.randint(0, top_k - 1)]
        new_sequences.append(new_seq)
    
    # Evaluate candidates
    best_loss = float('inf')
    best_sequence = input_sequence
    
    for i in range(0, len(new_sequences), batch_size):
        batch = new_sequences[i:i+batch_size]
        batch_tensor = torch.cat(batch, dim=0)
        batch_embeds = embed_layer(batch_tensor)
        
        with torch.no_grad():
            batch_losses = loss_function(batch_embeds, model, batch_tensor)
        
        min_loss_idx = batch_losses.argmin()
        if batch_losses[min_loss_idx] < best_loss:
            best_loss = batch_losses[min_loss_idx].item()
            best_sequence = batch[min_loss_idx]
    
    return best_sequence, best_loss


def loss_function_ranking(input_embeds, model, input_ids):
    """Language model loss for ranking optimization."""
    outputs = model(inputs_embeds=input_embeds, labels=input_ids)
    if input_embeds.shape[0] == 1:
        return outputs.loss
    else:
        losses = []
        for i in range(input_embeds.shape[0]):
            out = model(inputs_embeds=input_embeds[i:i+1], labels=input_ids[i:i+1])
            losses.append(out.loss)
        return torch.stack(losses)


def parse_args():
    parser = argparse.ArgumentParser(description="GCG Baseline Experiment")
    parser.add_argument("--catalog", required=True, choices=["cameras", "shampoo", "coffee_machines", "books"])
    parser.add_argument("--product_idx", type=int, required=True, help="Target product index (1-based)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_suffix_tokens", type=int, default=30)
    parser.add_argument("--test_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize W&B
    wandb.init(
        project="EE641 project",
        name=f"GCG-{args.catalog}-{args.product_idx}",
        config=vars(args)
    )
    
    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.getenv('HF_TOKEN'))
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        token=os.getenv('HF_TOKEN')
    )
    model.eval()
    device = model.device
    
    # Load products
    with open(f'data/catalogs/{args.catalog}.jsonl', 'r') as f:
        products = [json.loads(line) for line in f]
    product_list = [{"Name": p['Name'], "Natural": p['Natural']} for p in products]
    
    # Convert to 0-indexed
    target_idx = args.product_idx - 1
    product_names = [p['Name'] for p in product_list]
    target_product = product_names[target_idx]
    
    print(f"\\nTARGET: {args.product_idx}. {target_product}")
    
    # Initialize suffix tokens
    star_id = tokenizer('*', add_special_tokens=False)["input_ids"][0]
    suffix_tokens = torch.full((1, args.num_suffix_tokens), star_id, device=device)
    
    # Forbidden tokens
    forbidden_tokens = list(range(0, 10)) + [tokenizer.eos_token_id, tokenizer.pad_token_id]
    
    best_rank = float('inf')
    loss_history = []
    
    # Training loop
    pbar = tqdm(range(args.num_iter), desc="Training")
    
    for iter in pbar:
        # Generate prompt
        prompt_ids, suffix_idxs = prompt_generator(
            target_idx, product_list, tokenizer, device, suffix_tokens
        )
        
        # GCG step
        prompt_ids, curr_loss = gcg_step(
            prompt_ids, suffix_idxs, model, loss_function_ranking,
            forbidden_tokens, args.top_k, args.num_samples, args.batch_size
        )
        
        # Update suffix
        suffix_tokens = prompt_ids[0, suffix_idxs].unsqueeze(0)
        loss_history.append(curr_loss)
        
        # Log
        wandb.log({
            "iteration": iter + 1,
            "train/loss": curr_loss,
            "train/avg_loss": sum(loss_history[-100:]) / len(loss_history[-100:])
        })
        
        # Evaluate
        if (iter + 1) % args.test_iter == 0 or iter == args.num_iter - 1:
            with torch.no_grad():
                output = model.generate(
                    prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            output_text = tokenizer.decode(output[0, len(prompt_ids[0]):]).strip()
            rank = rank_products(output_text, product_names)[target_product]
            
            if rank < best_rank:
                best_rank = rank
            
            wandb.log({"eval/product_rank": rank, "best_rank": best_rank})
            pbar.set_postfix({"Rank": int(best_rank)})
    
    # Save final suffix
    final_suffix = tokenizer.decode(suffix_tokens[0], skip_special_tokens=True)
    os.makedirs("results", exist_ok=True)
    suffix_file = f"results/gcg_{args.catalog}_{args.product_idx}.txt"
    
    with open(suffix_file, "w") as f:
        f.write(final_suffix)
    
    wandb.log({
        "final_suffix": final_suffix,
        "final_rank": best_rank,
        "success": best_rank == 1
    })
    
    print(f"\\nGCG complete!")
    print(f"Final rank: {best_rank}")
    print(f"Suffix saved to: {suffix_file}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
