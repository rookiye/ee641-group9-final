"""Prompt Generation Utilities"""

def generate_product_prompt(products, user_msg, target_idx=None, suffix=""):
    """Generate prompt with product list."""
    prompt = f"User: {user_msg}\\n\\nProducts:\\n"
    for i, p in enumerate(products):
        desc = p['Natural']
        if target_idx is not None and i == target_idx:
            desc += f" {suffix}"
        prompt += f"{i+1}. {p['Name']}: {desc}\\n"
    prompt += "\\n\\nAssistant: Based on your needs, here are my recommendations:\\n1."
    return prompt
