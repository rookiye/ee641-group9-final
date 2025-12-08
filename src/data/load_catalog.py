"""Product Catalog Loading"""
import json

def load_catalog(catalog_name):
    """Load product catalog from JSONL file."""
    products = []
    with open(f'data/catalogs/{catalog_name}.jsonl', 'r') as f:
        for line in f:
            products.append(json.loads(line))
    return products
