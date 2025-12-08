"""Generate Figures from W&B Data"""
import wandb, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def main():
    api = wandb.Api()
    runs = api.runs("vthiyaga-usc/EE641 project")
    
    # Collect data
    data = []
    for run in runs:
        if run.state == 'finished':
            data.append({
                'method': run.config.get('method', 'Unknown'),
                'catalog': run.config.get('catalog', ''),
                'rank': run.summary.get('eval/product_rank', -1)
            })
    
    df = pd.DataFrame(data)
    
    # Generate comparison plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='method', y='rank')
    plt.title('Method Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Product Rank')
    plt.savefig('notebooks/figures/method_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: method_comparison.png")

if __name__ == "__main__":
    main()
