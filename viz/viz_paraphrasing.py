import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read JSONL file
data = []
with open('archieve/result.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)
df = df[df['fail'] == False]

# Group by watermark and attack, then calculate statistics
stats = df.groupby(['watermark', 'attack']).agg({
    'delta': ['min', 'max', 'mean', 'std', 
              lambda x: x.quantile(0.25),
              lambda x: x.quantile(0.75)]
}).reset_index()

stats.columns = ['watermark', 'attack', 'min', 'max', 'mean', 'std', 'q1', 'q3']

# Set style for all plots
plt.style.use('default')  # Use default style as base
sns.set_theme(style="whitegrid")  # Apply seaborn styling
sns.set_palette("husl")

# Create separate plot for each attack type
for attack_type in stats['attack'].unique():
    # Create new figure with white background
    plt.figure(figsize=(12, 7), facecolor='white')
    
    # Create box plot
    sns.boxplot(x='watermark', y='delta', data=df[df['attack'] == attack_type],
                color='lightblue', width=0.5)

    # Customize the plot
    plt.title(f'{attack_type.capitalize() if attack_type!="cwra" else attack_type.upper()} Attack Statistics by Watermark', 
              fontsize=24, pad=20)
    plt.xlabel('Watermark Algorithm', fontsize=18)
    plt.ylabel('Delta Score', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.tick_params(labelsize=14)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PNG with high DPI for better quality
    plt.savefig(f"viz/viz-attack/{attack_type}_analysis.png", 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
