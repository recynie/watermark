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

# Pivot table to create matrix format
matrix = df.pivot_table(
    values='delta',
    index='watermark',
    columns='attack',
    aggfunc='mean'
)

# Create heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', 
            vmin=0, vmax=10, center=5,
            cbar_kws={'label': 'Average Delta Value'})

plt.title('Average Delta Scores by Watermark and Attack Type')
plt.tight_layout()
plt.savefig('viz/matrix.png')
plt.show()
