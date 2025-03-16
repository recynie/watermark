import json
import pandas as pd
import plotly.graph_objects as go

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

# Create separate plot for each attack type
for attack_type in stats['attack'].unique():
    attack_data = stats[stats['attack'] == attack_type]
    
    # Create new figure for this attack
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=attack_data['watermark'],
        open=attack_data['q1'],
        high=attack_data['max'],
        low=attack_data['min'],
        close=attack_data['q3'],
        name='Statistics',
        increasing_line_color='blue',
        decreasing_line_color='blue',
        increasing_fillcolor='rgba(0, 0, 255, 0.3)',
        decreasing_fillcolor='rgba(0, 0, 255, 0.3)'
    ))

    # Update layout for this attack
    fig.update_layout(
        title=f'{attack_type.capitalize()} Attack Statistics by Watermark',
        yaxis_title='Delta Score',
        xaxis_title='Watermark Algorithm',
        showlegend=True,
        xaxis_tickangle=-45
    )

    # Save the plots for this attack
    fig.write_html(f"viz/{attack_type}_analysis.html")
    fig.write_image(f"viz/{attack_type}_analysis.png")
