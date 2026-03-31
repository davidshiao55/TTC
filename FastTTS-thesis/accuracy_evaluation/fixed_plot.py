import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Your data
result_num = {'aime': {'1.5B-7B': {'baseline': {'128': 5.0, '16': 4.0, '512': 5.0},
                      'spec_prefix': {'128': 5.0, '16': 4.0, '512': 7.0}},
          '7B-1.5B': {'baseline': {'128': 6.0, '16': 4.0, '512': 8.0},
                      'spec_prefix': {'128': 6.0, '16': 6.0, '512': 7.0}}},
 'amc': {'1.5B-7B': {'baseline': {'128': 22.0, '16': 16.0, '512': 22.0},
                     'spec_prefix': {'128': 23.0, '16': 16.0, '512': 23.0}},
         '7B-1.5B': {'baseline': {'128': 26.0, '16': 24.0, '512': 29.0},
                     'spec_prefix': {'128': 26.0, '16': 24.0, '512': 28.0}}}}

# Prepare data for plotting
plot_data = []
for dataset in ['aime', 'amc']:
    for model_size in result_num[dataset]:
        for prompt_type in result_num[dataset][model_size]:
            for n, num_correct in result_num[dataset][model_size][prompt_type].items():
                plot_data.append({
                    'Dataset': dataset.upper(),
                    'Model Size': model_size,
                    'Prompt Type': prompt_type,
                    'N': int(n),
                    'Correct': num_correct
                })

df = pd.DataFrame(plot_data)

# For better grouping, combine Model Size and Prompt Type
df['Method'] = df['Model Size'] + ' ' + df['Prompt Type']

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Create a shared legend
handles = []
labels = []

for i, dataset in enumerate(['AIME', 'AMC']):
    ax = axes[i]
    # Filter data for current dataset
    data = df[df['Dataset'] == dataset].copy()
    
    if not data.empty:
        # Sort N for consistent x-axis
        data = data.sort_values('N')
        
        # Draw barplot
        sns.barplot(
            data=data,
            x='N',
            y='Correct',
            hue='Method',
            ax=ax,
            palette='Set2'
        )
        
        ax.set_title(f"{dataset} Results", fontsize=14, fontweight='bold')
        ax.set_xlabel("N", fontsize=12)
        if i == 0:  # Only add y-label to left subplot
            ax.set_ylabel("Number of Correct Answers", fontsize=12)
        
        # Collect handles and labels for shared legend
        if i == 0:  # Only collect from first subplot
            handles, labels = ax.get_legend_handles_labels()
        
        # Remove individual legends
        ax.get_legend().remove()
        
        # Set x-ticks
        ax.set_xticks(sorted(data['N'].unique()))
    else:
        print(f"Warning: No data for dataset {dataset}")

# Add shared legend outside the plots
fig.legend(handles, labels, title="Method", loc='center right', bbox_to_anchor=(1.15, 0.5))

plt.tight_layout()
plt.show() 