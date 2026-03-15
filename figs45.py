import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing your newly generated CSVs
RESULTS_DIR = "Results/transferability_HD||num_epochs_10_lr_0.0001_num_seeds_3"

# Mappings to match the exact labels used in the paper
LLM_MAP = {
    'meta-llama': 'L-3-8b', 'meta-llama/Meta-Llama-3-8B-Instruct': 'L-3-8b',
    'mistralai': 'Mis-7b', 'mistralai/Mistral-7B-Instruct-v0.2': 'Mis-7b',
    'Qwen': 'Q-2.5-7b', 'Qwen/Qwen2.5-7B-Instruct': 'Q-2.5-7b'
}
DS_MAP = {'imdb': 'IMDB', 'movies': 'Movies', 'hotpotqa': 'HotpotQA'}

# The absolute best probability baselines from Table 1 and Table 5 to check for bolding
BEST_BASELINES = {
    'Mis-7b': {'IMDB': 57.58, 'Movies': 53.34, 'HotpotQA': 51.49},
    'L-3-8b': {'IMDB': 63.32, 'Movies': 56.42, 'HotpotQA': 53.30},
    'Q-2.5-7b': {'IMDB': 58.22, 'Movies': 70.27, 'HotpotQA': 65.91}
}

def parse_cell(cell_str):
    """Converts '0.8905 ± 0.0040' to (89.05, 0.40)"""
    if isinstance(cell_str, str) and '±' in cell_str:
        mean_str, std_str = cell_str.split('±')
        return float(mean_str.strip()) * 100, float(std_str.strip()) * 100
    return 0.0, 0.0

def load_matrices(file_prefix, entity_name):
    """Loads the FT and Scratch matrices and returns means, stds, and scratch_means"""
    ft_path = os.path.join(RESULTS_DIR, f"HD_Finetuning_{file_prefix}_{entity_name}.csv")
    scratch_path = os.path.join(RESULTS_DIR, f"HD_Training_from_scratch_{file_prefix}_{entity_name}.csv")
    
    df_ft = pd.read_csv(ft_path, index_col=0)
    df_scratch = pd.read_csv(scratch_path, index_col=0)
    
    means = np.zeros(df_ft.shape)
    stds = np.zeros(df_ft.shape)
    scratch_means = np.zeros(df_scratch.shape)
    
    for i in range(df_ft.shape[0]):
        for j in range(df_ft.shape[1]):
            means[i, j], stds[i, j] = parse_cell(df_ft.iloc[i, j])
            scratch_means[i, j], _ = parse_cell(df_scratch.iloc[i, j])
            
    return means, stds, scratch_means, list(df_ft.index), list(df_ft.columns)

def plot_heatmap_row(setup_type):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = sns.color_palette("coolwarm_r", as_cmap=True)
    
    if setup_type == 'cross_models':
        entities = ['imdb', 'movies', 'hotpotqa']
        file_prefix = 'cross_models_results_for_dataset'
        title_prefix = 'Dataset: '
    else:
        entities = ['meta-llama', 'mistralai', 'Qwen']
        file_prefix = 'cross_datasets_results_for_model'
        title_prefix = 'LLM: '

    for idx, entity in enumerate(entities):
        ax = axes[idx]
        means, stds, scratch_means, row_labels, col_labels = load_matrices(file_prefix, entity)
        
        # Format labels
        if setup_type == 'cross_models':
            row_labels = [LLM_MAP.get(r, r) for r in row_labels]
            col_labels = [LLM_MAP.get(c, c) for c in col_labels]
            plot_title = DS_MAP.get(entity, entity)
        else:
            row_labels = [DS_MAP.get(r, r) for r in row_labels]
            col_labels = [DS_MAP.get(c, c) for c in col_labels]
            plot_title = LLM_MAP.get(entity, entity)

        sns.heatmap(means, annot=False, cmap=cmap, fmt="", 
                    xticklabels=col_labels, yticklabels=row_labels, 
                    vmin=50, vmax=95, cbar=(idx==2), 
                    cbar_kws={'label': 'Test AUC'} if idx==2 else None, ax=ax)

        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                m, s, sm = means[i, j], stds[i, j], scratch_means[i, j]
                
                # Logic 1: Does it beat training from scratch?
                asterisk = " *" if m > sm else ""
                
                # Logic 2: Does it beat the baseline?
                target_llm = col_labels[j] if setup_type == 'cross_models' else plot_title
                target_ds = plot_title if setup_type == 'cross_models' else col_labels[j]
                is_bold = m > BEST_BASELINES.get(target_llm, {}).get(target_ds, 100)
                
                weight = 'bold' if is_bold else 'normal'
                text = f"{m:.2f}{asterisk}\n$\pm$ {s:.2f}"
                color = 'white' if m > 78 else 'black'
                
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                        color=color, fontweight=weight, fontsize=11)

        ax.set_xlabel('Target', fontweight='bold')
        ax.set_ylabel('Source', fontweight='bold')
        ax.set_title(plot_title, fontweight='bold', fontsize=14)

    plt.tight_layout()
    out_name = f"Figure_{'4_Cross_LLM' if setup_type == 'cross_models' else '5_Cross_Dataset'}.png"
    plt.savefig(out_name, dpi=300)
    print(f"Saved {out_name}")

if __name__ == "__main__":
    print("Generating Figure 4 (Cross-LLM)...")
    plot_heatmap_row('cross_models')
    
    print("Generating Figure 5 (Cross-Dataset)...")
    plot_heatmap_row('cross_datasets')