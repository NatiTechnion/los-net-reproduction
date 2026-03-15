import matplotlib.pyplot as plt
import numpy as np

data = {
    'HotpotQA': {
        'L-3-8b': [(64.73, 0.13), (66.69, 0.97), (72.33, 0.52)],
        'Mis-7b': [(69.86, 0.45), (69.44, 0.59), (73.16, 0.68)],
        'Q-2.5-7b': [(71.91, 0.47), (70.68, 0.83), (75.73, 1.74)]
    },
    'IMDB': {
        'L-3-8b': [(88.83, 0.42), (83.59, 1.30), (90.71, 0.89)],
        'Mis-7b': [(90.88, 0.61), (90.22, 3.24), (94.03, 0.64)],
        'Q-2.5-7b': [(85.55, 0.18), (88.40, 0.44), (88.92, 0.92)]
    },
    'Movies': {
        'L-3-8b': [(73.36, 0.23), (76.62, 0.36), (77.45, 0.73)],
        'Mis-7b': [(66.74, 0.63), (68.10, 0.43), (71.91, 0.43)],
        'Q-2.5-7b': [(73.14, 0.56), (75.53, 1.84), (86.23, 0.29)]
    }
}

models = ['ATP+R-MLP', 'ATP+R-Transf.', 'LOS-Net']
datasets = ['HotpotQA', 'IMDB', 'Movies']
llms = ['L-3-8b', 'Mis-7b', 'Q-2.5-7b']
colors = ['#8db4e2', '#e28d8d', '#8db4b4']

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for d_idx, ds in enumerate(datasets):
    for l_idx, llm in enumerate(llms):
        ax = axes[d_idx, l_idx]
        vals = data[ds][llm]
        means = [v[0] for v in vals]
        stds = [v[1] for v in vals]
        
        ax.bar(models, means, yerr=stds, color=colors[d_idx], capsize=5, alpha=0.8)
        ax.axhline(y=means[2], color=colors[d_idx], linestyle='-', linewidth=1, alpha=0.6)
        
        ax.set_title(f"{ds}, {llm}", fontweight='bold')
        ax.set_ylabel("Test AUC")
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right')
        
        ax.set_ylim(min(means) - 5, max(means) + 3)

plt.tight_layout()
plt.savefig("figure6.png", dpi=300)