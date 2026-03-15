import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def clean_llm(name):
    s = str(name).lower()
    if "llama" in s: return "llama"
    if "mistral" in s: return "mistral"
    if "qwen" in s: return "qwen"
    return s

def clean_ds(name):
    s = str(name).lower()
    if "hotpot" in s: return "hotpotqa"
    if "imdb" in s: return "imdb"
    if "movie" in s: return "movies"
    return s

# Hardcoding the verified K=1000 results from your Table 1 & Table 5
# Format: (mean, std)
k1000_data = {
    'movies': {
        'llama': (77.45, 0.73),
        'mistral': (71.91, 0.43),
        'qwen': (86.23, 0.29)
    },
    'hotpotqa': {
        'llama': (72.33, 0.52),
        'mistral': (73.16, 0.68),
        'qwen': (75.73, 1.74)
    },
    'imdb': {
        'llama': (90.71, 0.89),
        'mistral': (94.03, 0.64),
        'qwen': (88.92, 0.92)
    }
}

api = wandb.Api()
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

print("Pulling K <= 500 from LOS-Net-k...")
runs_k = api.runs("natipro/LOS-Net-k")
for r in runs_k:
    if r.state == "finished" and "best_test_AUC" in r.summary:
        c = r.config
        ds = clean_ds(c.get("train_dataset"))
        llm = clean_llm(c.get("LLM"))
        k = c.get("topk_dim")
        if k in [10, 50, 100, 500]:
            results[ds][llm][k].append(r.summary["best_test_AUC"] * 100)

datasets = [('movies', 'Movies', '#8db4b4'), ('hotpotqa', 'HotpotQA', '#8db4e2'), ('imdb', 'IMDB', '#e28d8d')]
llms = [('llama', 'L-3-8b'), ('mistral', 'Mis-7b'), ('qwen', 'Q-2.5-7b')]
k_vals = [10, 50, 100, 500, 1000]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.35)

for row, (ds_key, ds_name, color) in enumerate(datasets):
    for col, (llm_key, llm_name) in enumerate(llms):
        ax = axes[row, col]
        
        means = []
        stds = []
        for k in k_vals:
            if k == 1000:
                # Inject the pristine table data
                means.append(k1000_data[ds_key][llm_key][0])
                stds.append(k1000_data[ds_key][llm_key][1])
            else:
                scores = results[ds_key][llm_key][k]
                if scores:
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                else:
                    means.append(0)
                    stds.append(0)
                
        # Plot the bars
        ax.bar(range(len(k_vals)), means, yerr=stds, color=color, capsize=0, alpha=0.9, error_kw=dict(ecolor='gray', lw=2))
        
        ax.set_title(f"{ds_name}, {llm_name}")
        ax.set_ylabel("Test AUC")
        ax.set_xlabel("Top-K")
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels(k_vals)
        
        valid_means = [m for m in means if m > 0]
        if valid_means:
            ax.set_ylim(min(valid_means) - 1.5, max(valid_means) + 1.5)

plt.tight_layout()
plt.savefig("figure83.png", dpi=300)
print("\nPlot saved as figure83.png")