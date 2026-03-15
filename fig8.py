import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Clean up names so W&B strings match our dictionary keys perfectly
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

api = wandb.Api()
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

print("Pulling K <= 500 from LOS-Net-k...")
runs_k = api.runs("natipro/LOS-Net-k")

# Tracker for K <= 500 just to be absolutely safe
seed_tracker_k = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

for r in runs_k:
    if r.state == "finished" and "best_test_AUC" in r.summary:
        c = r.config
        ds = clean_ds(c.get("train_dataset"))
        llm = clean_llm(c.get("LLM"))
        k = c.get("topk_dim")
        seed = c.get("seed", -1)
        
        if k in [10, 50, 100, 500]:
            if seed not in seed_tracker_k[ds][llm][k]:
                seed_tracker_k[ds][llm][k].add(seed)
                results[ds][llm][k].append(r.summary["best_test_AUC"] * 100)

print("Pulling K = 1000 from LOS-Net-Standard...")
runs_std = api.runs("natipro/LOS-Net")

# Tracker to ensure we only get exactly one of each seed for K=1000
seed_tracker_std = defaultdict(lambda: defaultdict(set))

for r in runs_std:
    if r.state == "finished" and "best_test_AUC" in r.summary:
        c = r.config
        
        # STRICT FILTER: Matches exact hyperparameters and forces 300 epochs
        if (c.get("probe_model") == "LOS-Net" and 
            c.get("num_epochs") == 300 and
            c.get("hidden_dim") == 128 and 
            c.get("num_layers") == 1 and 
            c.get("heads") == 8 and
            c.get("dropout") == 0.3 and 
            float(c.get("lr", 0)) == 1e-4 and 
            float(c.get("weight_decay", 0)) == 0.001):
            
            ds = clean_ds(c.get("train_dataset"))
            llm = clean_llm(c.get("LLM"))
            seed = c.get("seed", -1)
            
            # Only append if we haven't seen this seed yet for this config
            if seed not in seed_tracker_std[ds][llm]:
                seed_tracker_std[ds][llm].add(seed)
                results[ds][llm][1000].append(r.summary["best_test_AUC"] * 100)

datasets = [('movies', 'Movies', '#8db4b4'), ('hotpotqa', 'HotpotQA', '#8db4e2'), ('imdb', 'IMDB', '#e28d8d')]
llms = [('llama', 'L-3-8b'), ('mistral', 'Mis-7b'), ('qwen', 'Q-2.5-7b')]
k_vals = [10, 50, 100, 500, 1000]

print("\n--- Data Check (Should be exactly 3 runs per K) ---")
for ds_key, ds_name, _ in datasets:
    for llm_key, llm_name in llms:
        counts = {k: len(results[ds_key][llm_key][k]) for k in k_vals}
        print(f"{ds_name} | {llm_name}: {counts}")

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.35)

for row, (ds_key, ds_name, color) in enumerate(datasets):
    for col, (llm_key, llm_name) in enumerate(llms):
        ax = axes[row, col]
        
        means = []
        stds = []
        for k in k_vals:
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
            min_val = min([m - s for m, s in zip(means, stds) if m > 0])
            max_val = max([m + s for m, s in zip(means, stds) if m > 0])
            padding = (max_val - min_val) * 0.1 # Just a tiny 10% visual pad
            ax.set_ylim(min_val - padding, max_val + padding)

plt.tight_layout()
plt.savefig("figure82.png", dpi=300)
print("\nPlot saved as figure82.png")