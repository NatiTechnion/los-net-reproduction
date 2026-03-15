import wandb
import numpy as np
from collections import defaultdict

def get_k_results():
    api = wandb.Api()
    apm = {10: 99.49, 50: 99.80, 100: 99.85, 500: 99.99, 1000: 99.99}
    results = defaultdict(list)
    k_sweep = api.sweep("LOS-Net-k/sweeps/aki4g8eg")
    for run in k_sweep.runs:
        if run.state != "finished" or "best_test_AUC" not in run.summary:
            continue
        k_val = run.config.get("topk_dim")
        if k_val in (10, 50, 100, 500):
            results[k_val].append(run.summary["best_test_AUC"] * 100.0)
    std_sweep = api.sweep("natipro/LOS-Net-Standard/sweeps/uc39o6dp")
    for run in std_sweep.runs:
        if run.state != "finished" or "best_test_AUC" not in run.summary:
            continue
        cfg = run.config
        if cfg.get("probe_model") == "LOS-Net" and cfg.get("num_epochs") == 300 and cfg.get("hidden_dim") == 128 and cfg.get("num_layers") == 1 and cfg.get("dropout") == 0.3 and float(cfg.get("weight_decay", 1.0)) == 0.0:
            results[1000].append(run.summary["best_test_AUC"] * 100.0)
    for k in apm:
        scores = results[k]
        if len(scores) == 0:
            print("k =", k, " apm% =", apm[k], " test auc = N/A")
            continue
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        print("k =", k, " apm% =", apm[k], " test auc = %.2f ± %.2f" % (mean_auc, std_auc))

if __name__ == "__main__":
    get_k_results()