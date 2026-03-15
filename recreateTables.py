import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
runs = api.runs("natipro/LOS-Net-Standard")  # change if needed

data = []

print("Fetching runs...")

for run in runs:
    if run.state != "finished":
        continue

    dataset_raw = run.config.get("train_dataset", "")
    llm_raw = run.config.get("LLM", "")
    method = run.config.get("probe_model", "")
    seed = run.config.get("seed")

    K = run.config.get("topk_preprocess")
    if K != 1000:
        continue

    if method != "LOS-Net":
        continue

    val_auc = run.summary.get("best_val_AUC")
    test_auc = run.summary.get("best_test_AUC")

    if val_auc is None or test_auc is None:
        continue

    # Map dataset
    dataset = None
    d_lower = dataset_raw.lower()
    if "hotpotqa" in d_lower:
        dataset = "HotpotQA"
    elif "imdb" in d_lower:
        dataset = "IMDB"
    elif "movies" in d_lower:
        dataset = "Movies"

    # Map LLM
    llm = None
    l_lower = llm_raw.lower()
    if "mistral-7b-instruct" in l_lower:
        llm = "Mistral-7b-instruct"
    elif "llama-3-8b-instruct" in l_lower:
        llm = "Llama3-8b-instruct"
    elif "qwen2.5-7b-instruct" in l_lower:
        llm = "Qwen-2.5-7b"

    if not dataset or not llm:
        continue

    data.append({
        "LLM": llm,
        "Dataset": dataset,
        "seed": seed,
        "hidden_dim": run.config.get("hidden_dim"),
        "num_layers": run.config.get("num_layers"),
        "dropout": run.config.get("dropout"),
        "weight_decay": run.config.get("weight_decay"),
        "Val_AUC": val_auc,
        "Test_AUC": test_auc
    })

df = pd.DataFrame(data)

if df.empty:
    print("No valid runs found.")
    exit()

# --------------------------------------------
# 1️⃣ Group by hyperparameter config
# --------------------------------------------
config_cols = [
    "LLM", "Dataset",
    "hidden_dim", "num_layers",
    "dropout", "weight_decay"
]

agg_configs = df.groupby(config_cols).agg(
    mean_val=("Val_AUC", "mean"),
    mean_test=("Test_AUC", "mean"),
    std_test=("Test_AUC", "std"),
    seed_count=("seed", "nunique")
).reset_index()

# Ensure exactly 3 seeds per config
agg_configs = agg_configs[agg_configs["seed_count"] == 3]

# --------------------------------------------
# 2️⃣ Select best config per (LLM, Dataset)
# --------------------------------------------
best_idx = agg_configs.groupby(["LLM", "Dataset"])["mean_val"].idxmax()
final_results = agg_configs.loc[best_idx].copy()

# --------------------------------------------
# 3️⃣ Format as mean ± std (percentage)
# --------------------------------------------
final_results["Formatted"] = (
    (final_results["mean_test"] * 100).round(2).astype(str)
    + " ± "
    + (final_results["std_test"] * 100).round(2).astype(str)
)

final_results["Method"] = "LOS-Net"

# --------------------------------------------
# 4️⃣ Pivot to Table 1 format
# --------------------------------------------
table = final_results.pivot(
    index="Method",
    columns=["LLM", "Dataset"],
    values="Formatted"
)

print("\n--- REPRODUCED TABLE 1 (LOS-Net row) ---\n")
print(table)

with open("table1_losnet.txt", "w") as f:
    f.write(table.to_string())

print("\nSaved to table1_losnet.txt")