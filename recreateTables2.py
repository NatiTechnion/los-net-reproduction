import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("natipro/LOS-Net-Transferability")

data = []

print("Fetching runs and filtering by best validation AUC...")
for run in runs:
    # 1. Grab your config keys
    dataset_raw = run.config.get("train_dataset", "")
    llm_raw = run.config.get("LLM", "")
    method_raw = run.config.get("probe_model", "")
    
    # Grab the hyperparameters that vary in your sweep
    hidden_dim = run.config.get("hidden_dim", 0)
    num_layers = run.config.get("num_layers", 0)
    dropout = run.config.get("dropout", 0)
    weight_decay = run.config.get("weight_decay", 0)
    
    # 2. Grab exact summary metrics you found
    val_auc = run.summary.get("best_val_AUC")
    test_auc = run.summary.get("best_test_AUC")
    
    # 3. Map raw dataset names
    dataset = ""
    if "hotpotqa" in dataset_raw.lower(): dataset = "HotpotQA"
    elif "imdb" in dataset_raw.lower(): dataset = "IMDB"
    elif "movies" in dataset_raw.lower(): dataset = "Movies"
    
    # 4. Map raw LLM paths to the exact Table 1 headers
    llm = ""
    llm_raw_lower = llm_raw.lower()
    if "mistralai/mistral-7b-instruct-v0.2" in llm_raw_lower: llm = "Mistral-7b-instruct"
    elif "meta-llama/meta-llama-3-8b-instruct" in llm_raw_lower: llm = "Llama3-8b-instruct"
    elif "qwen/qwen2.5-7b-instruct" in llm_raw_lower: llm = "Qwen-2.5-7b" 
    
    # 5. Map the method name
    method = ""
    if method_raw == "LOS-Net": method = "LOS-Net"
    elif "MLP" in method_raw: method = "ATP+R-MLP"
    elif "Transf" in method_raw: method = "ATP+R-Transf."
    
    # If we have valid data, append it
    if dataset and llm and method and val_auc is not None and test_auc is not None:
        data.append({
            "Method": method,
            "LLM": llm,
            "Dataset": dataset,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "Val_AUC": val_auc,
            "Test_AUC": test_auc
        })

df = pd.DataFrame(data)

if df.empty:
    print("No matching runs found with both Val and Test AUC!")
else:
    # Step A: Group by configs to evaluate specific setups
    config_groups = df.groupby(["Method", "LLM", "Dataset", "hidden_dim", "num_layers", "dropout", "weight_decay"])
    
    # Calculate mean Val and Test AUC for each setup
    agg_df = config_groups.agg(
        mean_val_auc=('Val_AUC', 'mean'),
        mean_test_auc=('Test_AUC', 'mean'),
        std_test_auc=('Test_AUC', 'std')
    ).reset_index()
    
    # Step B: Find the config with the MAX Validation AUC
    best_idx = agg_df.groupby(["Method", "LLM", "Dataset"])['mean_val_auc'].idxmax()
    best_configs = agg_df.loc[best_idx].copy()
    
    # Step C: Format exactly like the paper using only the best test metrics
    best_configs['Formatted_AUC'] = (best_configs['mean_test_auc'] * 100).round(2).astype(str) + " ± " + (best_configs['std_test_auc'] * 100).round(2).astype(str)
    
    # Pivot to look like Table 1
    pivot_table = best_configs.pivot(index=["Method"], columns=["LLM", "Dataset"], values="Formatted_AUC")
    final_table = pivot_table.fillna("Missing")
    
    print("\n--- REPRODUCED TABLE 1 (Filtered by Validation) ---")
    print(final_table)
    
    # --- SAVE TO FILE ---
    with open("result_run1.txt", "w") as f:
        f.write("--- REPRODUCED TABLE 1 (Filtered by Validation) ---\n\n")
        f.write(final_table.to_string())
        
    print("\nSaved successfully to result_run1.txt (including rows, cols, and formatting).")