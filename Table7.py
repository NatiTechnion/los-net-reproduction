import wandb

# Initialize W&B API
api = wandb.Api()

# Your specific project path
PROJECT_PATH = "natipro/LOS-Net-Standard"

print(f"Fetching runs from {PROJECT_PATH}...\n")
runs = api.runs(PROJECT_PATH)

best_runs = {}

# =====================================================================
# EXACT KEYS BASED ON YOUR CONFIG AND SUMMARY
# =====================================================================
DATASET_KEY = "train_dataset"  # e.g., 'movies'
MODEL_KEY = "LLM"              # e.g., 'mistralai/Mistral-7B-Instruct-v0.2'
VAL_AUC_KEY = "best_val_AUC"   # e.g., 0.7018...
# =====================================================================

# Helper function to clean up names for the table
def clean_name(model_string, dataset_string):
    # Clean model name
    if "Mistral" in model_string: m_name = "Mis-7b"
    elif "Llama-3" in model_string: m_name = "L-3-8b"
    elif "Qwen" in model_string: m_name = "Q-2.5-7b"
    else: m_name = model_string
    
    # Clean dataset name
    d_name = dataset_string.replace('_test', '').capitalize()
    if d_name == 'Hotpotqa': d_name = 'HotpotQA'
    if d_name == 'Imdb': d_name = 'IMDB'
    
    return m_name, d_name

for run in runs:
    if run.state != "finished":
        continue

    config = run.config
    summary = run.summary

    # Extract raw dataset and model
    raw_dataset = config.get(DATASET_KEY)
    raw_model = config.get(MODEL_KEY)
    
    if not raw_dataset or not raw_model:
        continue

    # Extract the validation AUC
    val_auc = summary.get(VAL_AUC_KEY)
    if val_auc is None:
        continue

    # Clean the names up
    model, dataset = clean_name(raw_model, raw_dataset)
    combo_key = (model, dataset)

    # Save the run if it's the first one we've seen, or if it has a higher Val AUC
    if combo_key not in best_runs or val_auc > best_runs[combo_key]['val_auc']:
        best_runs[combo_key] = {
            'id': run.id,
            'name': run.name,
            'val_auc': val_auc,
            'test_auc': summary.get('best_test_AUC', 'N/A'),
            'runtime_seconds': summary.get('_runtime', 0), # W&B natively logs this
            'epochs': summary.get('epoch', 0),
            'config': {k: v for k, v in config.items() if not k.startswith('_')} 
        }

# Print the results perfectly formatted for Table 7
print("="*65)
print(f"{' BEST RUNS BY VALIDATION AUC (For Table 7) ':=^65}")
print("="*65)

# Sort so they print in a nice order
for combo in sorted(best_runs.keys()):
    model, dataset = combo
    data = best_runs[combo]
    
    # Calculate minutes and seconds
    runtime_mins = int(data['runtime_seconds'] // 60)
    runtime_secs = int(data['runtime_seconds'] % 60)

    print(f"Model:   {model}")
    print(f"Dataset: {dataset}")
    print(f"Run ID:  {data['id']} ({data['name']})")
    print(f"Val AUC: {data['val_auc']:.4f}")
    print(f"Runtime: {runtime_mins}m {runtime_secs}s  <--- DROP THIS INTO TABLE 7")
    print(f"Test AUC:{data['test_auc']:.4f} (Finished at Epoch {data['epochs']})")
    print("-" * 65)