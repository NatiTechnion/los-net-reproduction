import wandb
from collections import defaultdict

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ENTITY = "natipro"  # Based on your previous W&B links
PROJECT = "LOS-Net-Extras"

def get_best_single_seed_runs():
    api = wandb.Api()
    
    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    # Dictionary to store the best run info for each combination
    # Key: (LLM, dataset)
    best_runs = defaultdict(lambda: {
        "run_id": None, 
        "val_auc": -1, 
        "test_auc": -1, 
        "url": None
    })
    
    valid_runs_counted = 0
    
    for run in runs:
        # Only look at successfully finished runs
        if run.state != "finished":
            continue
            
        config = run.config
        summary = run.summary
        
        # Extract model and dataset, handling list formats just in case
        llm = config.get("LLM", config.get("llm", config.get("model_name", "Unknown_LLM")))
        dataset = config.get("train_dataset", config.get("dataset", "Unknown_Dataset"))
        
        if isinstance(llm, list): llm = llm[0]
        if isinstance(dataset, list): dataset = dataset[0]
        
        if llm == "Unknown_LLM" or dataset == "Unknown_Dataset":
            continue
            
        # Extract the AUC metrics
        val_auc = summary.get("best_val_AUC")
        test_auc = summary.get("best_test_AUC")
        
        # Skip runs that crashed before logging AUCs
        if val_auc is None or test_auc is None:
            continue
            
        combination_key = (llm, dataset)
        
        # If this run has a higher Validation AUC, it becomes the new "best" run
        if val_auc > best_runs[combination_key]["val_auc"]:
            best_runs[combination_key] = {
                "run_id": run.id,
                "val_auc": val_auc,
                "test_auc": test_auc,
                "url": run.url
            }
            
        valid_runs_counted += 1
        
    print(f"\nSuccessfully parsed {valid_runs_counted} finished runs.")
    print("==================================================")
    print("BEST TEST AUCs (SELECTED VIA MAX VALIDATION AUC)")
    print("==================================================\n")
    
    # Sort and print the results
    for (llm, dataset), data in sorted(best_runs.items()):
        short_llm = llm.split('/')[-1] if '/' in llm else llm
        
        # Format AUCs as percentages for easy reading/copying
        val_pct = data["val_auc"] * 100
        test_pct = data["test_auc"] * 100
        
        print(f"[{short_llm} | {dataset}]")
        print(f"  Selected Val AUC:  {val_pct:.2f}")
        print(f"  Reported Test AUC: {test_pct:.2f}  <-- USE THIS FOR YOUR TABLE")
        print(f"  Run ID:            {data['run_id']}")
        print(f"  URL:               {data['url']}\n")

if __name__ == "__main__":
    get_best_single_seed_runs()