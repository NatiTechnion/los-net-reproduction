import wandb
from collections import defaultdict

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Replace with your W&B username or team name
ENTITY = "natipro" 
PROJECT = "LOS-Net-Original"

def get_max_runtime_runs():
    api = wandb.Api()
    
    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    # Grab all runs from the specified project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    # Dictionary to store the max runtime data for each combination
    # Key: (LLM, dataset), Value: dict of run info
    max_runtime_runs = defaultdict(lambda: {"run_id": None, "run_name": None, "runtime": -1, "url": None})
    
    valid_runs_counted = 0
    
    for run in runs:
        # We only want to look at runs that successfully finished
        if run.state != "finished":
            continue
            
        config = run.config
        
        # Extract the model and dataset from the run's config
        # .get() prevents KeyError if a run happens to be missing these fields
        llm = config.get("LLM", "Unknown_LLM")
        dataset = config.get("train_dataset", "Unknown_Dataset")
        
        # Sometimes W&B sweeps save config values as single-item lists
        if isinstance(llm, list): llm = llm[0]
        if isinstance(dataset, list): dataset = dataset[0]
        
        # We don't care about misconfigured or testing runs
        if llm == "Unknown_LLM" or dataset == "Unknown_Dataset":
            continue
            
        combination_key = (llm, dataset)
        
        # W&B automatically tracks total run time (in seconds) in the summary as '_runtime'
        runtime_seconds = run.summary.get("_runtime", 0)
        
        # If this run is longer than the current max for this combination, overwrite it
        if runtime_seconds > max_runtime_runs[combination_key]["runtime"]:
            max_runtime_runs[combination_key] = {
                "run_id": run.id,
                "run_name": run.name,
                "runtime": runtime_seconds,
                "url": run.url
            }
            
        valid_runs_counted += 1
        
    print(f"\nSuccessfully parsed {valid_runs_counted} finished runs.")
    print("==================================================")
    print("MAXIMUM RUNNING TIMES BY COMBINATION")
    print("==================================================\n")
    
    # Sort and print the results nicely
    for (llm, dataset), data in sorted(max_runtime_runs.items()):
        runtime = data["runtime"]
        
        # Convert raw seconds into hours, minutes, and seconds for readability
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        
        # Clean up the LLM name for printing (e.g., 'mistralai/Mistral-7B-Instruct-v0.2' -> 'Mistral-7B-Instruct-v0.2')
        short_llm = llm.split('/')[-1] if '/' in llm else llm
        
        print(f"[{short_llm} | {dataset}]")
        if hours > 0:
            print(f"  Max Time: {hours}h {minutes}m {seconds}s")
        else:
            print(f"  Max Time: {minutes}m {seconds}s")
        print(f"  Run ID:   {data['run_id']}")
        print(f"  URL:      {data['url']}\n")

if __name__ == "__main__":
    get_max_runtime_runs()