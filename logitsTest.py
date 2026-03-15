import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
import os
import re

# Helper to sort filenames numerically (1, 2, 10) instead of alphabetically (1, 10, 2)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

base_dir = "./preproc_data"
test_datasets = ["hotpotqa_test", "imdb_test", "movies_test"]

# Find all LLM directories
llm_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for llm_folder in llm_dirs:
    llm_path = os.path.join(base_dir, llm_folder)
    # Handle nested directories like meta-llama/Llama-3...
    subfolders = [f.path for f in os.scandir(llm_path) if f.is_dir()]
    if subfolders:
        llm_path = subfolders[0]

    for dataset in test_datasets:
        dataset_path = os.path.join(llm_path, dataset)
        if not os.path.exists(dataset_path): continue
            
        label_files = glob.glob(os.path.join(dataset_path, "label_*.pt"))
        if not label_files: continue
        label_files.sort(key=natural_sort_key)
            
        all_atp = []
        all_labels = []
        
        for label_file in label_files:
            match = re.search(r'label_(\d+)\.pt', os.path.basename(label_file))
            if not match: continue
            num = match.group(1)
            atp_path = os.path.join(dataset_path, f"ATP_output_{num}.pt")
            
            if os.path.exists(atp_path):
                # Load to CPU to keep VRAM free for training
                atp_chunk = torch.load(atp_path, map_location='cpu')
                label_chunk = torch.load(label_file, map_location='cpu')
                
                # Force to float64 to avoid float16 precision errors during log operations
                atp_np = atp_chunk.numpy().astype(np.float64) if torch.is_tensor(atp_chunk) else np.array(atp_chunk, dtype=np.float64)
                
                if torch.is_tensor(label_chunk):
                    label_np = label_chunk.numpy().flatten()
                else:
                    label_np = np.array([label_chunk]).flatten()
                
                all_atp.append(atp_np.flatten())
                all_labels.append(label_np)
                
        if not all_atp: continue
            
        final_labels = np.concatenate(all_labels)
        temp_atp = np.concatenate(all_atp)
        
        num_samples = len(final_labels)
        if num_samples > 0 and len(temp_atp) % num_samples == 0:
            seq_len = len(temp_atp) // num_samples
            final_atp = temp_atp.reshape(num_samples, seq_len)
        else:
            print(f"Skipping {dataset}: Shape mismatch.")
            continue
        
        # --- CALCULATE LOGITS (LOG-PROBS) ---
        # Clip values to the smallest representable float64 to avoid log(0) and -inf
        # This approximates the raw logit distribution used in the paper
        logits_atp = np.log(np.clip(final_atp, 1e-15, 1.0))
        
        # pooling operations as defined in Section 5.1
        z_mean = np.mean(logits_atp, axis=1)
        z_min = np.min(logits_atp, axis=1)
        z_max = np.max(logits_atp, axis=1)
        
        print(f"\n--- Model: {os.path.basename(llm_path)} | Dataset: {dataset} ---")
        
        # Ensure finite scores for ROC AUC
        if not np.all(np.isfinite(z_mean)):
            print("Warning: Non-finite values detected in logits. Using fallback cleanup.")
            z_mean = np.nan_to_num(z_mean, nan=-100.0, posinf=0.0, neginf=-100.0)

        print(f"Logits-mean: {roc_auc_score(final_labels, z_mean) * 100:.2f}")
        print(f"Logits-min:  {roc_auc_score(final_labels, z_min) * 100:.2f}")
        print(f"Logits-max:  {roc_auc_score(final_labels, z_max) * 100:.2f}")

print("\nFinished calculating Logits baselines!")