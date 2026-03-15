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

# Identify LLM directories
llm_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for llm_folder in llm_dirs:
    llm_path = os.path.join(base_dir, llm_folder)
    
    # Handle nested directories (e.g., meta-llama/Llama-3-8B-Instruct)
    subfolders = [f.path for f in os.scandir(llm_path) if f.is_dir()]
    if subfolders:
        llm_path = subfolders[0]

    for dataset in test_datasets:
        dataset_path = os.path.join(llm_path, dataset)
        if not os.path.exists(dataset_path): 
            continue
            
        # Get all label files and sort them NUMERICALLY
        label_files = glob.glob(os.path.join(dataset_path, "label_*.pt"))
        if not label_files: 
            continue
        label_files.sort(key=natural_sort_key)
            
        all_atp = []
        all_labels = []
        
        for label_file in label_files:
            # Extract the chunk number from the label filename
            # e.g., 'label_42.pt' -> '42'
            match = re.search(r'label_(\d+)\.pt', os.path.basename(label_file))
            if not match: continue
            num = match.group(1)
            
            atp_path = os.path.join(dataset_path, f"ATP_output_{num}.pt")
            
            if os.path.exists(atp_path):
                # Load with map_location to avoid VRAM overhead during baseline check
                atp_chunk = torch.load(atp_path, map_location='cpu')
                label_chunk = torch.load(label_file, map_location='cpu')
                
                # Convert to numpy and handle potential scalar/int labels
                atp_np = atp_chunk.numpy() if torch.is_tensor(atp_chunk) else np.array(atp_chunk)
                
                if torch.is_tensor(label_chunk):
                    label_np = label_chunk.numpy().flatten()
                else:
                    label_np = np.array([label_chunk]).flatten()
                
                all_atp.append(atp_np.flatten())
                all_labels.append(label_np)
                
        if not all_atp: 
            continue
            
        final_labels = np.concatenate(all_labels)
        temp_atp = np.concatenate(all_atp)
        
        # Reshape: Verify total floats / number of samples = sequence length
        num_samples = len(final_labels)
        if num_samples > 0 and len(temp_atp) % num_samples == 0:
            seq_len = len(temp_atp) // num_samples
            final_atp = temp_atp.reshape(num_samples, seq_len)
        else:
            print(f"Error: Alignment failed for {dataset}. Labels: {num_samples}, ATP total: {len(temp_atp)}")
            continue
        
        # Baselines math
        p_mean = np.mean(final_atp, axis=1)
        p_min = np.min(final_atp, axis=1)
        p_max = np.max(final_atp, axis=1)
        
        print(f"\n--- Model: {os.path.basename(llm_path)} | Dataset: {dataset} ---")
        print(f"Verified Alignment: {num_samples} samples x {seq_len} tokens")

        # Score reporting
        try:
            print(f"Probas-mean: {roc_auc_score(final_labels, p_mean) * 100:.2f}")
            print(f"Probas-min:  {roc_auc_score(final_labels, p_min) * 100:.2f}")
            print(f"Probas-max:  {roc_auc_score(final_labels, p_max) * 100:.2f}")
        except ValueError as e:
            print(f"AUC Error (likely only one class present in labels): {e}")

print("\nAll baselines processed.")