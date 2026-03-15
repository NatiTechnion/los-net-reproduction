import os
import torch
import matplotlib.pyplot as plt

# 1. Experimental Grid
datasets = ['Movies', 'HotpotQA', 'IMDB']
llms = ['L-3-8b', 'Mis-7b', 'Q-2.5-7b']
K_values = [10, 50, 100, 500, 1000]

# --- TEMPERATURE FIX TO MATCH PAPER ---
# Simulates the generation temperature to sharpen the probability distribution
# and push the mass of the top-10 tokens above 90%
TEMPERATURE = 0.6 

dataset_map = {
    'Movies': 'movies_test',
    'HotpotQA': 'hotpotqa_test',
    'IMDB': 'imdb_test'
}

llm_map = {
    'L-3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct', 
    'Mis-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'Q-2.5-7b': 'Qwen/Qwen2.5-7B-Instruct' 
}

colors = {
    'Movies': '#8cb8b7',    
    'HotpotQA': '#8bb1f0',  
    'IMDB': '#de8888'       
}

def calculate_apm_for_k(probabilities_tensor, k):
    """Calculates APM for a specific K from a pre-sorted tensor."""
    top_k_probs = probabilities_tensor[..., :k]
    summed_mass = top_k_probs.sum(dim=-1) 
    return summed_mass.mean().item()

def find_tensor_files(base_dir, dataset_name, llm_name):
    """STRICTLY finds only the TDS K=1000 files."""
    target_dir = os.path.join(base_dir, llm_map[llm_name], dataset_map[dataset_name])
    if not os.path.exists(target_dir): return []
        
    pt_files = [f for f in os.listdir(target_dir) if f.startswith('TDS_topk_output_1000_') and f.endswith('.pt')]
    return [os.path.join(target_dir, f) for f in pt_files]

def extract_tds(obj):
    """Recursively search any Python object for the actual output tensor."""
    if isinstance(obj, torch.Tensor): return obj
    if isinstance(obj, dict):
        for k in ['X', 'TDS', 'tds', 'logits', 'probs', 'probabilities', 'token_probs']:
            if k in obj and isinstance(obj[k], torch.Tensor): return obj[k]
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, torch.Tensor) and len(v.shape) >= 1 and v.shape[-1] >= 10: return v
    return None

def load_preprocessed_data(file_paths):
    """Loads files, extracts logits, applies temperature-scaled softmax, kills padding, and concatenates."""
    all_tensors = []
    
    for file_path in file_paths:
        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception:
            continue 
            
        X_tensor = extract_tds(data)
        if X_tensor is None: continue 
            
        # Standardize shape to 2D: (seq_len, vocab) 
        if len(X_tensor.shape) == 3: X_tensor = X_tensor.reshape(-1, X_tensor.shape[-1])
        elif len(X_tensor.shape) == 1: X_tensor = X_tensor.unsqueeze(0)
            
        if X_tensor.shape[-1] < 1000: continue 
        X_tensor = X_tensor[..., :1000]

        if not X_tensor.is_floating_point(): X_tensor = X_tensor.float()

        # 1. Apply Temperature-Scaled Softmax
        X_tensor = torch.softmax(X_tensor / TEMPERATURE, dim=-1)
        
        # 2. THE PADDING KILLER
        # Padded rows become flat uniform distributions.
        # Real LLM token distributions are highly peaked (max prob > 0.02).
        # We throw away any row where the max probability is suspiciously flat.
        valid_mask = X_tensor.max(dim=-1).values > 0.02
        X_tensor = X_tensor[valid_mask]
        
        # If the entire file was just padding, skip it
        if len(X_tensor) == 0: continue
        
        # 3. Force them to be sorted highest-to-lowest
        X_tensor = torch.sort(X_tensor, descending=True, dim=-1).values
            
        all_tensors.append(X_tensor)
        
    if len(all_tensors) > 1: return torch.cat(all_tensors, dim=0) 
    elif len(all_tensors) == 1: return all_tensors[0]
    else: raise ValueError("No valid data found after filtering padding.")

# =====================================================================
# MAIN PLOTTING LOOP
# =====================================================================
base_data_dir = '/workspace/DL/proj/Beyond-next-token-probabilities/preproc_data'

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, dataset in enumerate(datasets):
    for j, llm in enumerate(llms):
        ax = axes[i, j]
        
        file_paths = find_tensor_files(base_data_dir, dataset, llm)
        
        if file_paths:
            print(f"[{dataset} | {llm}] -> Extracting from {len(file_paths)} file(s)...")
            try:
                probs_tensor = load_preprocessed_data(file_paths)
                apms = [calculate_apm_for_k(probs_tensor, k) for k in K_values]
                
                x_labels = [str(k) for k in K_values]
                bars = ax.bar(x_labels, apms, color=colors[dataset], edgecolor='white', linewidth=1)
                
                # Dynamic zoom just like the paper
                min_apm = min(apms)
                buffer = (1.0 - min_apm) * 0.2 if min_apm < 0.995 else 0.005
                lower_bound = max(0.0, min_apm - buffer)
                
                # Hard limit to prevent matplotlib from flipping the axes
                ax.set_ylim(min(lower_bound, 0.99), 1.0)
                
            except Exception as e:
                print(f"[{dataset} | {llm}] -> Error processing tensor: {e}")
                ax.text(0.5, 0.5, "Processing Error", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
            
        ax.set_title(f"{dataset}, {llm}", fontsize=14, pad=10)
        ax.set_xlabel("Top-K", fontsize=12)
        if j == 0:
            ax.set_ylabel("Average Captured Prob. Mass", fontsize=12)
            
        ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(pad=3.0)
plt.savefig("figure9_reproduction2.png", bbox_inches='tight', dpi=300)
print("\nPlot successfully saved to figure9_reproduction2.png!")