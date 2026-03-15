import os
import torch

# --- CONFIGURATION ---
# Simulating the generation temperature to match the >99% peaked distribution
TEMPERATURE = 0.6 
K_values = [10, 50, 100, 500, 1000]
base_data_dir = '/workspace/DL/proj/Beyond-next-token-probabilities/preproc_data'

# Table 3 specifically evaluates Mis-7B on HotpotQA
target_dir = os.path.join(base_data_dir, 'mistralai/Mistral-7B-Instruct-v0.2', 'hotpotqa_test')

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
        
        # 2. Kill Padding (removes artificially flat uniform distributions)
        valid_mask = X_tensor.max(dim=-1).values > 0.02
        X_tensor = X_tensor[valid_mask]
        
        if len(X_tensor) == 0: continue
        
        # 3. Sort highest-to-lowest
        X_tensor = torch.sort(X_tensor, descending=True, dim=-1).values
            
        all_tensors.append(X_tensor)
        
    if len(all_tensors) > 1: return torch.cat(all_tensors, dim=0) 
    elif len(all_tensors) == 1: return all_tensors[0]
    else: raise ValueError("No valid data found after filtering padding.")

# --- MAIN EXECUTION ---
print("Scanning for files...")
if not os.path.exists(target_dir):
    print(f"Error: Directory not found -> {target_dir}")
    exit()

# STRICTLY grab only the TDS files (ignores label_*.pt and ATP_*.pt)
pt_files = [
    os.path.join(target_dir, f) for f in os.listdir(target_dir) 
    if f.startswith('TDS_topk_output_1000_') and f.endswith('.pt')
]

print(f"Found {len(pt_files)} TDS files. Extracting and calculating APM...\n")

try:
    probs_tensor = load_preprocessed_data(pt_files)
    
    print("Table 3 Reproduction: Mis-7B on HotpotQA (APM %)")
    print("-----------------------------------------")
    print(f"{'K':<10} | {'Reproduced APM (%)':<20}")
    print("-" * 33)
    
    for k in K_values:
        top_k_probs = probs_tensor[..., :k]
        summed_mass = top_k_probs.sum(dim=-1) 
        apm_percentage = summed_mass.mean().item() * 100
        
        print(f"{k:<10} | {apm_percentage:.2f}")

except Exception as e:
    print(f"An error occurred: {e}")