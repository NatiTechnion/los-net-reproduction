import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
from utils.constants import MAXIMAL_VOCAB_SIZE
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing

class CustomSavedDataset(Dataset):
    def __init__(self, preprocessed_dir, topk_preprocess=1000000, topk_dim=1000, input_output_flag='input', input_type = "LOS"):
        """
        Initialize the dataset.

        Args:
            base_dir (str): Base directory containing the saved data.
            LLM (str): Name of the language model directory.
            dataset_name (str): Name of the dataset directory.
        """
        self.preprocessed_dir = preprocessed_dir
        self.file_indices = self._get_indices()
        self.topk_dim = topk_dim
        self.input_type = input_type
    


        self.data = []

        for idx in tqdm(self.file_indices, desc="Loading data"):
            sorted_TDS_path = os.path.join(
                self.preprocessed_dir,
                f'TDS_topk_{input_output_flag}_{topk_preprocess}_{idx}.pt'
            )
            normalized_ATP_path = os.path.join(
                self.preprocessed_dir,
                f'ATP_{input_output_flag}_{idx}.pt'
            )
            ATP_R_path = os.path.join(
                self.preprocessed_dir,
                f'ATP_R_{input_output_flag}_{idx}.pt'
            )
            label_path = os.path.join(
                self.preprocessed_dir,
                f'label_{idx}.pt'
            )

            label = torch.load(label_path)

            if self.input_type == "LOS":
                sorted_TDS_normalized = torch.load(sorted_TDS_path)[:, :self.topk_dim]
                normalized_ATP = torch.load(normalized_ATP_path)
                ATP_R = torch.load(ATP_R_path)
                device = "cuda:0"
                #label_tensor = torch.tensor(label, device=device)
                self.data.append((
                    sorted_TDS_normalized.to(device),
                    normalized_ATP.to(device),
                    ATP_R.to(device),
                    label
                ))
            else:
                raise ValueError("Invalid input type.")
      
        
    def _get_indices(self):
        """Find all unique indices by looking for logits_canonized files."""

        files = os.listdir(self.preprocessed_dir)
        indices = sorted(
            set(int(f.split('_')[-1].split('.')[0])
                for f in files if f.startswith('') and 'label' in f)
        )
        return indices

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and return the sample at the given index.

        Args:
            idx (int): Index of the sample to load.

        """
     

        return self.data[idx]