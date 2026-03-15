# LOS-Net Reproduction and Improvements

This repository contains our reproduction and extension of the paper:

**Beyond Next Token Probabilities: Learnable, Fast Detection of Hallucinations and Data Contamination on LLM Output Distributions**  
(Guy Bar-Shalom et al., AAAI 2026)

The goal of this project was to reproduce the LOS-Net framework for hallucination detection and explore improvements in both **computational efficiency** and **predictive performance**.

---

# Project Overview

Large Language Models (LLMs) can produce hallucinated information - outputs that sound plausible but are factually incorrect. The LOS-Net framework proposes a **gray-box method** that analyzes **LLM Output Signatures (LOS)** to detect hallucinations.

In this project we:

- Reproduced the LOS-Net architecture and experimental setup
- Verified that the reported results can be replicated across several **LLM × dataset** combinations
- Introduced several **optimizations** to accelerate training and inference
- Performed large-scale hyperparameter exploration enabled by these optimizations to get better performance

---

# Installation and Running

1. Clone the following repository:

```git clone https://github.com/NatiTechnion/los-net-reproduction```

2. Create the conda environment and activate it:
```bash
conda env create -f los_net_repro.yml
conda activate LOS_Net
```

3. Generate Raw Datasets for Hallucination Detection using the given script ```./scripts/generate_HD_raw_datasets.sh```:

First, make the script executable and then run the script:
```bash
chmod +x ./scripts/generate_HD_raw_datasets.sh
./scripts/generate_HD_raw_datasets.sh [BASE_RAW_DATA_DIR] [NUM_PARALLEL_JOBS]
```

Example run:

```./scripts/generate_HD_raw_datasets.sh /workspace/DL/proj/Beyond-next-token-probabilities/raw_data 5```

4. Preprocess the data using the given script ```./scripts/preprocess_raw_datasets_LOS.sh```:

First, make the script executable and then run the script:

``` bash
chmod +x ./scripts/preprocess_raw_datasets_LOS.sh
bash ./scripts/preprocess_raw_datasets_LOS.sh [BASE_RAW_DATA_DIR] [BASE_PRE_PROCESSED_DATA_DIR] [NUM_PARALLEL_JOBS]
```

Example run:

```bash ./scripts/preprocess_raw_datasets_LOS.sh /workspace/DL/proj/Beyond-next-token-probabilities/raw_data /workspace/DL/proj/Beyond-next-token-probabilities/preproc_data 2```

5. Update your GPU index in the constructor of the ```CustomSavedDataset``` class in ```utils/dataset_preprocess.py```.

For example, if you use an NVIDIA GPU at index 1, change row 305 of that file to ```device = "cuda:1"```. Default value is ```cuda:0```.

6. Run the sweep you would like to run with its corresponding sweep file:

```wandb sweep --project [WANDB_PROJECT_NAME] [SWEEP_FILE]```

For example, the following command executes the main runs of the original paper for ```meta/movies```:

```wandb sweep --project LOS-Net ./sweeps/LOS/HD/meta_movies_output.yaml```

Note: Don't forget to login to your wandb account with ```wandb login``` if you want to save the result to your account.


Note: We don't recommend running anything on the cpu, it would take way too long.


If you want to run one of the scripts provided in this project (each explained in the report), simply run:

```bash
python [SCRIPT_NAME]
```

Note that these scripts are mainly used to automatically compute the results from the sweep runs. Therefore, you must first run the sweeps themselves before using these scripts on the reproduced results.

Alternatively, you may use the projects we generated instead of reproducing the data yourself.
All projects are linked in the report and are publicly available for inspection and reuse.