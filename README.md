# BiST-Former  
Bi-view Spectral–Temporal Gated Transformer for Harmful Brain Activity Classification

This repository contains the official PyTorch implementation of **BiST-Former**, a dual-path Transformer architecture for automated classification of harmful brain activity patterns from ICU EEG recordings.

The model explicitly separates temporal persistence and spectral organization through independent Transformer encoders and adaptively fuses both representations using a learned gating mechanism.

---

## Overview

BiST-Former is designed for multi-class classification of six ACNS-defined harmful brain activity patterns:

- Seizure (SZ)
- Generalized Periodic Discharges (GPD)
- Lateralized Periodic Discharges (LPD)
- Generalized Rhythmic Delta Activity (GRDA)
- Lateralized Rhythmic Delta Activity (LRDA)
- Other

The model operates on STFT representations of longitudinal bipolar EEG montages and processes:

- Temporal tokens (time-major representation)
- Spectral tokens (frequency-major representation)

Both pathways are encoded independently and fused via adaptive gating before classification.

---

## Repository Structure   
BiST-Former/   
│    
├── BISTFormer.py  
├── config.py # Hyperparameters and configuration   
├── requirements.txt # Python dependencies   
└── README.md   


---

## Data

This implementation expects the HMS Harmful Brain Activity dataset structure:   
BASE_DIR/   
│   
├── train.csv   
└── train_eegs/   
├── <eeg_id>.parquet   
├── ...   


Update `BASE_DIR` in `config.py` before training.

---

## STFT Configuration

- Sampling frequency: 200 Hz  
- Window length: 50 samples (0.25 s)  
- Overlap: 25 samples  
- FFT length: 50  
- Frequency bins: 26  
- Segment duration: 50 seconds  

---

## Training Configuration

- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 16  
- Gradient accumulation: 8 steps  
- Transformer layers: 8  
- Attention heads: 8  
- Embedding dimension: 128  
- Dropout: 0.3  
- Early stopping patience: 15 epochs  

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```
### 2. Train the model
```python Bistformer.py```

The best-performing model (based on validation loss) will be saved as:
best_model.pth


## Hardware

Experiments were conducted on NVIDIA H100 GPUs.
Training supports single-GPU CUDA execution.

## Reproducibility
For deterministic behavior:
Set random seeds in config.py
Use fixed splits
Ensure identical CUDA and PyTorch versions

## Citation

If you use this code in your research, please cite:
BiST-Former: Bi-view Spectral–Temporal Gated Transformer for Harmful Brain Activity Classification
