"""
Configuration file for BiST-Former training and evaluation.

All experiment parameters, data paths, and model hyperparameters
are centralized here for reproducibility and clarity.
"""

import os
import torch
import random
import numpy as np

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = "/path/to/hms-harmful-brain-activity-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
EEG_DIR = os.path.join(BASE_DIR, "train_eegs")

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Data Parameters
# ------------------------------------------------------------

SFREQ = 200                # Sampling frequency (Hz)
SEGMENT_DURATION = 50     # Segment length (seconds)

STFT_WINDOW = 50          # nperseg
STFT_OVERLAP = 25
STFT_NFFT = 50

NUM_CHANNELS = 16
NUM_CLASSES = 6


# ------------------------------------------------------------
# Model Parameters
# ------------------------------------------------------------

D_MODEL = 128
NUM_LAYERS = 8
NUM_HEADS = 8
DIM_FEEDFORWARD = 512
DROPOUT = 0.3


# ------------------------------------------------------------
# Training Parameters
# ------------------------------------------------------------

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 150
ACCUMULATION_STEPS = 8
PATIENCE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Data Splitting
# ------------------------------------------------------------

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_STATE = 42
