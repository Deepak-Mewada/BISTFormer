import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import stft
from tqdm import tqdm

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SFREQ = 200
NUM_CLASSES = 6
BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
ACCUM_STEPS = 8
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class HMSDataset(Dataset):
    """
    Dataset for HMS Harmful Brain Activity classification.
    Returns STFT tensor of shape (S, C, F).
    """

    def __init__(self, dataframe, base_dir, label_map, train=False):
        self.df = dataframe
        self.base_dir = base_dir
        self.label_map = label_map
        self.train = train

    def __len__(self):
        return len(self.df)

    def _load_bipolar(self, parquet_path, offset_seconds):
        df = pd.read_parquet(parquet_path)

        C1 = ['Fp1','Fp2','F7','F8','T3','T4','T5','T6','Fp1','Fp2','F3','F4','C3','C4','P3','P4']
        C2 = ['F7','F8','T3','T4','T5','T6','O1','O2','F3','F4','C3','C4','P3','P4','O1','O2']

        arr1 = df[C1].to_numpy().T
        arr2 = df[C2].to_numpy().T
        arr1[np.isnan(arr1)] = 1e-4
        arr2[np.isnan(arr2)] = 1e-4

        bipolar = arr1 - arr2

        center = int(offset_seconds * SFREQ)
        start = center - (50 * SFREQ) // 2
        end = center + (50 * SFREQ) // 2

        return bipolar[:, start:end]

    def _apply_stft(self, signal):
        channels, _ = signal.shape
        features = []

        for ch in range(channels):
            _, _, Z = stft(signal[ch], fs=SFREQ, nperseg=50)
            features.append(np.abs(Z.T))

        return np.stack(features, axis=1)  # (S, C, F)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = row["eeg_id"]
        label = row["expert_consensus"]
        offset = row["eeg_label_offset_seconds"]

        path = os.path.join(self.base_dir, "train_eegs", f"{eeg_id}.parquet")

        signal = self._load_bipolar(path, offset)
        stft_tensor = self._apply_stft(signal)

        x = torch.tensor(stft_tensor, dtype=torch.float32)
        y = torch.tensor(self.label_map[label], dtype=torch.long)

        return x, y


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, layers=8, heads=8, dim_ff=512, dropout=0.3):
        super().__init__()
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pos(x)
        x = self.encoder(x)
        return self.norm(x)


class BiSTFormer(nn.Module):
    def __init__(self, S=401, C=16, F=26, d_model=128):
        super().__init__()

        self.temporal_proj = nn.Linear(C * F, d_model)
        self.spectral_proj = nn.Linear(S * C, d_model)

        self.temporal_encoder = TransformerEncoder(d_model)
        self.spectral_encoder = TransformerEncoder(d_model)

        self.gate = nn.Linear(2 * d_model, 2)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, NUM_CLASSES),
        )

    def forward(self, x):
        B, S, C, F = x.shape

        # Temporal pathway
        temp = self.temporal_proj(x.view(B, S, C * F))
        temp = self.temporal_encoder(temp).mean(dim=1)

        # Spectral pathway
        spec = self.spectral_proj(
            x.permute(0, 3, 1, 2).reshape(B, F, S * C)
        )
        spec = self.spectral_encoder(spec).mean(dim=1)

        # Gating
        combined = torch.cat([temp, spec], dim=1)
        weights = torch.softmax(self.gate(combined), dim=-1)
        temp = temp * weights[:, 0:1]
        spec = spec * weights[:, 1:2]

        fused = torch.cat([temp, spec], dim=1)

        return self.classifier(fused)


# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

def train(model, train_loader, val_loader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        running_loss = 0
        correct = 0
        total = 0

        for step, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss {train_loss:.4f}, "
            f"Train Acc {train_acc:.4f}, "
            f"Val Loss {val_loss:.4f}, "
            f"Val Acc {val_acc:.4f}"
        )


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total
