# cnn_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, roc_auc_score)
import time

# --- Device check, printed prominently, first thing the script does ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{'='*50}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU — training will be significantly slower.")
print(f"{'='*50}\n")

# --- Load raw windows (CNN gets the raw 1024-sample sequence, not engineered features) ---
import pandas as pd

df = pd.read_csv('data/merged_file.csv')
df['label'] = df['label'].str.upper()
df['label_encoded'] = (df['label'] == 'ARMED').astype(np.int8)
df['IQSAMPLES'] = df['IQSAMPLES'].astype('float32')
df = df.drop(columns=['label'])

WINDOW_SIZE = 1024
iq_values = df['IQSAMPLES'].values
iq_labels = df['label_encoded'].values
n_complete_windows = len(iq_values) // WINDOW_SIZE
trimmed_len = n_complete_windows * WINDOW_SIZE
windows = iq_values[:trimmed_len].reshape(n_complete_windows, WINDOW_SIZE)
window_labels = iq_labels[:trimmed_len].reshape(n_complete_windows, WINDOW_SIZE)[:, 0]

print(f"Total windows: {windows.shape}")

# --- Normalize windows (CNN needs scaled input too, same principle as ANN) ---
# Per-window z-score normalization (each window normalized independently,
# since absolute power level varies hugely window to window, as we saw earlier)
windows_mean = windows.mean(axis=1, keepdims=True)
windows_std = windows.std(axis=1, keepdims=True) + 1e-8  # avoid divide-by-zero
windows_normalized = (windows - windows_mean) / windows_std

# --- Train/val/test split — SAME stratified approach as RF/ANN ---
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    windows_normalized, window_labels, test_size=0.20, stratify=window_labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42  # 0.25 of 80% = 20% overall
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# --- PyTorch Dataset wrapper ---
class DroneRFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, 1024) — 1 channel
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = DroneRFDataset(X_train, y_train)
val_dataset = DroneRFDataset(X_val, y_val)
test_dataset = DroneRFDataset(X_test, y_test)

# Small batch size given 4GB VRAM constraint — we'll verify this fits before scaling up
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- FTLW-CNN architecture: lightweight, conv+pool blocks, designed to target ~288K params (Table 4) ---
class FTLW_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # After 3 pooling layers (each halves length): 1024 -> 512 -> 256 -> 128
        # 64 channels * 128 length = 8192 flattened features
        self.fc1 = nn.Linear(64 * 128, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

model = FTLW_CNN().to(device)

# --- Parameter count check, against paper's stated 288K (Table 4) ---
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params:,} (paper reports 288K for FTLW-CNN)")

# --- Loss and optimizer (Table 2: lr=1e-3 for FTLW-CNN) ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
EPOCHS = 50  # paper's stated epoch count for FTLW-CNN
print(f"\nTraining for {EPOCHS} epochs, batch size {BATCH_SIZE}...")

train_losses, val_losses = [], []

overall_start = time.time()
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * X_batch.size(0)

    epoch_train_loss /= len(train_dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_val_loss += loss.item() * X_batch.size(0)

    epoch_val_loss /= len(val_dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

total_time = time.time() - overall_start
print(f"\nTotal training time: {total_time/60:.2f} minutes")

# --- Evaluate on test set ---
model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).cpu().numpy().flatten()
        all_probs.extend(outputs)
        all_preds.extend((outputs > 0.5).astype(int))
        all_labels.extend(y_batch.numpy().flatten())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"\n--- Test Set Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print(f"\nConfusion Matrix:\n{cm}")

torch.save(model.state_dict(), 'data/cnn_model.pt')
np.save('data/cnn_train_losses.npy', np.array(train_losses))
np.save('data/cnn_val_losses.npy', np.array(val_losses))
print("\nModel and loss history saved.")