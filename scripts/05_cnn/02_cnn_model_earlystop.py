# cnn_model_earlystop.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, roc_auc_score)
import time
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

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

windows_mean = windows.mean(axis=1, keepdims=True)
windows_std = windows.std(axis=1, keepdims=True) + 1e-8
windows_normalized = (windows - windows_mean) / windows_std

X_temp, X_test, y_temp, y_test = train_test_split(
    windows_normalized, window_labels, test_size=0.20, stratify=window_labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

class DroneRFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 32
train_loader = DataLoader(DroneRFDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DroneRFDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(DroneRFDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class FTLW_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
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
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Manual early stopping implementation ---
PATIENCE = 7
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

MAX_EPOCHS = 50
train_losses, val_losses = [], []

print(f"Training with early stopping (patience={PATIENCE})...\n")
overall_start = time.time()

for epoch in range(MAX_EPOCHS):
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
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_val_loss += loss.item() * X_batch.size(0)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # --- Early stopping check ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())  # checkpoint best weights
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1} "
                  f"(no improvement for {PATIENCE} epochs). Best val_loss: {best_val_loss:.4f}")
            break

total_time = time.time() - overall_start
print(f"\nTotal training time: {total_time/60:.2f} minutes")

# --- Restore best weights (not the final epoch's weights) ---
model.load_state_dict(best_model_state)
print("Restored best model weights.")

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

print(f"\n--- Test Set Performance (early-stopped, best weights) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print(f"\nConfusion Matrix:\n{cm}")

torch.save(model.state_dict(), 'data/cnn_model_earlystop.pt')
np.save('data/cnn_train_losses_es.npy', np.array(train_losses))
np.save('data/cnn_val_losses_es.npy', np.array(val_losses))
print("\nModel saved.")