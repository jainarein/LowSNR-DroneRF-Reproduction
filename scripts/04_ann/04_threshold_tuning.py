# threshold_tuning.py

import numpy as np
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, roc_auc_score)

model = keras.models.load_model('data/ann_model_extended.keras')

X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# --- Get predicted probabilities on VALIDATION set (never touch test set for tuning) ---
val_proba = model.predict(X_val).flatten()

# --- Sweep thresholds, find the one that maximizes F1 on validation data ---
thresholds = np.arange(0.05, 0.95, 0.01)
best_threshold = 0.5
best_f1 = 0

print("Threshold sweep on VALIDATION set:")
for t in thresholds:
    preds = (val_proba > t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest threshold found: {best_threshold:.2f} (validation F1: {best_f1:.4f})")

# --- Apply this threshold to the TEST set, exactly once, as the final evaluation ---
test_proba = np.load('data/ann_test_proba_extended.npy')
y_pred_tuned = (test_proba > best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_tuned)
precision = precision_score(y_test, y_pred_tuned)
recall = recall_score(y_test, y_pred_tuned)
f1 = f1_score(y_test, y_pred_tuned)
auc = roc_auc_score(y_test, test_proba)  # AUC is threshold-independent, unchanged

print(f"\n--- Test Set Performance (tuned threshold = {best_threshold:.2f}) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

cm = confusion_matrix(y_test, y_pred_tuned)
print(f"\nConfusion Matrix:\n{cm}")