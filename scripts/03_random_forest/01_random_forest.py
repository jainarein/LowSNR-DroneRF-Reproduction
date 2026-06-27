# random_forest_model.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import joblib
import time

# Load the SMOTE-balanced training set, and the untouched val/test sets
X_train = np.load('data/X_train_smote.npy')
y_train = np.load('data/y_train_smote.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")

# --- Model matching paper's Table 2 parameters for FTLW-RF ---
rf_model = RandomForestClassifier(
    n_estimators=10,        # "10 decision trees" per paper
    criterion='entropy',    # explicitly stated in paper text
    max_depth=None,         # "no restrictions on maximum depth"
    random_state=42         # paper: "random state is being fixed at 42"
)

print("\nTraining Random Forest...")
start = time.time()
rf_model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Training completed in {train_time:.2f} seconds")

# --- Evaluate on TEST set (the untouched, realistically-imbalanced set) ---
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # probability of class 1 (ARMED)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- Test Set Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Measure inference time per sample (paper reports this in Table 6)
start = time.time()
_ = rf_model.predict(X_test[:1000])
inference_time = (time.time() - start) / 1000 * 1000  # ms per sample
print(f"\nAverage inference time per sample: {inference_time:.3f} ms")

# Save model
joblib.dump(rf_model, 'data/rf_model.pkl')
print("\nModel saved to data/rf_model.pkl")