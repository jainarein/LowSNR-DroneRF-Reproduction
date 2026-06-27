# ann_model.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, roc_auc_score)
import time

# Load data — same splits as RF, for fair comparison
X_train = np.load('data/X_train_smote.npy')
y_train = np.load('data/y_train_smote.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")

# --- Architecture exactly matching paper's FTLW-ANN description ---
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# --- Optimizer matching Table 2: learning rate 1e-4 ---
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- Train: 30 epochs, batch size 32, matching Table 2 ---
print("\nTraining ANN...")
start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)
train_time = time.time() - start
print(f"\nTraining completed in {train_time:.2f} seconds")

# --- Evaluate on test set ---
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

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

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

model.save('data/ann_model.keras')
print("\nModel saved to data/ann_model.keras")