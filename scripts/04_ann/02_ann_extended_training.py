# ann_extended_training.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import time

X_train = np.load('data/X_train_smote.npy')
y_train = np.load('data/y_train_smote.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- EarlyStopping: stop if val_loss doesn't improve for 15 consecutive epochs,
# restore the weights from whichever epoch had the BEST val_loss (not the last epoch) ---
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("Training with extended epochs (up to 300) + early stopping...")
start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,        # generous ceiling; early stopping will likely cut this short
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
train_time = time.time() - start
print(f"\nTraining stopped after {len(history.history['loss'])} actual epochs, "
      f"took {train_time:.2f} seconds")

# --- Plot curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Loss over epochs (extended training)')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Accuracy over epochs (extended training)')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/ann_extended_training_curves.png')
plt.show()

# --- Evaluate on test set ---
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- Test Set Performance (extended training) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

print(f"\nPredicted probability stats: min={y_pred_proba.min():.4f}, "
      f"max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}, "
      f"median={np.median(y_pred_proba):.4f}")

model.save('data/ann_model_extended.keras')
np.save('data/ann_test_proba_extended.npy', y_pred_proba)