# diagnose_ann.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Reload data
X_train = np.load('data/X_train_smote.npy')
y_train = np.load('data/y_train_smote.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Rebuild the exact same model and retrain, but this time CAPTURE the history object properly
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

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

# --- Plot loss and accuracy curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Accuracy over epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/ann_training_curves.png')
plt.show()

print("Saved plot to data/ann_training_curves.png")

# --- Print final-epoch values explicitly ---
print(f"\nFinal train loss: {history.history['loss'][-1]:.4f}")
print(f"Final val loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

# --- Look at the actual distribution of predicted probabilities on test set ---
y_pred_proba = model.predict(X_test).flatten()
print(f"\nPredicted probability stats on TEST set:")
print(f"Min: {y_pred_proba.min():.4f}, Max: {y_pred_proba.max():.4f}")
print(f"Mean: {y_pred_proba.mean():.4f}, Median: {np.median(y_pred_proba):.4f}")

model.save('data/ann_model_v2.keras')
np.save('data/ann_test_proba.npy', y_pred_proba)  # save for threshold tuning next