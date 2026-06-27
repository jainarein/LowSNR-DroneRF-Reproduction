# train_test_split_and_smote.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('data/scaled_features.csv')

feature_cols = ['mean', 'std', 'skewness', 'kurtosis',
                'energy_detection', 'spectral_peak_freq', 'spectral_peak_mag',
                'spectral_centroid', 'spectral_spread', 'total_spectral_energy']

X = df[feature_cols].values
y = df['label'].values

# --- Step 1: Stratified train/test split ---
# Paper uses 60:40 split (60% train, remaining 40% split evenly between test/validation)
# We'll replicate: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train shape: {X_train.shape}, class balance: {np.bincount(y_train) / len(y_train)}")
print(f"Validation shape: {X_val.shape}, class balance: {np.bincount(y_val) / len(y_val)}")
print(f"Test shape: {X_test.shape}, class balance: {np.bincount(y_test) / len(y_test)}")

# --- Step 2: SMOTE — applied ONLY to training data ---
print("\nApplying SMOTE to training data only...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {np.bincount(y_train)}")
print(f"After SMOTE: {np.bincount(y_train_smote)}")
print(f"Train shape after SMOTE: {X_train_smote.shape}")

# --- Save everything for the next stages ---
np.save('data/X_train_smote.npy', X_train_smote)
np.save('data/y_train_smote.npy', y_train_smote)
np.save('data/X_train_raw.npy', X_train)      # unbalanced version, for ANN/CNN if we want class weights instead
np.save('data/y_train_raw.npy', y_train)
np.save('data/X_val.npy', X_val)
np.save('data/y_val.npy', y_val)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)

print("\nAll splits saved to data/")