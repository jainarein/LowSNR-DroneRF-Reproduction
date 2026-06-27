# scale_features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/engineered_features.csv')

feature_cols = ['mean', 'std', 'skewness', 'kurtosis',
                'energy_detection', 'spectral_peak_freq', 'spectral_peak_mag',
                'spectral_centroid', 'spectral_spread', 'total_spectral_energy']

print("Before clipping:")
print(df[feature_cols].describe())

# --- Step 1: Winsorize (clip) extreme outliers per feature, at 1st/99th percentile ---
df_clipped = df.copy()
for col in feature_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
    df_clipped[col] = df[col].clip(lower, upper)
    print(f"{col}: clipped {n_clipped} values ({100*n_clipped/len(df):.2f}%) outside [{lower:.4g}, {upper:.4g}]")

# --- Step 2: Standardize the clipped features ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_clipped[feature_cols])

scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
scaled_df['label'] = df['label'].values

print("\nAfter clipping + standardization:")
print(scaled_df[feature_cols].describe())

scaled_df.to_csv('data/scaled_features.csv', index=False)
joblib.dump(scaler, 'data/feature_scaler.pkl')  # save scaler for later use on any new data

print("\nSaved scaled features to data/scaled_features.csv")
print("Saved scaler object to data/feature_scaler.pkl")