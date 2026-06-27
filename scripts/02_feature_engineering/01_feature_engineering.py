# feature_engineering.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal.windows import blackmanharris
import time

FS = 10_000_000  # sample rate, Hz (Table 1)

def compute_time_domain_features(denoised_window):
    """
    Statistical features from the EMD-denoised window.
    """
    mean_val = np.mean(denoised_window)
    std_val = np.std(denoised_window)
    skew_val = skew(denoised_window)
    kurt_val = kurtosis(denoised_window)
    return mean_val, std_val, skew_val, kurt_val

def compute_frequency_domain_features(raw_window, fs=FS):
    """
    Spectral features from FFT/PSD of the ORIGINAL raw (non-denoised) window.
    """
    N = len(raw_window)

    # Same preprocessing as Stage 3/4: remove DC, apply Blackman-Harris
    centered = raw_window - np.mean(raw_window)
    bh = blackmanharris(N)
    windowed = centered * bh

    fft_result = np.fft.fft(windowed)
    magnitude = np.abs(fft_result)[:N // 2]
    freqs = np.fft.fftfreq(N, d=1/fs)[:N // 2]

    # PSD (equation 9, periodogram method)
    psd = (np.abs(fft_result) ** 2) / (N * fs)
    psd = psd[:N // 2]

    # --- Energy detection feature (equation 3) ---
    energy_detection = np.mean(np.abs(raw_window) ** 2)

    # --- Spectral peak ---
    peak_idx = np.argmax(magnitude)
    spectral_peak_freq = freqs[peak_idx]
    spectral_peak_mag = magnitude[peak_idx]

    # --- Spectral centroid: weighted average frequency, weighted by magnitude ---
    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-12)

    # --- Spectral spread: "standard deviation" of the spectrum around centroid ---
    spectral_spread = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-12)
    )

    # --- Total spectral energy (sum of PSD across all frequencies) ---
    total_spectral_energy = np.sum(psd)

    return (energy_detection, spectral_peak_freq, spectral_peak_mag,
            spectral_centroid, spectral_spread, total_spectral_energy)


def extract_features_for_all_windows(raw_windows, denoised_windows):
    """
    Builds the final feature matrix: one row per window, columns are
    the engineered features described above.
    """
    n_windows = len(raw_windows)
    feature_names = [
        'mean', 'std', 'skewness', 'kurtosis',                      # time-domain (denoised)
        'energy_detection', 'spectral_peak_freq', 'spectral_peak_mag',  # freq-domain (raw)
        'spectral_centroid', 'spectral_spread', 'total_spectral_energy'
    ]

    features = np.zeros((n_windows, len(feature_names)), dtype=np.float64)

    start = time.time()
    for i in range(n_windows):
        mean_val, std_val, skew_val, kurt_val = compute_time_domain_features(denoised_windows[i])
        (energy_det, peak_freq, peak_mag, centroid,
         spread, total_energy) = compute_frequency_domain_features(raw_windows[i])

        features[i] = [mean_val, std_val, skew_val, kurt_val,
                        energy_det, peak_freq, peak_mag,
                        centroid, spread, total_energy]

        if (i + 1) % 20000 == 0:
            elapsed = time.time() - start
            print(f"Processed {i+1}/{n_windows} windows | {elapsed:.1f}s elapsed")

    elapsed = time.time() - start
    print(f"\nFeature extraction complete: {n_windows} windows in {elapsed:.1f}s")

    feature_df = pd.DataFrame(features, columns=feature_names)
    return feature_df


if __name__ == '__main__':
    print("Loading raw windows and denoised windows...")

    # Rebuild raw windows (same as before)
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
    raw_windows = iq_values[:trimmed_len].reshape(n_complete_windows, WINDOW_SIZE)
    window_labels = iq_labels[:trimmed_len].reshape(n_complete_windows, WINDOW_SIZE)[:, 0]

    # Load denoised windows (already consolidated)
    denoised_windows = np.load('data/denoised_windows_full.npy')

    print(f"Raw windows shape: {raw_windows.shape}")
    print(f"Denoised windows shape: {denoised_windows.shape}")
    assert raw_windows.shape == denoised_windows.shape, "Shape mismatch between raw and denoised!"

    feature_df = extract_features_for_all_windows(raw_windows, denoised_windows)
    feature_df['label'] = window_labels

    feature_df.to_csv('data/engineered_features.csv', index=False)
    print("\nSaved to data/engineered_features.csv")
    print(feature_df.head())
    print(feature_df.describe())