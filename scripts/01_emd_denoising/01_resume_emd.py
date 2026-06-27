# resume_emd.py

import numpy as np
import pandas as pd
import time
import multiprocessing
import os
from PyEMD import EMD

def emd_denoise_single(window, discard_first_n_imfs=1):
    emd = EMD()
    window_f64 = window.astype(np.float64)
    imfs = emd.emd(window_f64)
    if imfs.shape[0] <= discard_first_n_imfs:
        return window_f64.astype(np.float32)
    denoised = np.sum(imfs[discard_first_n_imfs:], axis=0)
    return denoised.astype(np.float32)

def main():
    print("Loading dataset...")
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

    print(f"Windows shape: {windows.shape}")

    BATCH_SIZE = 10_000
    N_PROCESSES = multiprocessing.cpu_count()  # use all available cores locally
    CHECKPOINT_DIR = 'data/emd_checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    n_total_windows = len(windows)
    n_batches = (n_total_windows + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Total windows: {n_total_windows:,} | Batches: {n_batches} | Processes: {N_PROCESSES}")

    overall_start = time.time()

    for batch_idx in range(n_batches):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'emd_batch_{batch_idx:03d}.npy')

        if os.path.exists(checkpoint_path):
            print(f"Batch {batch_idx+1}/{n_batches}: already done, skipping.")
            continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, n_total_windows)
        batch_windows = windows[start_idx:end_idx]

        batch_start = time.time()
        with multiprocessing.Pool(processes=N_PROCESSES) as pool:
            batch_results = pool.map(emd_denoise_single, batch_windows)

        batch_results_array = np.array(batch_results, dtype=np.float32)
        np.save(checkpoint_path, batch_results_array)

        batch_elapsed = time.time() - batch_start
        print(f"Batch {batch_idx+1}/{n_batches} done in {batch_elapsed:.1f}s "
              f"({len(batch_windows)} windows) | saved to {checkpoint_path}")

    overall_elapsed = time.time() - overall_start
    print(f"\nAll batches complete. Total time: {overall_elapsed/60:.1f} minutes")

if __name__ == '__main__':
    # Required on Windows: multiprocessing needs this guard to avoid
    # infinite recursive process spawning when the script is imported
    # by each worker process
    main()