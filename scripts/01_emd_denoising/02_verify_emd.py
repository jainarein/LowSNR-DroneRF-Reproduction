# verify_emd.py

import numpy as np
import os

CHECKPOINT_DIR = 'data/emd_checkpoints'
n_batches = 14

all_batches = []
for i in range(n_batches):
    path = os.path.join(CHECKPOINT_DIR, f'emd_batch_{i:03d}.npy')
    batch = np.load(path)
    all_batches.append(batch)
    print(f"Batch {i}: shape {batch.shape}, dtype {batch.dtype}")

denoised_windows = np.concatenate(all_batches, axis=0)
print(f"\nFinal denoised_windows shape: {denoised_windows.shape}")

# Sanity checks
print(f"Any NaNs? {np.isnan(denoised_windows).any()}")
print(f"Any Infs? {np.isinf(denoised_windows).any()}")
print(f"Min: {denoised_windows.min()}, Max: {denoised_windows.max()}")

# Save the consolidated result so we never need to re-load 14 separate files again
np.save('data/denoised_windows_full.npy', denoised_windows)
print("\nSaved consolidated array to data/denoised_windows_full.npy")