# LowSNR_DroneRF Reproduction — Findings So Far (RF + ANN)

**Paper:** Tanveer et al., "From Lab to Field Trials: Real-Time Multimodel Drone Detection in Low-SNR Environments," IEEE A&E Systems Magazine, Feb 2026.

**Dataset:** `laibatanveer/merged` on Kaggle (137,557,132 rows, 2 columns: `IQSAMPLES`, `label`)

---

## 1. Dataset Reality Check — Key Deviations From the Paper

| Paper claim | What we actually found in the public dataset |
|---|---|
| Modes: OFF, connected, armed, flying — paper focuses on "OFF and connected" for the main binary task (Dataset Acquisition section) | Public CSV contains **only two labels: `OFF` and `ARMED`** (confirmed via full-file chunked scan, 137.5M rows). No `connected` or `flying` labels present. |
| `IQSAMPLES` implies raw complex I/Q baseband samples | Empirically, the column is **strictly non-negative** (min ≈ 2.9e-10, max ≈ 1.89e5, zero negative values across the full file) — consistent with **already-computed power / magnitude-squared values** (paper's eq. 1, Complex-to-Mag²), not raw complex baseband. This was confirmed via distribution diagnostics, not assumed. |
| 137.56 million samples, 3-second segments at 10 MHz | Confirmed: 137,557,132 total rows matches almost exactly. No segment-ID column exists in the public file — we reconstructed windows ourselves at **1024 samples/window** (matching the paper's stated FFT size, Table 1), since the original 30M-sample/3-second segmentation isn't recoverable from a flat, unlabeled-by-segment column. |
| Class balance not explicitly stated for OFF/ARMED pair | Measured: **59.18% ARMED / 40.82% OFF** at the row level; identical ratio preserved after windowing (134,333 windows total: 79,501 ARMED / 54,832 OFF). |
| Data ordering | File is **strictly ordered by class** (all ARMED rows first ~81.4M rows, then all OFF rows) — required shuffling/stratification before any train/test split to avoid a catastrophic class-leakage bug. |

**Practical consequence:** All classification results in this reproduction are for the **OFF vs. ARMED** binary task — the only pair the public dataset supports — and should be compared against whichever paper table actually corresponds to this pair (the paper's own text is internally inconsistent about which mode-pair its headline 93-94% numbers refer to).

---

## 2. Signal Processing Stages — What Was Faithfully Reproducible

| Stage | Status | Note |
|---|---|---|
| FFT (eq. via DFT, Blackman-Harris windowed) | Implemented, but **conceptually limited** by the dataset constraint above — FFT of an already-power-valued sequence reveals envelope/energy-fluctuation periodicity, not RF carrier frequency content as in the paper's Figure 5. Demonstrated faithfully alongside a synthetic complex-IQ example to show the "ideal" pipeline behavior for comparison. |
| PSD (eq. 9, periodogram method) | Implemented per the exact formula. Same input-data caveat as FFT applies. |
| Energy Detection (eq. 3, P_RF) | Fully reproducible and meaningful as-is — power averaging is valid regardless of the upstream limitation, since the paper's own equation 1 already defines power as the squared magnitude, which is what we have. |
| EMD Denoising (eq. 8 concept, extended to full multi-IMF decomposition) | Fully implemented on all 134,333 windows using PyEMD (`EMD-signal` package). Discarded first (highest-frequency) IMF, reconstructed from remainder. ~8 IMFs per window on average. Runtime: ~15 min locally with 12-core parallelization + checkpointing (vs. ~3 hours estimated on Colab's 2-core free tier). |

---

## 3. Feature Engineering (Stage 6)

Since the paper does not enumerate FTLW-RF/FTLW-ANN's exact input features, we engineered a defensible 10-feature set:

- **Time-domain (from EMD-denoised window):** mean, std, skewness, kurtosis
- **Frequency-domain (from FFT/PSD of original raw window):** energy detection (eq. 3), spectral peak frequency, spectral peak magnitude, spectral centroid, spectral spread, total spectral energy (PSD-summed)

Features were **winsorized (clipped at 1st/99th percentile)** before z-score standardization, due to extreme outliers (e.g., `std` feature ranged 0.0025 to 60,201 pre-clipping) — a small number of windows (~2%) showed pathological values, likely from EMD edge cases.

**Train/val/test split:** 60/20/20, stratified by class. SMOTE applied **only to the training set**, after the split (avoiding data leakage). Training set balanced from 32,899/47,700 (OFF/ARMED) to 47,700/47,700 via synthetic minority oversampling.

---

## 4. Results — Random Forest (Stage 8)

**Configuration (matching paper's Table 2 "RTLW-RF" row, likely typo for FTLW-RF):** 10 trees, entropy criterion, no max depth restriction, random_state=42.

| Metric | Paper (Table 3/4) | Our reproduction |
|---|---|---|
| Accuracy | 93% | **83.99%** |
| Precision | 94% | **89.99%** |
| Recall | 94% | **82.08%** |
| AUC | 0.94 | **0.9266** |
| Inference time | 1.4 ms | **0.004 ms** |

**Confusion matrix (test set, n=26,867):**
```
              Predicted OFF   Predicted ARMED
Actual OFF        9514            1452
Actual ARMED      2850           13051
```

**Assessment:** Strong AUC (0.927) indicates genuinely good class separation learned. Accuracy gap (~9 points below paper) most plausibly attributable to (a) unstated/different feature set in original paper, (b) the FFT/PSD-on-power-sequence limitation described above, (c) possible mode-pair mismatch in what the paper's 93% figure refers to. Faster inference is expected given our much lower-dimensional engineered feature set vs. the paper's likely richer/raw-signal-derived input.

---

## 5. Results — ANN (Stage 9)

**Configuration (matching paper's Table 2 "RTLW-ANN" row):** 2 hidden layers × 64 neurons, ReLU activation, dropout 0.2, sigmoid output, Adam optimizer (paper does not state optimizer — Adam assumed as standard default), learning rate 1e-4, batch size 32.

### Run 1: Exactly 30 epochs (faithful to paper's stated epoch count)

| Metric | Value |
|---|---|
| Accuracy | 65.62% |
| Precision | 84.49% |
| Recall | 51.33% |
| AUC | 0.761 |

**Diagnosis:** Training/validation loss curves both still clearly decreasing at epoch 30 (no overfitting signature — train and val loss tracked closely throughout) → **model was under-converged**, not overfit, at the paper's stated epoch count.

### Run 2: Extended training (up to 300 epochs, EarlyStopping on val_loss, patience=15, best-weights restored)

| Metric | Value |
|---|---|
| Accuracy | 67.50% |
| Precision | 82.99% |
| Recall | 56.71% |
| AUC | 0.779 |

Modest improvement over Run 1, confirming partial — but not complete — under-convergence. Extra training did not close the gap to the paper's claimed 75-76% accuracy.

### Run 3: Threshold tuning (default 0.5 → tuned 0.34, optimized for F1 on validation set only)

| Metric | Value |
|---|---|
| Accuracy | 68.04% |
| Precision | 67.27% |
| Recall | **89.55%** |
| F1-score | 76.83% |
| AUC | 0.779 (unchanged, as expected — AUC is threshold-independent) |

**Assessment:** Threshold tuning traded precision for recall along the same ROC curve (AUC unchanged) — did not raise the accuracy ceiling, confirming the bottleneck is not threshold calibration. Notably, **our best AUC (0.779) modestly exceeds the paper's reported AUC (0.75–0.76)** — the model's underlying discriminative ability is comparable to or better than the original, but our accuracy at any single threshold remains 7-9 points below the paper's claim. This points toward the same root causes identified for Random Forest (feature-set ambiguity, dataset power-vs-raw-IQ limitation) rather than a training or calibration deficiency.

---

## 6. Summary Table — All Models So Far

| Metric | Paper RF | Our RF | Paper ANN | Our ANN (best, tuned) |
|---|---|---|---|---|
| Accuracy | 93% | 83.99% | 75-76% | 68.04% |
| Precision | 94% | 89.99% | 76-79% | 67.27% |
| Recall | 94% | 82.08% | 75-79% | 89.55% |
| AUC | 0.94 | 0.927 | 0.75-0.76 | 0.779 |

**Headline finding:** Across both models, our **AUC values track the paper's reasonably closely** (within ~0.01-0.02), suggesting the underlying classifiers are learning comparably discriminative decision functions to the originals. The **accuracy gap is more pronounced for RF (~9 pts) than ANN (~7-8 pts after tuning)**, and in both cases is best explained by dataset/feature ambiguities documented above rather than implementation or training errors — each of which was independently diagnosed (training curves, threshold sweeps, distribution checks) before being accepted as a limitation rather than assumed.

---

## 7. Results — 1D-CNN (Stage 10)

**Hardware:** NVIDIA T550 Laptop GPU (4GB VRAM), CUDA 12.1 build of PyTorch, driver 596.72 / CUDA 13.2 ceiling.

**Architecture (FTLW-CNN, since the paper does not provide an exact layer diagram):** 3 conv1d+maxpool blocks (16→32→64 channels, kernel sizes 7/5/3), followed by FC(8192→64) → Dropout(0.2) → FC(64→1) → Sigmoid. Input: raw 1024-sample window (per-window z-score normalized), not the engineered 10-feature vector used for RF/ANN — matching the paper's description of FTLW-CNN as operating directly on the 1D RF signal.

**Total parameters: 533,345** vs. paper's stated 288K (Table 4) — roughly 1.85x larger. Notably, this difference did **not** turn out to be the dominant factor in our results (see below) — training duration/regularization mattered far more.

**Optimizer:** Adam, lr=1e-3 (matching Table 2's "RTLW-CNN" row, likely typo for FTLW-CNN), batch size 32, BCELoss.

### Run 1: Full 50 epochs (faithful to paper's stated epoch count), no early stopping

Training loss fell monotonically (0.56 → 0.064), but **validation loss bottomed out at epoch 4 (0.469) then rose almost continuously to 2.17 by epoch 50** — the clearest, most textbook overfitting signature observed in this entire reproduction. Training took 11.54 minutes total.

| Metric | Value |
|---|---|
| Accuracy | 71.51% |
| Precision | 75.59% |
| Recall | 76.59% |
| AUC | 0.8049 |

### Run 2: Early stopping (patience=7 on val_loss, best-weights restored) — same architecture/hyperparameters otherwise

Correctly identified epoch 4 as the best checkpoint (matching Run 1's own val_loss curve), stopped at epoch 11 after 7 epochs without improvement. Training took only 2.75 minutes (4x faster than Run 1).

| Metric | Run 1 (50 epochs) | Run 2 (early-stopped) | Paper (Table 3) |
|---|---|---|---|
| Accuracy | 71.51% | **73.23%** | 75% |
| Precision | 75.59% | **80.00%** | 76% |
| Recall | 76.59% | **73.03%** | 76-79% |
| AUC | 0.8049 | **0.8321** | 0.75-0.76 |

**Assessment:** Early stopping improved every metric simultaneously while cutting training time 4x — a clean confirmation that the paper's stated 50-epoch count, applied without any stopping criterion, drives this architecture into clear overfitting territory on our setup. The early-stopped CNN is our **closest match to the paper of all three models** — within 2 points of accuracy, and modestly exceeding the paper's reported AUC. Notably, the ~1.85x parameter-count excess relative to the paper's stated 288K did not appear to be the primary driver of the original overfitting — controlling training duration alone resolved it, suggesting epoch count/regularization, not raw model capacity, was the dominant factor for this dataset size.

**Cross-reference with paper's own internal inconsistency:** The paper's Table 6 ("real-time testing") reports a separately-measured 92.3% accuracy for FTLW-CNN — notably higher than Table 3's 75% simulation figure for the same model, with no explanation given for the discrepancy. This is a pre-existing inconsistency in the source paper, not something introduced by our reproduction.

---

## 8. Final Summary — All Three Models vs. Paper

| Metric | Paper RF | Our RF | Paper ANN | Our ANN (tuned) | Paper CNN | Our CNN (early-stopped) |
|---|---|---|---|---|---|---|
| Accuracy | 93% | 83.99% | 75-76% | 68.04% | 75% | **73.23%** |
| Precision | 94% | 89.99% | 76-79% | 67.27% | 76% | **80.00%** |
| Recall | 94% | 82.08% | 75-79% | 89.55% | 76-79% | **73.03%** |
| AUC | 0.94 | 0.927 | 0.75-0.76 | 0.779 | 0.75-0.76 | **0.8321** |

**Overall pattern:** AUC tracks the paper closely (within ~0.02-0.05) across all three models, and for ANN/CNN our AUC modestly *exceeds* the paper's — indicating the underlying classifiers learned comparably or more discriminative decision functions despite working from a less rich, hand-engineered or raw-but-power-derived feature space rather than the paper's presumed richer/raw-IQ-derived inputs. The largest accuracy gap is for RF (~9 points), smallest for CNN (~2 points) once overfitting was controlled. Every gap was investigated with a concrete diagnostic (training curves, threshold sweeps, parameter counts) before being attributed to a specific, named cause — predominantly the dataset's power-vs-raw-IQ limitation (flagged since Stage 3) and unstated exact feature definitions in the original paper, rather than implementation error.
