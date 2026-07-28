# LowSNR_DroneRF: Reproduction of "From Lab to Field Trials: Real-Time Multimodel Drone Detection in Low-SNR Environments"

This repository contains a full software/ML reproduction of the IEEE A&E Systems Magazine paper by Tanveer et al. (Feb 2026), using the publicly released `LowSNR_DroneRF` dataset. No SDR hardware was used — this reproduces the **signal processing and machine learning pipeline only**, starting from the dataset the paper's authors released on Kaggle.

**Full methodology, diagnostics, and results are documented in [`docs/FINDINGS.md`](docs/FINDINGS.md).** This README is a quick orientation; the findings doc is the actual technical report.

## What this reproduces

- Signal processing: FFT, Power Spectral Density (periodogram method), Energy Detection, Empirical Mode Decomposition (EMD) denoising
- Feature engineering: time-domain statistics (from EMD-denoised signal) + frequency-domain spectral features (from raw signal)
- Three classifiers matching the paper's architecture: **Random Forest**, **ANN** (2×64 ReLU + dropout), **1D-CNN**
- Class imbalance handling via SMOTE (training data only, applied after train/test split)
- Full evaluation: accuracy, precision, recall, F1, AUC, confusion matrices

## Key finding

Across all three models, our reproduced **AUC closely tracks the paper's reported values** (within ~0.02–0.05, and for ANN/CNN our AUC modestly *exceeds* the paper's), while raw accuracy runs somewhat below the paper's claims. Every gap is diagnosed with concrete evidence (training curves, threshold sweeps, parameter counts) and traced to specific, named causes — primarily that the public dataset only supports an OFF-vs-ARMED binary task (not OFF-vs-connected, as one section of the paper's text describes) and stores already-processed power values rather than raw complex IQ samples. Full detail in `docs/FINDINGS.md`, Section 1.

| Metric | Paper RF | Our RF | Paper ANN | Our ANN (tuned) | Paper CNN | Our CNN (early-stopped) |
|---|---|---|---|---|---|---|
| Accuracy | 93% | 83.99% | 75-76% | 68.04% | 75% | 73.23% |
| AUC | 0.94 | 0.927 | 0.75-0.76 | 0.779 | 0.75-0.76 | 0.8321 |

## Repository structure

```
scripts/
  01_emd_denoising/        # EMD decomposition + denoising, checkpointed (parallelized)
    01_resume_emd.py
    02_verify_emd.py
  02_feature_engineering/  # Feature extraction, outlier clipping, scaling, SMOTE
    01_feature_engineering.py
    02_scale_features.py
    03_train_test_split_and_smote.py
  03_random_forest/        # FTLW-RF reproduction
    01_random_forest.py
  04_ann/                  # FTLW-ANN reproduction
    01_ann_baseline.py
    02_ann_extended_training.py
    03_diagnose_ann.py
    04_threshold_tuning.py
  05_cnn/                  # FTLW-CNN reproduction
    01_cnn_model.py
    02_cnn_model_earlystop.py
results/
  figures/                 # Training curve plots (ANN baseline, diagnostic, extended)
docs/
  FINDINGS.md              # Full methodology, results, and honest comparison against the paper
requirements.txt           # Python dependencies (pip freeze)
.gitignore
README.md
```

## Dataset

Source: [`laibatanveer/merged`](https://www.kaggle.com/datasets/laibatanveer/merged) on Kaggle (the `LowSNR_DroneRF` dataset referenced in the paper, 137.56 million IQ-derived power samples).

The dataset (~3GB) is **not included in this repository** — `data/` is gitignored. To reproduce:

```bash
pip install kaggle
kaggle datasets download -d laibatanveer/merged -p data --unzip
```

Then run scripts in numbered order, stage folder by stage folder (e.g., everything in `01_emd_denoising/` before moving to `02_feature_engineering/`, and so on through `05_cnn/`). Each script reads its inputs from `data/` and writes its outputs back to `data/`.

## Environment setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

GPU (CUDA) used for CNN training: NVIDIA T550 Laptop GPU, PyTorch with CUDA 12.1 build. CPU is sufficient for the EMD, feature engineering, Random Forest, and ANN stages.

## Known deviations from the paper (see `docs/FINDINGS.md` for full detail)

1. The public dataset contains only **OFF** and **ARMED** labels — not the OFF/connected/armed/flying four-mode set the paper's text describes for its main classification task.
2. The `IQSAMPLES` column in the public dataset is empirically **already power-valued** (non-negative, consistent with the paper's own Complex-to-Mag² step), not raw complex baseband — confirmed via distribution diagnostics, not assumed.
3. The paper does not specify exact input features for FTLW-RF/FTLW-ANN; a defensible 10-feature set (time-domain statistics + spectral features) was engineered and used consistently across both models.
4. The paper's Table 3 (simulation, 75% CNN accuracy) and Table 6 (real-time, 92.3% CNN accuracy) report notably different figures for the same model with no reconciliation given — a pre-existing inconsistency in the source paper.
