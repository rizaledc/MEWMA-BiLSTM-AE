# MEWMA-BiLSTM-AE

**Hybrid Two-Layer Anomaly Detection for IoT Smart Dormitory: Integrating MEWMA and BiLSTM-Autoencoder with Score-Level Fusion**

---

## Overview

This repository contains the code and experimental artifacts for a hybrid anomaly detection system designed for multi-node IoT smart dormitory environments. The system combines two complementary detection layers:

- **Layer 1 — MEWMA** (Multivariate Exponentially Weighted Moving Average): A statistical process control chart that detects gradual statistical shifts across 24 sensor features with low computational cost and high interpretability.
- **Layer 2 — BiLSTM-AE** (Bidirectional LSTM Autoencoder): A deep learning model that captures complex temporal and non-linear patterns via reconstruction error on sliding window sequences.
- **Fusion Engine**: A score-level weighted fusion that combines both layers, achieving complementary detection — the only method that detects all critical events (fire, water leak, panic) while maintaining the lowest false positive rate.

## Dataset

- **Source**: Real operational data from a smart dormitory IoT system (confidential industry source)
- **Period**: November 2025 – June 2026 (8 months)
- **Granularity**: 30-minute intervals, 11,616 records
- **Nodes**: 7 categories (comfort, safety, energy, water, access control, facility, CCTV) + gateway
- **Features**: 24 sensor + 5 temporal = 29 input dimensions

## Key Results

| Method | AUC-ROC | F1 | FPR | CRITICAL Detected |
|--------|---------|-----|------|-------------------|
| Isolation Forest | 0.712 | 0.289 | 0.200 | 2/4 |
| One-Class SVM | 0.694 | 0.287 | 0.150 | 3/4 |
| MEWMA (Layer 1) | 0.648 | 0.240 | 0.200 | 2/4 |
| BiLSTM-AE (Layer 2) | 0.641 | 0.239 | 0.200 | 3/4 |
| **Hybrid Fusion (Ours)** | **0.670** | **0.278** | **0.150** | **4/4** |

> Hybrid Fusion is the **only method** that detects all 4 critical events (fire, water leak, panic) in the test set, while maintaining the lowest false positive rate (0.150).

## Repository Structure

```
MEWMA-BiLSTM-AE/
├── Percobaan 1/                          # First trial experiments
│   ├── Code First Trial/
│   ├── Pict/
│   └── experiment_report.md
│
├── Percobaan 2/                          # Main experiments (current)
│   ├── notebooks/
│   │   ├── 01_data_loading_eda.ipynb     # Data loading + EDA
│   │   ├── 02_mewma.ipynb               # MEWMA Layer 1
│   │   ├── 03_bilstm_ae.ipynb           # BiLSTM-AE Layer 2
│   │   ├── 04_baselines.ipynb           # IF, OCSVM baselines
│   │   └── 05_fusion_evaluation.ipynb   # Fusion + final evaluation
│   ├── figures/
│   │   ├── fig05_mewma_scores.png
│   │   ├── fig06_bilstm_scores.png
│   │   ├── fig07_roc_curves.png
│   │   └── fig08_critical_case_study.png
│   ├── artifacts/                        # Model weights & scores (.gitignore large files)
│   └── experiment_report.md
│
└── README.md
```

## Layer Complementarity

The core finding is that each layer captures different types of anomalies:

| Critical Event | Type | MEWMA Score | BiLSTM Score | Fusion Score | Dominant Layer |
|---------------|------|-------------|--------------|--------------|----------------|
| CRIT 1 | Panic | 0.903 | 0.660 | 0.782 | MEWMA |
| CRIT 2 | Water Leak | 0.773 | 0.930 | 0.852 | BiLSTM |
| CRIT 3 | Fire | 0.914 | 1.000 | 0.957 | Both |
| CRIT 4 | Water Leak | 0.605 | 0.995 | 0.800 | BiLSTM |

MEWMA excels at sudden multivariate shifts (panic button), BiLSTM-AE excels at sensor-pattern anomalies (water leak, fire). Fusion captures both.

## Requirements

```
python >= 3.10
pandas >= 2.0
numpy >= 1.24
tensorflow >= 2.15
scikit-learn >= 1.3
scipy >= 1.11
matplotlib >= 3.7
seaborn >= 0.13
statsmodels >= 0.14
```

## How to Run

1. Upload `Smart_Dormitory_CLEAN_2025-11-01_to_2026-06-30.csv` to Google Colab
2. Run notebooks in order (01 → 05)
3. Each notebook is self-contained — loads data from CSV and recomputes preprocessing

## Authors

- Rizal Wahyu Pratama
- Khulika Malkan

## License

This project is part of academic research. Please contact the authors for usage permissions.
