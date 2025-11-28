# Privacy-Preserving ML: Healthcare Risk Prediction

Course Project (CS670 - Cryptographic Techniques), IIT Kanpur  
Jan 2024 – Apr 2024

This project implements a **privacy-preserving machine learning pipeline** for healthcare readmission risk prediction using:

- **Paillier Homomorphic Encryption** for encrypted collaborative inference
- **Secure Multi-Party Computation (MPC)** for secure aggregation of statistics
- **Differential Privacy** via Laplacian noise on 50K+ synthetic patient records

## Features

- Synthetic healthcare dataset (50K+ records)
- Plaintext logistic regression baseline
- Differentially private data transformation (Laplace mechanism)
- Encrypted inference using Paillier PHE on model inputs
- MPyC-based secure aggregation demo
- Benchmarking of accuracy and latency vs plaintext baseline

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Plaintext baseline
bash scripts/run_plaintext_baseline.sh

# Privacy-preserving pipeline + benchmark
bash scripts/run_privacy_pipeline.sh
```

Results are stored under `experiments/results.csv`.
