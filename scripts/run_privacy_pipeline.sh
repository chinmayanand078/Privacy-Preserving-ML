#!/usr/bin/env bash
set -e

python -m src.synthetic_healthcare
python -m src.data_preprocessing
python -m src.train_privacy_pipeline
python -m src.benchmark
