#!/usr/bin/env bash
set -e

python -m src.synthetic_healthcare
python -m src.train_plaintext
