import pathlib

# Project root (one level above src/)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "synthetic_healthcare.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "synthetic_healthcare_dp.csv"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Differential Privacy
DP_EPSILON = 1.0
DP_SENSITIVITY = 1.0

# ML
TEST_SIZE = 0.2
RANDOM_STATE = 42
