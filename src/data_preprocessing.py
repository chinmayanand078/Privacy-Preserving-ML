import numpy as np
import pandas as pd
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, DP_EPSILON, DP_SENSITIVITY, DATA_DIR

NUMERICAL_FEATURES = ["age", "bmi", "systolic_bp", "cholesterol", "num_conditions"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def add_laplace_noise(
    df: pd.DataFrame,
    epsilon: float = DP_EPSILON,
    sensitivity: float = DP_SENSITIVITY,
    features=NUMERICAL_FEATURES,
) -> pd.DataFrame:
    """
    Add Laplace noise to selected numerical features
    for (ε, 0)-Differential Privacy at the record level.
    """
    noisy_df = df.copy()
    scale = sensitivity / epsilon
    for col in features:
        noise = np.random.laplace(loc=0.0, scale=scale, size=len(df))
        noisy_df[col] = noisy_df[col] + noise
    return noisy_df


def preprocess_and_save():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    df = load_raw()
    noisy_df = add_laplace_noise(df)
    noisy_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved DP-perturbed dataset to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_and_save()
