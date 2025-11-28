import numpy as np
import pandas as pd
from .config import RAW_DATA_PATH, DATA_DIR, RANDOM_STATE


def generate_synthetic_healthcare(n_samples: int = 50000) -> pd.DataFrame:
    """
    Generate a simple synthetic 'healthcare' dataset.
    Binary classification: readmission_risk (0/1).
    """
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 90, size=n_samples)
    bmi = rng.normal(loc=27, scale=5, size=n_samples).clip(15, 45)
    systolic_bp = rng.normal(loc=120, scale=15, size=n_samples).clip(80, 200)
    cholesterol = rng.normal(loc=190, scale=30, size=n_samples).clip(100, 350)
    num_conditions = rng.integers(0, 5, size=n_samples)
    smoker = rng.integers(0, 2, size=n_samples)
    diabetic = rng.integers(0, 2, size=n_samples)

    logits = (
        0.03 * (age - 50)
        + 0.05 * (bmi - 25)
        + 0.02 * (systolic_bp - 120)
        + 0.04 * num_conditions
        + 0.5 * smoker
        + 0.6 * diabetic
    )
    probs = 1 / (1 + np.exp(-logits))
    readmission_risk = rng.binomial(1, probs)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "num_conditions": num_conditions,
            "smoker": smoker,
            "diabetic": diabetic,
            "readmission_risk": readmission_risk,
        }
    )
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_healthcare()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved synthetic dataset with {len(df)} rows to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
