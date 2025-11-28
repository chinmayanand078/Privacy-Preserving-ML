import pandas as pd
from .config import RAW_DATA_PATH
from .synthetic_healthcare import generate_synthetic_healthcare
from .models import train_logistic_regression


def main():
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Loaded existing dataset from {RAW_DATA_PATH}")
    except FileNotFoundError:
        print("Raw dataset not found. Generating synthetic healthcare data...")
        df = generate_synthetic_healthcare()
        df.to_csv(RAW_DATA_PATH, index=False)

    model, acc, _ = train_logistic_regression(df)
    print(f"[PLAINTEXT] Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
