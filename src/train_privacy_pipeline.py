import pandas as pd
from sklearn.metrics import accuracy_score
from .config import PROCESSED_DATA_PATH
from .synthetic_healthcare import generate_synthetic_healthcare
from .data_preprocessing import preprocess_and_save
from .models import train_logistic_regression, FEATURE_COLUMNS
from .he_paillier import PaillierContext


def ensure_dp_data():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Loaded DP dataset from {PROCESSED_DATA_PATH}")
        return df
    except FileNotFoundError:
        print("DP dataset not found. Generating raw + DP-perturbed data...")
        raw_df = generate_synthetic_healthcare()
        raw_df.to_csv(PROCESSED_DATA_PATH.parent.parent / "raw" / "synthetic_healthcare.csv", index=False)
        preprocess_and_save()
        return pd.read_csv(PROCESSED_DATA_PATH)


def encrypted_inference_demo(model, X_test, y_test):
    he = PaillierContext()
    w = model.coef_[0]
    b = float(model.intercept_[0])

    y_pred = []
    for _, row in X_test.iterrows():
        enc_x = he.encrypt_vector(row.values)
        enc_score = he.encrypted_dot(enc_x, w, b)
        score = he.decrypt(enc_score)
        prob = 1 / (1 + pow(2.718281828, -score))
        y_pred.append(int(prob >= 0.5))

    acc = accuracy_score(y_test, y_pred)
    return acc


def main():
    df_dp = ensure_dp_data()

    model, acc_plain_on_dp, (X_test, y_test) = train_logistic_regression(df_dp)
    print(f"[DP PLAINTEXT] Test accuracy on DP-perturbed data: {acc_plain_on_dp:.4f}")

    acc_enc = encrypted_inference_demo(model, X_test[FEATURE_COLUMNS], y_test)
    print(f"[HE ENCRYPTED INFERENCE] Test accuracy: {acc_enc:.4f}")


if __name__ == "__main__":
    main()
