import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE, EXPERIMENTS_DIR
from .synthetic_healthcare import generate_synthetic_healthcare
from .data_preprocessing import preprocess_and_save
from .he_paillier import PaillierContext
from .models import FEATURE_COLUMNS, TARGET_COLUMN


def load_or_generate():
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        df = generate_synthetic_healthcare()
        df.to_csv(RAW_DATA_PATH, index=False)
    try:
        df_dp = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        preprocess_and_save()
        df_dp = pd.read_csv(PROCESSED_DATA_PATH)
    return df, df_dp


def benchmark():
    df, df_dp = load_or_generate()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model_plain = LogisticRegression(max_iter=500)

    t0 = time.time()
    model_plain.fit(X_train, y_train)
    train_time_plain = time.time() - t0

    t0 = time.time()
    y_pred_plain = model_plain.predict(X_test)
    inf_time_plain = time.time() - t0
    acc_plain = accuracy_score(y_test, y_pred_plain)

    X_dp = df_dp[FEATURE_COLUMNS]
    y_dp = df_dp[TARGET_COLUMN]
    X_train_dp, X_test_dp, y_train_dp, y_test_dp = train_test_split(
        X_dp, y_dp, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_dp
    )

    model_dp = LogisticRegression(max_iter=500)
    t0 = time.time()
    model_dp.fit(X_train_dp, y_train_dp)
    train_time_dp = time.time() - t0

    he = PaillierContext()
    enc_inf_start = time.time()
    y_pred_enc = []
    for _, row in X_test_dp.iterrows():
        enc_x = he.encrypt_vector(row.values)
        enc_score = he.encrypted_dot(enc_x, model_dp.coef_[0], float(model_dp.intercept_[0]))
        score = he.decrypt(enc_score)
        prob = 1 / (1 + pow(2.718281828, -score))
        y_pred_enc.append(int(prob >= 0.5))
    enc_inf_time = time.time() - enc_inf_start
    acc_enc = accuracy_score(y_test_dp, y_pred_enc)

    latency_overhead = enc_inf_time / max(inf_time_plain, 1e-9)

    results = {
        "acc_plain": acc_plain,
        "acc_dp_he": acc_enc,
        "train_time_plain": train_time_plain,
        "train_time_dp": train_time_dp,
        "inf_time_plain": inf_time_plain,
        "inf_time_dp_he": enc_inf_time,
        "latency_overhead_x": latency_overhead,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / "results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print("Benchmark results:", results)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    benchmark()
