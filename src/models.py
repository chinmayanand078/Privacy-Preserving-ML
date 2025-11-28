import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .config import TEST_SIZE, RANDOM_STATE

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "cholesterol",
    "num_conditions",
    "smoker",
    "diabetic",
]
TARGET_COLUMN = "readmission_risk"


def train_logistic_regression(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, (X_test, y_test)
