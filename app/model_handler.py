import pickle
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_v1.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def predict(model, data):
    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in data]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    input_df = pd.DataFrame([[data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": "v1"
    }