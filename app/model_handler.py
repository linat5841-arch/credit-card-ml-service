import os
import pickle
import pandas as pd


MODEL_PATH = os.path.join("models", "model_v1.pkl")

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


def load_model(model_path: str = MODEL_PATH):
    """
    Загружает обученную модель из файла.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Преобразует входной JSON в DataFrame с фиксированным порядком признаков.
    """
    if not isinstance(data, dict):
        raise ValueError("Входные данные должны быть JSON-объектом.")

    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in data]
    if missing_features:
        raise ValueError(
            f"Отсутствуют обязательные признаки: {', '.join(missing_features)}"
        )

    row = {feature: data[feature] for feature in FEATURE_COLUMNS}
    features_df = pd.DataFrame([row])

    return features_df


def predict(data: dict, model) -> dict:
    """
    Выполняет предсказание по входным данным.
    """
    features = preprocess_input(data)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model_version": "v1",
    }