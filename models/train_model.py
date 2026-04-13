import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Путь к данным
DATA_PATH = os.path.join("data", "raw", "UCI_Credit_Card.csv")

# Путь для сохранения модели
MODEL_PATH = os.path.join("models", "model_v1.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Удаляем лишний столбец ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


def preprocess(df):
    # Целевая переменная
    target = "default.payment.next.month"

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def train():
    print("Загрузка данных...")
    df = load_data()

    print("Предобработка...")
    X, y = preprocess(df)

    print("Разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Обучение модели...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Оценка модели...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Сохранение модели...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    train()
    