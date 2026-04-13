import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# === Загрузка данных ===
df = pd.read_csv("data/raw/UCI_Credit_Card.csv")

# Удаляем ID если есть
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# Переименование target
df = df.rename(columns={"default.payment.next.month": "target"})

X = df.drop("target", axis=1)
y = df["target"]

# === Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === МОДЕЛЬ v1 (Logistic Regression) ===
model_v1 = LogisticRegression(max_iter=1000)
model_v1.fit(X_train, y_train)

with open("models/model_v1.pkl", "wb") as f:
    pickle.dump(model_v1, f)

print("Model v1 saved")

# === МОДЕЛЬ v2 (Random Forest) ===
model_v2 = RandomForestClassifier(n_estimators=50, random_state=42)
model_v2.fit(X_train, y_train)

with open("models/model_v2.pkl", "wb") as f:
    pickle.dump(model_v2, f)

print("Model v2 saved")
