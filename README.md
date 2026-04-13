# Credit Card Default Prediction ML Service

## Описание проекта

Сервис машинного обучения для прогнозирования дефолта по кредитным картам на основе датасета **Default of Credit Card Clients Dataset** (UCI Machine Learning Repository).

Проект реализует production-like pipeline:
- обучение и сохранение ML-моделей
- загрузку моделей для инференса
- REST API на Flask
- контейнеризацию через Docker
- базовую оркестрацию через Docker Compose
- A/B-тестирование двух версий модели

---

## Цель проекта

Разработать и внедрить сервис прогнозирования дефолта, готовый к развёртыванию и тестированию разных версий моделей в production-like среде.

---

## Датасет

Используется датасет **Default of Credit Card Clients Dataset**.

Содержит:
- демографические данные клиентов
- кредитный лимит
- историю платежей
- суммы счетов и платежей

Целевая переменная:
- `default.payment.next.month` — дефолт в следующем месяце

---

## Используемые модели

В проекте реализованы две версии:

- **v1** — LogisticRegression  
- **v2** — RandomForestClassifier  

Это позволяет реализовать A/B-тестирование.

---

## Структура проекта

```text
Project_ML/
├── app/
│   ├── api.py
│   └── model_handler.py
├── data/
│   └── raw/
├── docker/
│   └── Dockerfile
├── models/
│   ├── model_v1.pkl
│   ├── model_v2.pkl
│   └── train_model.py
├── tests/
│   └── test_api.py
├── ARCHITECTURE.md
├── ab_test_plan.md
├── docker-compose.yml
├── README.md
└── requirements.txt