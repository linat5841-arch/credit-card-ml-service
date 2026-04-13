from flask import Flask, request, jsonify

from model_handler import load_model, predict


app = Flask(__name__)

# Загружаем модель при старте
model = load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Пустой JSON"}), 400

        result = predict(data, model)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    