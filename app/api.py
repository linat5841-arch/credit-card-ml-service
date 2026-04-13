from flask import Flask, request, jsonify
from app.model_handler import load_model, predict

app = Flask(__name__)

model = load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Empty request body"}), 400

        result = predict(model, data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
