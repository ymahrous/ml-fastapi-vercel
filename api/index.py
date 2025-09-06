import joblib
from flask import Flask, request, jsonify

model = joblib.load("car-recommender.joblib")
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Car Recommender API!"

@app.route("/car/predict", methods=["POST"])
def recommend_car():
    try:
        data = request.get_json()

        if "age" not in data or "gender" not in data:
            return jsonify({"error": "Missing required fields: age and gender"}), 400

        age = data["age"]
        gender = data["gender"]

        recommendation = model.predict([[age, gender]])
        return jsonify({"pred": recommendation[0]})

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
