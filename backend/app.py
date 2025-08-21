from flask import Flask, request, jsonify
import pandas as pd
import joblib, os

app = Flask(__name__)

# Load model
model_path = "model/churn_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found! Please run train_model.py first.")
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        # Predict churn
        preds = model.predict(df)
        df["Churn Prediction"] = preds

        churn_rate = float(df["Churn Prediction"].mean()) * 100

        return jsonify({
            "churn_rate": churn_rate,
            "predictions": df["Churn Prediction"].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
