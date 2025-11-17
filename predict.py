import pickle
from flask import Flask, request, jsonify

model_file = "stroke_model.bin"

with open(model_file, "rb") as f_in:
    dv, scaler, model, bmi_median = pickle.load(f_in)

app = Flask("stroke-predictor")

BEST_THRESHOLD = 0.164  # From validation

@app.route("/predict", methods=["POST"])
def predict():

    patient = request.get_json()

    # Missing BMI handling
    if patient.get("bmi") is None:
        patient["bmi"] = bmi_median

    # Convert dict → one-hot
    X = dv.transform([patient])
    X = scaler.transform(X)

    # Predict probability
    prob = model.predict_proba(X)[0, 1]

    # Apply threshold
    stroke = prob >= BEST_THRESHOLD

    result = {
        "stroke_probability": float(prob),
        "stroke": bool(stroke)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
