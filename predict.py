import pickle
from flask import Flask, request, jsonify

model_file = "stroke_model.bin"

with open(model_file, "rb") as f_in:
    dv, scaler, model, bmi_median = pickle.load(f_in)

app = Flask("stroke-predictor")

@app.route("/predict", methods=["POST"])
def predict():

    patient = request.get_json()

    # missing BMI handling
    if patient.get("bmi") is None:
        patient["bmi"] = bmi_median

    # dict → vector
    X = dv.transform([patient])
    X = scaler.transform(X)

    # decision tree outputs
    prob = model.predict_proba(X)[0, 1]     # probability
    stroke = model.predict(X)[0]            # class (0 or 1)

    result = {
        "stroke_probability": float(prob),
        "stroke": bool(stroke)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
