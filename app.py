import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    sex = request.form["sex"]
    chest_pain = request.form["chest_pain"]
    resting_bp = float(request.form["resting_bp"])
    cholesterol = float(request.form["cholesterol"])
    fasting_bs = int(request.form["fasting_bs"])
    resting_ecg = request.form["resting_ecg"]
    max_hr = float(request.form["max_hr"])
    exercise_angina = request.form["exercise_angina"]
    oldpeak = float(request.form["oldpeak"])
    st_slope = request.form["st_slope"]

    data = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak, "ST_Slope": st_slope
    }
    input_df = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_df, columns=["Sex", "ChestPainType",
                                   "RestingECG", "ExerciseAngina", "ST_Slope"],
                                   drop_first=True)

    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_cols]

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    risk_pct = round(probability[1] * 100, 1)
    confidence = round(max(probability) * 100, 1)
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

    return render_template("index.html", result=result, risk_pct=risk_pct,
                           confidence=confidence, prediction=int(prediction),
                           show_result=True)
@app.route("/findings")
def findings():
    return render_template("findings.html")
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)