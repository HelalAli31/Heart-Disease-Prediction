"""
Predict heart disease for a new patient using the saved model.

Train the model first with:   python main.py
Then run this script with:    python predict.py

To try a different patient, edit the EXAMPLE_PATIENT dictionary below.

DISCLAIMER: This is an educational machine learning project.
The prediction is NOT a medical diagnosis.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "heart_disease_model.joblib"

# The 11 input features, in the same form as the columns of heart.csv.
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
# Same cleaning rule as main.py: a value of 0 here means "not recorded".
ZERO_MEANS_MISSING = ["RestingBP", "Cholesterol"]

# ===========================================================================
# EXAMPLE PATIENT (for demonstration only - an invented, not a real, person)
# ===========================================================================
EXAMPLE_PATIENT = {
    "Age": 58,               # years
    "Sex": "M",              # M = male, F = female
    "ChestPainType": "ASY",  # TA, ATA, NAP or ASY (asymptomatic)
    "RestingBP": 140,        # resting blood pressure [mm Hg]
    "Cholesterol": 260,      # serum cholesterol [mg/dl] (0 = not recorded)
    "FastingBS": 1,          # 1 if fasting blood sugar > 120 mg/dl, else 0
    "RestingECG": "Normal",  # Normal, ST or LVH
    "MaxHR": 120,            # maximum heart rate achieved
    "ExerciseAngina": "Y",   # exercise-induced angina: Y or N
    "Oldpeak": 1.8,          # ST depression (numeric)
    "ST_Slope": "Flat",      # Up, Flat or Down
}


def load_model(path):
    """Load the trained pipeline saved by main.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python main.py' first to train it."
        )
    return joblib.load(path)


def predict_patient(model, patient):
    """Return (predicted class, probability of heart disease) for one patient."""
    missing = [feature for feature in FEATURES if feature not in patient]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    # One-row DataFrame with the same column names used during training.
    patient_df = pd.DataFrame([patient], columns=FEATURES)
    for column in ZERO_MEANS_MISSING:
        patient_df[column] = patient_df[column].replace(0, np.nan)

    # The saved pipeline imputes, scales and one-hot encodes automatically.
    predicted_class = int(model.predict(patient_df)[0])
    probability = float(model.predict_proba(patient_df)[0, 1])
    return predicted_class, probability


def main():
    model = load_model(MODEL_PATH)
    predicted_class, probability = predict_patient(model, EXAMPLE_PATIENT)

    print("=" * 60)
    print("HEART DISEASE PREDICTION - EXAMPLE PATIENT")
    print("=" * 60)
    for feature, value in EXAMPLE_PATIENT.items():
        print(f"  {feature:<15} {value}")

    label = "Heart disease" if predicted_class == 1 else "No heart disease"
    print("\nPredicted class:", predicted_class, f"({label})")
    print(f"Predicted probability of heart disease: {probability:.1%}")
    print("(Class 1 is predicted when the probability is 50% or higher.)")

    print(
        "\nDISCLAIMER: This is an educational prediction from a student machine\n"
        "learning project. It is NOT a medical diagnosis. Anyone with health\n"
        "concerns should consult a qualified medical professional."
    )


if __name__ == "__main__":
    main()
