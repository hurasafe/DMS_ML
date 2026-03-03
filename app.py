import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

model   = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

# These must match your original dataset columns exactly
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_cols   = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

class StrokeRequest(BaseModel):
    age: float
    hypertension: int           # 0 or 1
    heart_disease: int          # 0 or 1
    avg_glucose_level: float
    bmi: float
    gender: str                 # e.g. "Male" / "Female"
    ever_married: str           # "Yes" / "No"
    work_type: str              # "Private" / "Self-employed" / "Govt_job" / "children" / "Never_worked"
    Residence_type: str         # "Urban" / "Rural"
    smoking_status: str         # "never smoked" / "formerly smoked" / "smokes" / "Unknown"

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(req: StrokeRequest):
    try:
        # 1. Rebuild the raw input as a dataframe
        raw = pd.DataFrame([{
            "age": req.age,
            "hypertension": req.hypertension,
            "heart_disease": req.heart_disease,
            "avg_glucose_level": req.avg_glucose_level,
            "bmi": req.bmi,
            "gender": req.gender,
            "ever_married": req.ever_married,
            "work_type": req.work_type,
            "Residence_type": req.Residence_type,
            "smoking_status": req.smoking_status,
        }])

        # 2. One-hot encode the categorical columns
        ohe_array = encoder.transform(raw[categorical_cols])
        ohe_df    = pd.DataFrame(ohe_array, columns=encoder.get_feature_names_out(categorical_cols))

        # 3. Combine numerical + encoded columns (same order as training)
        final = pd.concat([raw[numerical_cols].reset_index(drop=True), ohe_df], axis=1)

        # 4. Predict
        prediction  = int(model.predict(final)[0])
        probability = model.predict_proba(final)[0].tolist()

        return {
            "stroke_prediction": prediction,        # 0 = no stroke, 1 = stroke
            "probability": {
                "no_stroke": round(probability[0], 4),
                "stroke":    round(probability[1], 4),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
