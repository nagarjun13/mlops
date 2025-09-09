# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model at startup
model = joblib.load("models/model.joblib")

app = FastAPI(title="Premier League Predictor")

class MatchInput(BaseModel):
    home_team: str
    away_team: str

@app.post("/predict")
def predict_match(match: MatchInput):
    # Build dataframe with input
    df = pd.DataFrame([{
        "HomeTeam": match.home_team,
        "AwayTeam": match.away_team
    }])

    # Predict
    pred = model.predict(df)[0]

    return {
        "home_team": match.home_team,
        "away_team": match.away_team,
        "predicted_result": pred  # e.g. 'H', 'A', 'D'
    }

