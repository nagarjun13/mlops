import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

# Load model at startup (mount /models in Docker or bake into image)
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
pipe = joblib.load(MODEL_PATH)

class MatchFeatures(BaseModel):
    home_team: str
    away_team: str
    home_pts_last5: float
    home_gf_avg_last5: float
    home_ga_avg_last5: float
    away_pts_last5: float
    away_gf_avg_last5: float
    away_ga_avg_last5: float

    @field_validator('home_team', 'away_team')
    @classmethod
    def trim(cls, v: str) -> str:
        return v.strip()

app = FastAPI(title="EPL Predictor", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: MatchFeatures):
    X = [payload.model_dump()]
    proba = pipe.predict_proba(X)[0].tolist()
    classes = list(map(str, pipe.classes_.tolist()))
    # Map to human labels
    label_map = {'H': 'home_win', 'D': 'draw', 'A': 'away_win'}
    return {
        "classes": [label_map.get(c, c) for c in classes],
        "probabilities": proba
    }

