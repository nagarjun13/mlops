# src/app.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load model at startup
model = joblib.load("models/model.joblib")

app = FastAPI(title="Premier League Predictor")

# Mount static directory for serving images (src/static)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


class MatchInput(BaseModel):
    home_team: str
    away_team: str


@app.get("/", response_class=HTMLResponse)
def form_page():
    return """
    <html>
        <head>
            <title>Premier League Predictor</title>
            <style>
                body {
                    background-image: url('/static/epl_wallpaper.jpg');
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                    font-family: Arial, sans-serif;
                    color: white;
                    text-align: center;
                    padding-top: 100px;
                }
                .form-container {
                    background: rgba(0, 0, 0, 0.7);
                    padding: 20px;
                    border-radius: 10px;
                    display: inline-block;
                }
                input, button {
                    padding: 10px;
                    margin: 5px;
                    border-radius: 5px;
                    border: none;
                }
                button {
                    background: #ffcc00;
                    font-weight: bold;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div class="form-container">
                <h1>Premier League Match Predictor</h1>
                <form action="/predict_form" method="post">
                    <input type="text" name="home_team" placeholder="Home Team" required>
                    <input type="text" name="away_team" placeholder="Away Team" required>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </body>
    </html>
    """


@app.post("/predict")
def predict_match(match: MatchInput):
    df = pd.DataFrame([{
        "HomeTeam": match.home_team,
        "AwayTeam": match.away_team
    }])

    pred = model.predict(df)[0]

    return {
        "home_team": match.home_team,
        "away_team": match.away_team,
        "predicted_result": pred  # 'H', 'A', 'D'
    }


@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(home_team: str = Form(...), away_team: str = Form(...)):
    df = pd.DataFrame([{
        "HomeTeam": home_team,
        "AwayTeam": away_team
    }])

    pred = model.predict(df)[0]

    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    background-image: url('/static/epl_wallpaper.jpg');
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                    font-family: Arial, sans-serif;
                    color: white;
                    text-align: center;
                    padding-top: 100px;
                }}
                .result-container {{
                    background: rgba(0, 0, 0, 0.7);
                    padding: 20px;
                    border-radius: 10px;
                    display: inline-block;
                }}
                a {{
                    color: #ffcc00;
                    text-decoration: none;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="result-container">
                <h2>Prediction Result</h2>
                <p><b>{home_team}</b> vs <b>{away_team}</b></p>
                <p>Predicted Outcome: <b>{pred}</b></p>
                <a href="/">🔙 Back</a>
            </div>
        </body>
    </html>
    """

