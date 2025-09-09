import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/model.joblib")

# Example: new matches to predict
# Make sure the columns match your training features
new_matches = pd.DataFrame({
    "HomeTeam": ["Arsenal", "Chelsea"],
    "AwayTeam": ["Liverpool", "Manchester United"]
})

# Predict outcomes
predictions = model.predict(new_matches)
print("Predicted outcomes:", predictions)

