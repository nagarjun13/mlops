import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load raw match data from CSV and normalize column names."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse Date column safely
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Normalize column names for consistency
    rename_map = {
        "Home": "HomeTeam",   # Kaggle dataset
        "Away": "AwayTeam",   # Kaggle dataset
        "Winner": "FTR"       # Kaggle dataset
    }
    df.rename(columns=rename_map, inplace=True)

    # Add match_id column
    df = df.sort_values("Date").reset_index(drop=True)
    df["match_id"] = df.index.astype(int)

    return df


def build_training_table(df: pd.DataFrame):
    """Build feature matrix (X) and labels (y) for training."""
    required_cols = ["match_id", "Date", "HomeTeam", "AwayTeam", "FTR"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    base = df[required_cols].copy()

    # Features (categorical: teams), label (FTR)
    X = base[["HomeTeam", "AwayTeam"]]
    y = base["FTR"]

    return X, y


def make_encoder():
    """Return a OneHotEncoder compatible with modern sklearn versions."""
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)

