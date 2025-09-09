import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.features import load_raw, build_training_table


def train(data_csv: str, model_out: str, test_size: float = 0.2) -> None:
    # Load raw data
    df = load_raw(data_csv)

    # Build features and labels
    X, y = build_training_table(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Separate categorical and numerical columns
    categorical = ['HomeTeam', 'AwayTeam']
    numerical = [col for col in X.columns if col not in categorical]

    # Preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                numerical
            ),
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=True),  # ✅ fixed
                categorical
            ),
        ]
    )

    # Build pipeline
    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ]
    )

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Accuracy: {acc:.3f}")

    # Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"💾 Model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Premier League match outcome predictor")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--out", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for splitting dataset")

    args = parser.parse_args()
    train(args.data, args.out, args.test_size)

