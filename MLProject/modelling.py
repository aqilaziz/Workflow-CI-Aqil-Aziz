from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


TARGET_COLUMN = "diagnosis"
DATA_DIR = Path(__file__).resolve().parent / "breast_cancer_preprocessing"
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return x_train, y_train, x_test, y_test


def main() -> dict:
    x_train, y_train, x_test, y_test = load_data()
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("breast_cancer_random_forest")
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="baseline_random_forest_autolog") as run:
        model = RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }
        mlflow.log_metrics(metrics)
        joblib.dump(model, Path(__file__).resolve().parent / "model.pkl")
        summary = {
            "run_id": run.info.run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "metrics": metrics,
        }
        Path("run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
