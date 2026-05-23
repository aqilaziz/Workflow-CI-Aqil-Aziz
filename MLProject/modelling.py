from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("breast_cancer_random_forest")
    mlflow.sklearn.autolog(log_models=True)

    active_run = mlflow.active_run()
    run_context = (
        nullcontext(active_run)
        if active_run
        else mlflow.start_run(run_name="baseline_random_forest_autolog")
    )

    with run_context as run:
        model = RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        summary = {
            "run_id": run.info.run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment": "breast_cancer_random_forest",
            "logging": "mlflow.sklearn.autolog",
        }
        (Path(__file__).resolve().parent / "run_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
