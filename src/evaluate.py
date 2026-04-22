from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import LABEL_ORDER


def calculate_error_rate(accuracy: float) -> float:
    return round(1.0 - accuracy, 4)


def evaluate_predictions(y_true, y_pred) -> dict:
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    return {
        "accuracy": accuracy,
        "error_rate": calculate_error_rate(accuracy),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "classification_report": classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER),
    }


def build_summary_row(
    model_name: str,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
    train_accuracy: float,
) -> pd.Series:
    return pd.Series(
        {
            "model": model_name,
            "accuracy": accuracy,
            "error_rate": calculate_error_rate(accuracy),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "train_accuracy": train_accuracy,
        }
    )


def save_text_report(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
