from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import EXPECTED_COLUMNS, FEATURE_COLUMNS, LABEL_ORDER, RANDOM_STATE, TEST_SIZE


def assign_grade_label(final_cgpa: float) -> str:
    if final_cgpa >= 3.6:
        return "A"
    if final_cgpa >= 3.2:
        return "B"
    if final_cgpa >= 2.5:
        return "C"
    if final_cgpa >= 2.0:
        return "D"
    return "F"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df


def prepare_train_test_split(df: pd.DataFrame):
    processed = df.copy()
    processed["Grade_Label"] = processed["Final_CGPA"].apply(assign_grade_label)
    processed["Grade_Label"] = pd.Categorical(processed["Grade_Label"], categories=LABEL_ORDER, ordered=True)

    if processed["Grade_Label"].isna().any():
        raise ValueError("Grade label generation produced invalid values.")

    class_counts = processed["Grade_Label"].value_counts()
    if (class_counts == 0).any():
        raise ValueError(f"One or more classes are empty after label generation: {class_counts.to_dict()}")

    X = processed[FEATURE_COLUMNS].copy()
    y = processed["Grade_Label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, processed
