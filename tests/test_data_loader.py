from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.data_loader import assign_grade_label, load_dataset, prepare_train_test_split


def test_assign_grade_label_uses_expected_boundaries():
    assert assign_grade_label(3.9) == "A"
    assert assign_grade_label(3.6) == "A"
    assert assign_grade_label(3.59) == "B"
    assert assign_grade_label(3.2) == "B"
    assert assign_grade_label(3.19) == "C"
    assert assign_grade_label(2.5) == "C"
    assert assign_grade_label(2.49) == "D"
    assert assign_grade_label(2.0) == "D"
    assert assign_grade_label(1.99) == "F"


def test_load_dataset_raises_for_missing_columns():
    temp_dir = Path(".tmp")
    temp_dir.mkdir(exist_ok=True)
    csv_path = temp_dir / f"bad_{uuid4().hex}.csv"
    pd.DataFrame({"Student_ID": ["ID0001"], "Gender": ["Male"]}).to_csv(csv_path, index=False)

    try:
        with pytest.raises(ValueError, match="Missing required columns"):
            load_dataset(csv_path)
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_prepare_train_test_split_returns_80_20_split():
    rows = []
    grades = [3.8] * 20 + [3.3] * 20 + [2.7] * 20 + [2.2] * 20 + [1.5] * 20
    for index, cgpa in enumerate(grades, start=1):
        rows.append(
            {
                "Student_ID": f"ID{index:05d}",
                "Gender": "Male" if index % 2 else "Female",
                "Age": 20,
                "Major": "Engineering",
                "Attendance_Pct": 80.0,
                "Study_Hours_Per_Day": 4.0,
                "Previous_CGPA": 3.0,
                "Sleep_Hours": 7.0,
                "Social_Hours_Week": 6,
                "Final_CGPA": cgpa,
            }
        )

    df = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test, processed = prepare_train_test_split(df)

    assert len(processed) == 100
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "Final_CGPA" not in X_train.columns
    assert "Student_ID" not in X_train.columns
    assert set(y_train.unique()) == {"A", "B", "C", "D", "F"}
    assert set(y_test.unique()) == {"A", "B", "C", "D", "F"}
