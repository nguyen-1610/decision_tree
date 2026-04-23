import pandas as pd

from src.features import build_model_pipeline
from src.visualize import save_full_tree_svg


def test_save_full_tree_svg_writes_svg_output(tmp_path):
    X = pd.DataFrame(
        {
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "Major": [
                "Engineering",
                "Science",
                "Arts",
                "Business",
                "Engineering",
                "Science",
                "Arts",
                "Business",
                "Engineering",
                "Science",
            ],
            "Age": [20, 21, 22, 20, 21, 22, 20, 21, 22, 20],
            "Attendance_Pct": [95.0, 88.0, 78.0, 65.0, 92.0, 85.0, 75.0, 68.0, 90.0, 82.0],
            "Study_Hours_Per_Day": [6.0, 5.0, 4.0, 2.0, 6.5, 5.5, 3.5, 2.5, 6.2, 5.2],
            "Previous_CGPA": [3.8, 3.4, 2.9, 2.2, 3.7, 3.3, 2.8, 2.1, 3.9, 3.2],
            "Sleep_Hours": [7.0, 7.5, 6.5, 6.0, 7.2, 7.0, 6.8, 6.2, 7.1, 6.9],
            "Social_Hours_Week": [4, 5, 7, 9, 4, 5, 8, 10, 3, 6],
        }
    )
    y = pd.Series(["A", "B", "C", "D", "A", "B", "C", "F", "A", "B"])

    model = build_model_pipeline(max_depth=3)
    model.fit(X, y)
    feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())

    output_path = tmp_path / "toy_tree_full.svg"
    save_full_tree_svg(model, feature_names, output_path, "toy tree")

    content = output_path.read_text(encoding="utf-8")
    assert output_path.exists()
    assert "<svg" in content
    assert "toy tree" in content
