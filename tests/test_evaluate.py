from src.evaluate import build_summary_row, calculate_error_rate


def test_calculate_error_rate_returns_one_minus_accuracy():
    assert calculate_error_rate(0.82) == 0.18


def test_build_summary_row_contains_expected_fields():
    row = build_summary_row(
        model_name="baseline",
        accuracy=0.8,
        macro_f1=0.75,
        weighted_f1=0.79,
        train_accuracy=0.95,
    )
    expected_columns = {"model", "accuracy", "error_rate", "macro_f1", "weighted_f1", "train_accuracy"}
    assert expected_columns.issubset(set(row.index))
    assert row["model"] == "baseline"
