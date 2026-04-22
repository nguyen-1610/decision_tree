# Decision Tree Student Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable decision-tree classification project that reads `data/Student_data.csv`, creates `A/B/C/D/F` labels from `Final_CGPA`, trains a baseline model plus three improved variants, and saves tables and figures required for the lab report.

**Architecture:** The project will use a small `src/` package with focused modules for data loading, feature preprocessing, evaluation, and visualization. A single training entry point will orchestrate the full experiment through scikit-learn pipelines so preprocessing, model fitting, evaluation, and artifact generation stay reproducible and easy to rerun. A companion notebook at `notebooks/experiment.ipynb` will present the same workflow interactively by importing functions from `src/`, not by reimplementing the full logic in notebook cells.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, pytest, jupyter, ipykernel

---

## File Structure Map

Planned files and responsibilities:

- `requirements.txt`
  Dependency list for the local `.venv`.
- `README.md`
  Short setup and run instructions for classmates.
- `src/__init__.py`
  Marks `src` as a package.
- `src/config.py`
  Central constants for paths, random seed, label order, and model settings.
- `src/data_loader.py`
  Read CSV, validate schema, derive grade labels, and split train/test with `test_size=0.2`.
- `src/features.py`
  Build the `ColumnTransformer` and model pipelines.
- `src/evaluate.py`
  Compute metrics, save reports, and assemble comparison rows.
- `src/visualize.py`
  Save confusion matrices, decision-tree plots, and feature-importance charts.
- `src/train.py`
  Main script that runs baseline and improved experiments end to end.
- `notebooks/experiment.ipynb`
  Interactive walkthrough for screenshots, quick experiments, and presentation demos.
- `tests/test_data_loader.py`
  Tests for label mapping, schema validation, and 80/20 split behavior.
- `tests/test_features.py`
  Tests for pipeline construction and excluded columns.
- `tests/test_evaluate.py`
  Tests for metrics output shape and error-rate calculation.
- `outputs/figures/.gitkeep`
  Keeps the figures directory present.
- `outputs/reports/.gitkeep`
  Keeps the reports directory present.
- `outputs/tables/.gitkeep`
  Keeps the tables directory present.

Context note:

- the workspace is not a git repository, so commit steps are intentionally replaced with verification checkpoints

### Task 1: Scaffold the project layout and dependency files

**Files:**
- Create: `D:\Class\CSAI\university\requirements.txt`
- Create: `D:\Class\CSAI\university\README.md`
- Create: `D:\Class\CSAI\university\src\__init__.py`
- Create: `D:\Class\CSAI\university\src\config.py`
- Create: `D:\Class\CSAI\university\notebooks\experiment.ipynb`
- Create: `D:\Class\CSAI\university\outputs\figures\.gitkeep`
- Create: `D:\Class\CSAI\university\outputs\reports\.gitkeep`
- Create: `D:\Class\CSAI\university\outputs\tables\.gitkeep`

- [ ] **Step 1: Create the local virtual environment**

Run: `python -m venv .venv`
Expected: a new `.venv` directory appears under `D:\Class\CSAI\university`

- [ ] **Step 2: Write `requirements.txt`**

```text
pandas>=2.2,<3.0
numpy>=2.1,<3.0
scikit-learn>=1.6,<2.0
matplotlib>=3.9,<4.0
seaborn>=0.13,<1.0
joblib>=1.4,<2.0
pytest>=8.3,<9.0
jupyter>=1.1,<2.0
ipykernel>=6.29,<7.0
```

- [ ] **Step 3: Write `README.md` with setup and run commands**

```markdown
# Decision Tree Student Performance

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Place the dataset at `data/Student_data.csv`.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.train
```

## Outputs

Generated files will be saved in:

- `outputs/tables`
- `outputs/reports`
- `outputs/figures`

## Notebook

For step-by-step presentation and screenshots:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Open `notebooks/experiment.ipynb`.
```

- [ ] **Step 4: Write `src/config.py`**

```python
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Student_data.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_ORDER = ["A", "B", "C", "D", "F"]
EXPECTED_COLUMNS = [
    "Student_ID",
    "Gender",
    "Age",
    "Major",
    "Attendance_Pct",
    "Study_Hours_Per_Day",
    "Previous_CGPA",
    "Sleep_Hours",
    "Social_Hours_Week",
    "Final_CGPA",
]

CATEGORICAL_FEATURES = ["Gender", "Major"]
NUMERIC_FEATURES = [
    "Age",
    "Attendance_Pct",
    "Study_Hours_Per_Day",
    "Previous_CGPA",
    "Sleep_Hours",
    "Social_Hours_Week",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
```

- [ ] **Step 5: Create the output directories and package marker**

Run: `New-Item -ItemType Directory -Force -Path 'D:\Class\CSAI\university\src','D:\Class\CSAI\university\tests','D:\Class\CSAI\university\notebooks','D:\Class\CSAI\university\outputs\figures','D:\Class\CSAI\university\outputs\reports','D:\Class\CSAI\university\outputs\tables'`
Expected: all listed directories exist

Write `D:\Class\CSAI\university\src\__init__.py`:

```python
"""Decision tree student performance project package."""
```

- [ ] **Step 6: Verification checkpoint**

Run: `Get-ChildItem -Recurse -Path 'D:\Class\CSAI\university\src','D:\Class\CSAI\university\notebooks','D:\Class\CSAI\university\outputs'`
Expected: `src`, `notebooks`, `outputs\figures`, `outputs\reports`, and `outputs\tables` are present

### Task 2: Implement data loading and grade-label generation from tests first

**Files:**
- Create: `D:\Class\CSAI\university\src\data_loader.py`
- Create: `D:\Class\CSAI\university\tests\test_data_loader.py`

- [ ] **Step 1: Write the failing tests for label mapping and schema validation**

```python
from pathlib import Path

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


def test_load_dataset_raises_for_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"Student_ID": ["ID0001"], "Gender": ["Male"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(csv_path)


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
```

- [ ] **Step 2: Run the data-loader tests to verify they fail**

Run: `pytest tests/test_data_loader.py -v`
Expected: FAIL because `src.data_loader` does not exist yet

- [ ] **Step 3: Write `src/data_loader.py`**

```python
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
```

- [ ] **Step 4: Run the tests again**

Run: `pytest tests/test_data_loader.py -v`
Expected: PASS

- [ ] **Step 5: Verification checkpoint on the real dataset**

Run: `python -c "from src.config import DATA_PATH; from src.data_loader import load_dataset, prepare_train_test_split; df = load_dataset(DATA_PATH); X_train, X_test, y_train, y_test, processed = prepare_train_test_split(df); print(df.shape, len(X_train), len(X_test), sorted(y_train.unique()))"`
Expected: `(5000, 10)` plus `4000` train rows and `1000` test rows

### Task 3: Build the preprocessing and model pipeline

**Files:**
- Create: `D:\Class\CSAI\university\src\features.py`
- Create: `D:\Class\CSAI\university\tests\test_features.py`

- [ ] **Step 1: Write the failing tests for the preprocessing pipeline**

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, build_model_pipeline


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_build_model_pipeline_contains_preprocessor_and_classifier():
    pipeline = build_model_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]
```

- [ ] **Step 2: Run the feature tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL because `src.features` does not exist yet

- [ ] **Step 3: Write `src/features.py`**

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_model_pipeline(**classifier_params) -> Pipeline:
    classifier = DecisionTreeClassifier(random_state=RANDOM_STATE, **classifier_params)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
```

- [ ] **Step 4: Run the feature tests again**

Run: `pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Verification checkpoint on fit capability**

Run: `python -c "from src.config import DATA_PATH; from src.data_loader import load_dataset, prepare_train_test_split; from src.features import build_model_pipeline; df = load_dataset(DATA_PATH); X_train, X_test, y_train, y_test, _ = prepare_train_test_split(df); model = build_model_pipeline(); model.fit(X_train, y_train); print(type(model.named_steps['classifier']).__name__)"`
Expected: prints `DecisionTreeClassifier`

### Task 4: Implement evaluation helpers and save structured metrics

**Files:**
- Create: `D:\Class\CSAI\university\src\evaluate.py`
- Create: `D:\Class\CSAI\university\tests\test_evaluate.py`

- [ ] **Step 1: Write the failing tests for evaluation outputs**

```python
import pandas as pd

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
```

- [ ] **Step 2: Run the evaluation tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL because `src.evaluate` does not exist yet

- [ ] **Step 3: Write `src/evaluate.py`**

```python
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


def build_summary_row(model_name: str, accuracy: float, macro_f1: float, weighted_f1: float, train_accuracy: float) -> pd.Series:
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
```

- [ ] **Step 4: Run the evaluation tests again**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Verification checkpoint**

Run: `python -c "from src.evaluate import calculate_error_rate; print(calculate_error_rate(0.875))"`
Expected: `0.125`

### Task 5: Implement visualization helpers for matrices, trees, and feature importance

**Files:**
- Create: `D:\Class\CSAI\university\src\visualize.py`

- [ ] **Step 1: Write `src/visualize.py`**

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from src.config import LABEL_ORDER


def save_confusion_matrix_figure(matrix, output_path: Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_importance_figure(feature_names, importances, output_path: Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), x="importance", y="feature", orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_tree_figure(model_pipeline, feature_names, output_path: Path, title: str, max_depth: int = 3) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier = model_pipeline.named_steps["classifier"]
    plt.figure(figsize=(24, 12))
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=LABEL_ORDER,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
```

- [ ] **Step 2: Verification checkpoint**

Run: `python -c "import src.visualize; print('visualize imported')"`
Expected: prints `visualize imported`

### Task 6: Implement the training script and baseline experiment

**Files:**
- Create: `D:\Class\CSAI\university\src\train.py`

- [ ] **Step 1: Write `src/train.py` with the baseline workflow first**

```python
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import DATA_PATH, FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from src.data_loader import load_dataset, prepare_train_test_split
from src.evaluate import build_summary_row, evaluate_predictions, save_text_report
from src.features import build_model_pipeline
from src.visualize import save_confusion_matrix_figure, save_feature_importance_figure, save_tree_figure


def ensure_output_dirs() -> None:
    for path in (FIGURES_DIR, REPORTS_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_transformed_feature_names(model_pipeline, X_train) -> list[str]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def run_single_experiment(model_name: str, X_train, X_test, y_train, y_test, **classifier_params) -> dict:
    model = build_model_pipeline(**classifier_params)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = round(accuracy_score(y_train, train_predictions), 4)
    metrics = evaluate_predictions(y_test, test_predictions)
    metrics["train_accuracy"] = train_accuracy
    metrics["model_name"] = model_name
    metrics["pipeline"] = model

    summary_row = build_summary_row(
        model_name=model_name,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        train_accuracy=train_accuracy,
    )
    metrics["summary_row"] = summary_row
    return metrics


def save_experiment_artifacts(result: dict, X_train) -> None:
    model_name = result["model_name"]
    pipeline = result["pipeline"]
    feature_names = get_transformed_feature_names(pipeline, X_train)

    save_text_report(REPORTS_DIR / f"{model_name}_classification_report.txt", result["classification_report"])
    save_confusion_matrix_figure(
        result["confusion_matrix"],
        FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        f"{model_name} confusion matrix",
    )
    save_feature_importance_figure(
        feature_names,
        pipeline.named_steps["classifier"].feature_importances_,
        FIGURES_DIR / f"{model_name}_feature_importance.png",
        f"{model_name} feature importance",
    )
    save_tree_figure(
        pipeline,
        feature_names,
        FIGURES_DIR / f"{model_name}_tree.png",
        f"{model_name} decision tree",
    )


def main() -> None:
    ensure_output_dirs()
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, processed = prepare_train_test_split(df)

    class_distribution = processed["Grade_Label"].value_counts().sort_index()
    class_distribution.to_csv(TABLES_DIR / "class_distribution.csv", header=["count"])

    baseline_result = run_single_experiment("baseline", X_train, X_test, y_train, y_test)
    save_experiment_artifacts(baseline_result, X_train)

    summary_df = pd.DataFrame([baseline_result["summary_row"]])
    summary_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install dependencies into `.venv`**

Run: `.\.venv\Scripts\python -m pip install --upgrade pip`
Expected: `pip` upgrades inside `.venv`

Run: `.\.venv\Scripts\python -m pip install -r requirements.txt`
Expected: required packages install without touching the global Python environment

- [ ] **Step 3: Run the baseline training script**

Run: `.\.venv\Scripts\python -m src.train`
Expected: prints a one-row summary table for `baseline` and creates artifacts under `outputs/`

- [ ] **Step 4: Verification checkpoint**

Run: `Get-ChildItem -Path 'D:\Class\CSAI\university\outputs\figures','D:\Class\CSAI\university\outputs\reports','D:\Class\CSAI\university\outputs\tables'`
Expected: baseline report, confusion matrix, tree figure, feature-importance chart, and `model_comparison.csv`

### Task 7: Add improved decision-tree configurations and comparison output

**Files:**
- Modify: `D:\Class\CSAI\university\src\train.py`

- [ ] **Step 1: Extend the experiment list in `src/train.py`**

Replace the body of `main()` with this experiment loop:

```python
def main() -> None:
    ensure_output_dirs()
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, processed = prepare_train_test_split(df)

    class_distribution = processed["Grade_Label"].value_counts().sort_index()
    class_distribution.to_csv(TABLES_DIR / "class_distribution.csv", header=["count"])

    experiment_settings = [
        ("baseline", {}),
        ("max_depth_tuned", {"max_depth": 5}),
        ("min_samples_tuned", {"min_samples_split": 20, "min_samples_leaf": 10}),
        ("pruned_tree", {"ccp_alpha": 0.002}),
    ]

    results = []
    for model_name, classifier_params in experiment_settings:
        result = run_single_experiment(model_name, X_train, X_test, y_train, y_test, **classifier_params)
        save_experiment_artifacts(result, X_train)
        results.append(result["summary_row"])

    summary_df = pd.DataFrame(results).sort_values(by=["accuracy", "macro_f1"], ascending=False)
    summary_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    best_model_name = summary_df.iloc[0]["model"]
    save_text_report(REPORTS_DIR / "best_model.txt", f"Best model: {best_model_name}\n")
    print(summary_df.to_string(index=False))
```

- [ ] **Step 2: Run the full experiment suite**

Run: `.\.venv\Scripts\python -m src.train`
Expected: prints a four-row comparison table for `baseline`, `max_depth_tuned`, `min_samples_tuned`, and `pruned_tree`

- [ ] **Step 3: Verification checkpoint**

Run: `Import-Csv 'D:\Class\CSAI\university\outputs\tables\model_comparison.csv' | Format-Table`
Expected: comparison rows include `accuracy`, `error_rate`, `macro_f1`, `weighted_f1`, and `train_accuracy`

### Task 8: Add the companion notebook for interactive experimentation and presentation

**Files:**
- Create: `D:\Class\CSAI\university\notebooks\experiment.ipynb`
- Modify: `D:\Class\CSAI\university\README.md`

- [ ] **Step 1: Write `notebooks/experiment.ipynb` with these sections**

Notebook content outline:

```text
1. Title and project goal
2. Import statements and notebook path setup
3. Load dataset from ../data/Student_data.csv
4. Preview the raw data and dataset shape
5. Create and inspect grade-label distribution
6. Run the 80/20 stratified split
7. Train and evaluate the baseline tree
8. Train and evaluate the three improved trees
9. Display the comparison table from outputs/tables/model_comparison.csv
10. Display saved figures for confusion matrix, tree, and feature importance
11. Short markdown conclusion for presentation talking points
```

- [ ] **Step 2: Use `src/` functions inside the notebook rather than duplicating logic**

Core import cell content:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import DATA_PATH
from src.data_loader import load_dataset, prepare_train_test_split
from src.train import run_single_experiment
```

- [ ] **Step 3: Add a notebook cell to run the baseline and improved experiments**

Core experiment cell content:

```python
df = load_dataset(DATA_PATH)
X_train, X_test, y_train, y_test, processed = prepare_train_test_split(df)

experiment_settings = [
    ("baseline", {}),
    ("max_depth_tuned", {"max_depth": 5}),
    ("min_samples_tuned", {"min_samples_split": 20, "min_samples_leaf": 10}),
    ("pruned_tree", {"ccp_alpha": 0.002}),
]

results = []
for model_name, classifier_params in experiment_settings:
    result = run_single_experiment(model_name, X_train, X_test, y_train, y_test, **classifier_params)
    results.append(result["summary_row"])

comparison_df = pd.DataFrame(results).sort_values(by=["accuracy", "macro_f1"], ascending=False)
comparison_df
```

- [ ] **Step 4: Update `README.md` to mention the notebook**

Append this section:

```markdown
## Presentation notebook

Use `notebooks/experiment.ipynb` when you need:

- step-by-step screenshots for the report
- quick interactive reruns during analysis
- a cleaner flow for the presentation video
```

- [ ] **Step 5: Verification checkpoint**

Run: `.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/experiment.ipynb --output experiment.executed.ipynb`
Expected: notebook executes successfully and creates an executed copy

### Task 9: Add analysis-friendly outputs for the report and final project verification

**Files:**
- Modify: `D:\Class\CSAI\university\src\train.py`
- Modify: `D:\Class\CSAI\university\README.md`

- [ ] **Step 1: Append tree-depth and leaf-count information to the summary rows**

Update `run_single_experiment()` in `src/train.py`:

```python
    classifier = model.named_steps["classifier"]
    metrics["tree_depth"] = classifier.get_depth()
    metrics["leaf_count"] = classifier.get_n_leaves()

    summary_row = build_summary_row(
        model_name=model_name,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        train_accuracy=train_accuracy,
    )
    summary_row["tree_depth"] = metrics["tree_depth"]
    summary_row["leaf_count"] = metrics["leaf_count"]
```

- [ ] **Step 2: Save a short experiment notes file for report writing**

Add this block near the end of `main()` in `src/train.py`:

```python
    notes_lines = [
        "Experiment notes",
        "80/20 stratified train-test split",
        "Labels: A>=3.6, B>=3.2, C>=2.5, D>=2.0, F<2.0",
        "Improvements: max_depth tuning, min_samples tuning, cost-complexity pruning",
    ]
    save_text_report(REPORTS_DIR / "experiment_notes.txt", "\n".join(notes_lines) + "\n")
```

- [ ] **Step 3: Update `README.md` to mention the generated comparison file**

Append this section:

```markdown
## Key result files

- `outputs/tables/class_distribution.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/reports/best_model.txt`
- `outputs/reports/experiment_notes.txt`
```

- [ ] **Step 4: Run the final verification commands**

Run: `pytest tests -v`
Expected: all tests pass

Run: `.\.venv\Scripts\python -m src.train`
Expected: the full training run completes and rewrites outputs successfully

Run: `.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebooks/experiment.ipynb --output experiment.executed.ipynb`
Expected: notebook executes end to end without import-path errors

Run: `Get-Content 'D:\Class\CSAI\university\outputs\reports\best_model.txt'`
Expected: shows the chosen best model name

- [ ] **Step 5: Final checkpoint**

Run: `Get-ChildItem -Recurse -Path 'D:\Class\CSAI\university\outputs'`
Expected: output directories contain tables, report text files, and figure assets ready for the lab report
