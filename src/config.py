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
