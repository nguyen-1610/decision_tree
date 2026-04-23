# Decision Tree Student Performance Project

## Setup

Using a virtual environment is optional. The project runs as long as the active Python interpreter has the packages from `requirements.txt`.

### Option A: Use a virtual environment

1. Create the local environment:
   ```powershell
   python -m venv .venv
   ```
2. Activate it, then install dependencies:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

### Option B: Install packages without `venv`

From the project root, install the dependencies into your current Python environment:

```powershell
python -m pip install --user -r requirements.txt
```

## Data

Place the dataset at `data/Student_data.csv`.

## Run

Open a terminal in the `decision_tree` folder, then train the model with:

```powershell
python -m src.train
```

Do not use `python src\train.py`. This project imports modules with the `src.` package prefix, so it should be started with `python -m src.train` from the project root.

## VS Code

In VS Code, open the `decision_tree` folder and run the commands above in the integrated terminal. Make sure the selected Python interpreter is the one where you installed the requirements.

## Tests

Run the test suite with:

```powershell
python -m pytest -q
```

## Outputs

Generated artifacts are written to:

- `outputs/figures`
- `outputs/reports`
- `outputs/tables`

## Notebooks

Open `notebooks/experiment.ipynb` from `jupyter notebook` for the presentation notebook.

## Presentation notebook

Use `notebooks/experiment.ipynb` when you need:

- step-by-step screenshots for the report
- quick interactive reruns during analysis
- a cleaner flow for the presentation video

## Key result files

- `outputs/tables/class_distribution.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/reports/best_model.txt`
- `outputs/reports/experiment_notes.txt`
