# Decision Tree Student Performance Project

## Setup

1. Create the local environment:
   ```powershell
   python -m venv .venv
   ```
2. Activate it, then install dependencies:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Data

Place the dataset at `data/Student_data.csv`.

## Run

Train the model with:

```powershell
python -m src.train
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
