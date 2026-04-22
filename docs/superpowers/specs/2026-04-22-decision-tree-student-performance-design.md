# Decision Tree Classification Design for Student Performance Dataset

## 1. Goal

Build a reproducible decision tree classification project for the selected Kaggle dataset, using student attributes and habits to predict academic performance labels derived from `Final_CGPA`.

The project must satisfy the lab requirements in `Lab 3 - Decision Tree.pdf`:

- select and describe a public dataset
- build a baseline decision tree
- evaluate the model with classification metrics
- present and interpret the resulting tree
- implement 2-3 improvement methods
- compare baseline and improved models

## 2. Dataset Scope

Selected dataset:

- Source: Kaggle, "University Student Performance & Habits Dataset"
- URL: <https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset>
- Records: 5,000
- Raw columns: 10

Expected columns from the dataset:

- `Student_ID`
- `Gender`
- `Age`
- `Major`
- `Attendance_Pct`
- `Study_Hours_Per_Day`
- `Previous_CGPA`
- `Sleep_Hours`
- `Social_Hours_Week`
- `Final_CGPA`

Rationale for suitability:

- the dataset is tabular, clean, and small enough for fast experimentation
- it contains a mix of categorical and numeric predictors, which works well for decision trees after straightforward preprocessing
- `Final_CGPA` can be converted into interpretable academic-performance classes
- the features are easy to explain in the report and presentation

## 3. Problem Formulation

This project will use classification rather than regression.

Target labels will be created from `Final_CGPA` using the approved academic-style grading scale:

- `A`: `Final_CGPA >= 3.6`
- `B`: `3.2 <= Final_CGPA < 3.6`
- `C`: `2.5 <= Final_CGPA < 3.2`
- `D`: `2.0 <= Final_CGPA < 2.5`
- `F`: `Final_CGPA < 2.0`

Design choice:

- the raw `Final_CGPA` column is used only to derive the label
- after label creation, raw `Final_CGPA` must be excluded from the feature matrix to avoid target leakage
- `Student_ID` must also be excluded because it is an identifier, not a predictive signal

Final feature set:

- categorical features: `Gender`, `Major`
- numeric features: `Age`, `Attendance_Pct`, `Study_Hours_Per_Day`, `Previous_CGPA`, `Sleep_Hours`, `Social_Hours_Week`

## 4. Project Structure

The implementation should create and use the following structure inside the workspace:

```text
university/
├─ .venv/
├─ data/
│  └─ Student_data.csv
├─ notebooks/
│  └─ experiment.ipynb
├─ outputs/
│  ├─ figures/
│  ├─ reports/
│  └─ tables/
├─ src/
│  ├─ data_loader.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ visualize.py
│  └─ utils.py
├─ README.md
└─ requirements.txt
```

Notes:

- `.venv` must be created locally in the workspace, not through global installation
- the dataset file should be placed under `data/`
- the code should be organized enough that rerunning the full experiment is easy
- the notebook should reuse logic from `src/` rather than duplicating the full workflow in notebook cells

## 5. Data Flow

The end-to-end data flow should be:

1. load the CSV dataset from `data/`
2. validate expected columns
3. derive grade labels from `Final_CGPA`
4. drop `Student_ID` and raw `Final_CGPA` from features
5. split data into train and test sets with an 80/20 ratio using stratification on the grade label
6. preprocess categorical columns with one-hot encoding
7. pass numeric columns through without scaling
8. train the model inside a single `Pipeline`
9. evaluate baseline and improved models on the same held-out test set
10. save metrics, plots, and comparison artifacts to `outputs/`

Reasoning:

- keeping preprocessing and the classifier in one pipeline reduces leakage and makes the project reproducible
- tree-based models do not require numeric scaling, so the preprocessing stays simple

## 6. Baseline Model

Baseline model:

- `DecisionTreeClassifier` from scikit-learn
- a fixed `random_state` for reproducibility
- start with a simple, documented baseline configuration rather than an aggressive tuned model

Baseline evaluation outputs:

- class distribution table
- train/test split summary
- confusion matrix
- classification report with precision, recall, and F1-score
- overall accuracy
- error rate computed as `1 - accuracy`
- optional macro and weighted F1 if useful for class imbalance discussion

Baseline tree analysis outputs:

- decision tree figure
- tree depth and number of leaves
- feature importance ranking
- short explanation of the most important decision nodes and branches

Overfitting diagnosis:

- compare training accuracy and testing accuracy
- examine whether the tree becomes deep relative to the dataset simplicity
- comment on whether some classes are poorly separated or rarely predicted

## 7. Improvement Methods

The project will implement three improvement strategies so the group can compare at least 2-3 alternatives as required.

### Method 1: Limit Tree Depth

Change:

- tune `max_depth`

Purpose:

- reduce overfitting
- improve generalization on the test set
- produce a smaller, more interpretable tree

Expected discussion:

- shallow trees may underfit
- moderate depth may improve test accuracy and stability

### Method 2: Constrain Splits and Leaves

Change:

- tune `min_samples_split`
- tune `min_samples_leaf`

Purpose:

- prevent brittle rules built from very small subgroups
- smooth the decision boundaries
- reduce sensitivity to noise in the training data

Expected discussion:

- stricter split constraints may improve minority-class precision or recall
- too much regularization may reduce accuracy if the tree becomes too coarse

### Method 3: Cost-Complexity Pruning

Change:

- tune `ccp_alpha`

Purpose:

- prune back branches that add complexity without enough predictive benefit
- improve test performance and readability

Expected discussion:

- pruning can reduce overfitting and simplify the final tree
- too much pruning may collapse useful decision structure

## 8. Model Selection Strategy

The implementation should compare candidate settings in a disciplined way.

Recommended approach:

- keep one train/test split for final evaluation
- use `test_size=0.2` and `random_state` fixed for reproducibility
- tune parameters only on the training portion using cross-validation, or use a validation split inside training
- use a primary selection metric that balances class performance, such as macro F1, while still reporting accuracy and error rate for the lab

Reasoning:

- plain accuracy can hide weak performance on smaller classes
- macro F1 better reflects whether the classifier handles all grade categories reasonably well

## 9. Error Handling and Validation

The code should fail clearly when:

- the dataset file is missing
- expected columns are not present
- the grade-label transformation produces empty classes
- the user runs training before installing dependencies in `.venv`

Minimum validation behavior:

- print a clear message about the missing file path or missing columns
- stop execution rather than continue with silent assumptions

## 10. Outputs for Report and Presentation

The code should generate reusable artifacts for the written report and presentation video:

- class distribution table
- baseline metrics table
- improvement metrics table
- comparison table across all models
- confusion matrix image for the selected best model
- decision tree visualization image for the baseline and, if readable, the best model
- feature-importance chart

These artifacts should be saved under `outputs/` with stable filenames so they can be inserted into the report.

Notebook deliverable:

- provide `notebooks/experiment.ipynb` for interactive exploration, screenshot capture, and presentation support
- the notebook should show the main workflow step by step: data preview, label creation, train/test split, baseline model, improved models, comparison table, and generated figures
- the notebook should call functions from `src/` where practical so the project has one main source of truth

## 11. Testing and Verification

Before considering the implementation complete, verify that:

- the `.venv` is created inside the workspace
- dependencies install into `.venv`, not globally
- the training script runs end-to-end on the chosen dataset
- the outputs are written successfully
- metrics are reproducible across reruns with the same random seed

Minimum manual verification:

- run the main training entry point
- inspect the generated metrics summary
- confirm the confusion matrix and tree figure files exist

## 12. Assumptions

- the user will provide the dataset CSV manually in the `data/` directory
- the dataset is available locally at `data/Student_data.csv`
- the selected Kaggle dataset columns match the public metadata
- the report itself is not generated in this phase; this phase only prepares a clean, runnable project and artifacts for the report
- because the current workspace is not a git repository, the design spec cannot be committed unless version control is initialized later

## 13. Non-Goals

This project will not include:

- global Python package installation
- non-tree models as the main experimental focus
- automated report writing
- deployment, web UI, or database components
- a second independent implementation of the same workflow inside the notebook

## 14. Recommended Implementation Outcome

After implementation, the workspace should allow a classmate to:

1. create or activate `.venv`
2. install `requirements.txt`
3. place the CSV file into `data/`
4. run one command to train baseline and improved models
5. open `notebooks/experiment.ipynb` to inspect the workflow interactively
6. inspect `outputs/` for tables, metrics, and figures

This outcome is sufficient to support the code, report, and presentation parts of the lab assignment.
