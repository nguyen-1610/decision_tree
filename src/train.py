import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import DATA_PATH, FIGURES_DIR, REPORTS_DIR, TABLES_DIR
from src.data_loader import load_dataset, prepare_train_test_split
from src.evaluate import build_summary_row, evaluate_predictions, save_text_report
from src.features import build_model_pipeline
from src.visualize import save_confusion_matrix_figure, save_feature_importance_figure, save_full_tree_svg, save_tree_figure


def ensure_output_dirs() -> None:
    for path in (FIGURES_DIR, REPORTS_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_transformed_feature_names(model_pipeline) -> list[str]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def run_single_experiment(model_name: str, X_train, X_test, y_train, y_test, **classifier_params) -> dict:
    model = build_model_pipeline(**classifier_params)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = round(accuracy_score(y_train, train_predictions), 4)
    metrics = evaluate_predictions(y_test, test_predictions)
    classifier = model.named_steps["classifier"]
    metrics["train_accuracy"] = train_accuracy
    metrics["tree_depth"] = classifier.get_depth()
    metrics["leaf_count"] = classifier.get_n_leaves()
    metrics["model_name"] = model_name
    metrics["pipeline"] = model

    summary_row = build_summary_row(
        model_name=model_name,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        train_accuracy=train_accuracy,
    )
    summary_row["tree_depth"] = metrics["tree_depth"]
    summary_row["leaf_count"] = metrics["leaf_count"]
    metrics["summary_row"] = summary_row
    return metrics


def save_experiment_artifacts(result: dict) -> None:
    model_name = result["model_name"]
    pipeline = result["pipeline"]
    feature_names = get_transformed_feature_names(pipeline)

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
    save_full_tree_svg(
        pipeline,
        feature_names,
        FIGURES_DIR / f"{model_name}_tree_full.svg",
        f"{model_name} decision tree (full)",
    )


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
        save_experiment_artifacts(result)
        results.append(result["summary_row"])

    summary_df = pd.DataFrame(results).sort_values(by=["accuracy", "macro_f1"], ascending=False)
    summary_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    best_model_name = summary_df.iloc[0]["model"]
    save_text_report(REPORTS_DIR / "best_model.txt", f"Best model: {best_model_name}\n")
    notes_lines = [
        "Experiment notes",
        "80/20 stratified train-test split",
        "Labels: A>=3.6, B>=3.2, C>=2.5, D>=2.0, F<2.0",
        "Improvements: max_depth tuning, min_samples tuning, cost-complexity pruning",
    ]
    save_text_report(REPORTS_DIR / "experiment_notes.txt", "\n".join(notes_lines) + "\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
