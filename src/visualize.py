from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

from src.config import LABEL_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
