from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from sklearn.tree import plot_tree

from src.config import LABEL_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRADE_TREE_COLORMAP = LinearSegmentedColormap.from_list(
    "grade_tree",
    [
        (0.0, "#c62828"),
        (0.25, "#f57c00"),
        (0.5, "#ffd54f"),
        (0.75, "#9ccc65"),
        (1.0, "#2e7d32"),
    ],
)
GRADE_SCORE_MAP = {label: index for index, label in enumerate(reversed(LABEL_ORDER))}
COMPACT_FEATURE_LABELS = {
    "Age": "Age",
    "Attendance_Pct": "Attend%",
    "Study_Hours_Per_Day": "Study/day",
    "Previous_CGPA": "Prev CGPA",
    "Sleep_Hours": "Sleep",
    "Social_Hours_Week": "Social/wk",
}
ELLIPSIS_FACE_COLOR = "#9ca3af"
ELLIPSIS_EDGE_COLOR = "#111827"
NODE_TEXT_COLOR = "#1f2937"


def _blend_with_white(color, strength: float) -> tuple[float, float, float, float]:
    red, green, blue, _ = to_rgba(color)
    strength = max(0.0, min(strength, 1.0))
    return (
        1 - ((1 - red) * strength),
        1 - ((1 - green) * strength),
        1 - ((1 - blue) * strength),
        1.0,
    )


def _iter_rendered_node_ids(tree, node_id: int = 0, current_depth: int = 0, max_depth: int | None = None) -> list[int]:
    children_left = tree.children_left
    children_right = tree.children_right
    is_leaf = children_left[node_id] == children_right[node_id]

    if is_leaf:
        return [node_id]

    if max_depth is not None and current_depth == max_depth:
        return [node_id, children_left[node_id], children_right[node_id]]

    return [
        node_id,
        *_iter_rendered_node_ids(tree, children_left[node_id], current_depth + 1, max_depth),
        *_iter_rendered_node_ids(tree, children_right[node_id], current_depth + 1, max_depth),
    ]


def _get_node_annotations(artists) -> list:
    return [artist for artist in artists if getattr(artist, "get_bbox_patch", None) and artist.get_bbox_patch()]


def _compact_category_value(value: str) -> str:
    category_aliases = {
        "Engineering": "Eng",
        "Business": "Biz",
        "Science": "Sci",
        "Female": "F",
        "Male": "M",
    }
    return category_aliases.get(value, value)


def _format_feature_name(feature_name: str) -> str:
    cleaned = feature_name.replace("categorical__", "").replace("numeric__", "")
    if cleaned.startswith("Gender_"):
        return f"Gender={_compact_category_value(cleaned[len('Gender_'):])}"
    if cleaned.startswith("Major_"):
        return f"Major={_compact_category_value(cleaned[len('Major_'):])}"
    return COMPACT_FEATURE_LABELS.get(cleaned, cleaned.replace("_", " "))


def _format_split_rule(feature_name: str, threshold: float) -> str:
    compact_name = _format_feature_name(feature_name)
    if "=" in compact_name and abs(threshold - 0.5) < 0.11:
        return compact_name
    return f"{compact_name} <= {threshold:.2f}"


def _simplify_full_tree_labels(artists, classifier, feature_names) -> None:
    tree = classifier.tree_
    node_annotations = _get_node_annotations(artists)
    rendered_node_ids = _iter_rendered_node_ids(tree)

    for annotation, node_id in zip(node_annotations, rendered_node_ids):
        values = tree.value[node_id][0]
        sample_count = int(tree.n_node_samples[node_id])
        predicted_label = classifier.classes_[int(values.argmax())]
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

        if is_leaf:
            label = f"{predicted_label} | n={sample_count}"
        else:
            split_rule = _format_split_rule(feature_names[tree.feature[node_id]], tree.threshold[node_id])
            label = f"{split_rule}\n{predicted_label} | n={sample_count}"

        annotation.set_text(label)
        annotation.set_color(NODE_TEXT_COLOR)
        annotation.set_linespacing(1.0)


def _apply_grade_tree_palette(artists, classifier, max_depth: int | None = None) -> None:
    node_annotations = _get_node_annotations(artists)
    rendered_node_ids = _iter_rendered_node_ids(classifier.tree_, max_depth=max_depth)
    class_scores = [GRADE_SCORE_MAP.get(label, index) for index, label in enumerate(classifier.classes_)]
    max_score = max(GRADE_SCORE_MAP.values()) or 1

    for annotation, node_id in zip(node_annotations, rendered_node_ids):
        patch = annotation.get_bbox_patch()
        if "(...)" in annotation.get_text():
            patch.set_facecolor(ELLIPSIS_FACE_COLOR)
            patch.set_edgecolor(ELLIPSIS_EDGE_COLOR)
            patch.set_linewidth(0.8)
            annotation.set_color(NODE_TEXT_COLOR)
            continue

        values = classifier.tree_.value[node_id][0]
        total = float(values.sum())
        if total == 0:
            continue

        probabilities = values / total
        normalized_score = sum(probability * score for probability, score in zip(probabilities, class_scores)) / max_score
        purity = float(probabilities.max())
        base_color = GRADE_TREE_COLORMAP(normalized_score)
        fill_color = _blend_with_white(base_color, 0.35 + (0.65 * purity))
        edge_color = _blend_with_white(base_color, min(1.0, 0.55 + (0.55 * purity)))

        patch.set_facecolor(fill_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(0.8)
        annotation.set_color(NODE_TEXT_COLOR)


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
    _apply_grade_tree_palette(plt.gca().texts, classifier, max_depth=max_depth)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _get_full_tree_plot_style(leaf_count: int) -> dict[str, float]:
    if leaf_count > 500:
        return {
            "width_per_leaf": 0.36,
            "height_per_level": 1.4,
            "font_size": 4,
        }
    if leaf_count > 250:
        return {
            "width_per_leaf": 0.44,
            "height_per_level": 1.55,
            "font_size": 4,
        }
    if leaf_count > 120:
        return {
            "width_per_leaf": 0.54,
            "height_per_level": 1.75,
            "font_size": 5,
        }
    if leaf_count > 60:
        return {
            "width_per_leaf": 0.66,
            "height_per_level": 2.0,
            "font_size": 6,
        }
    return {
        "width_per_leaf": 0.92,
        "height_per_level": 2.3,
        "font_size": 7,
    }


def save_full_tree_svg(model_pipeline, feature_names, output_path: Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier = model_pipeline.named_steps["classifier"]
    style = _get_full_tree_plot_style(classifier.get_n_leaves())

    figure_width = max(26.0, classifier.get_n_leaves() * style["width_per_leaf"])
    figure_height = max(12.0, (classifier.get_depth() + 1) * style["height_per_level"])

    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=LABEL_ORDER,
        filled=True,
        rounded=True,
        fontsize=style["font_size"],
        precision=2,
        ax=axis,
    )
    _simplify_full_tree_labels(axis.texts, classifier, feature_names)
    _apply_grade_tree_palette(axis.texts, classifier)
    axis.set_title(title)
    figure.tight_layout(pad=1.0)
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
