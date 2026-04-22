from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features import build_model_pipeline, build_preprocessor


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_build_model_pipeline_contains_preprocessor_and_classifier():
    pipeline = build_model_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]
