"""Multi-model training, evaluation, and serialization.

Benchmarks four classical ML models on TF-IDF features and persists
the best-performing pipeline (vectorizer + classifier) to disk.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    LABEL_MAP_INV,
    MODELS_DIR,
    NGRAM_RANGE,
    PASSIVE_AGGRESSIVE_MAX_ITER,
    RANDOM_STATE,
    SGD_MAX_ITER,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
)

logger = logging.getLogger(__name__)


def _build_pipelines() -> dict[str, Pipeline]:
    """Create named pipelines for each candidate model.

    Each pipeline bundles a TF-IDF vectorizer with a classifier so the
    entire feature-extraction → prediction flow can be serialized as a
    single artifact.

    Returns:
        Dictionary mapping model name to its sklearn ``Pipeline``.
    """
    tfidf_params: dict = dict(
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
    )

    pipelines: dict[str, Pipeline] = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(
                max_iter=1_000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),
        "SGD Classifier": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", SGDClassifier(
                max_iter=SGD_MAX_ITER,
                random_state=RANDOM_STATE,
                loss="hinge",
            )),
        ]),
        "Passive Aggressive": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", SGDClassifier(
                loss="hinge",
                penalty=None,
                learning_rate="constant",
                eta0=1.0,
                max_iter=PASSIVE_AGGRESSIVE_MAX_ITER,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return pipelines


def train_and_evaluate(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Train all candidate models and return a comparison table.

    Args:
        x_train: Preprocessed training texts.
        x_test: Preprocessed test texts.
        y_train: Encoded training labels.
        y_test: Encoded test labels.

    Returns:
        A tuple of:
        - ``results_df``: DataFrame with Accuracy, F1 (macro), Precision,
          and Recall for each model, sorted by F1 descending.
        - ``details``: Dictionary keyed by model name containing
          ``"pipeline"``, ``"y_pred"``, ``"confusion_matrix"``, and
          ``"classification_report"`` for downstream use (plots, etc.).
    """
    pipelines = _build_pipelines()
    records: list[dict] = []
    details: dict[str, dict] = {}

    for name, pipeline in pipelines.items():
        logger.info("Training: %s ...", name)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(
            y_test, y_pred,
            target_names=list(LABEL_MAP_INV.values()),
            output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred)

        records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1 (macro)": round(f1, 4),
            "Precision (macro)": round(report["macro avg"]["precision"], 4),
            "Recall (macro)": round(report["macro avg"]["recall"], 4),
        })

        details[name] = {
            "pipeline": pipeline,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        logger.info(
            "%s — Accuracy: %.4f | F1: %.4f", name, acc, f1
        )

    results_df = (
        pd.DataFrame(records)
        .sort_values("F1 (macro)", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, details


def save_best_model(
    details: dict[str, dict],
    results_df: pd.DataFrame,
    output_dir: Path = MODELS_DIR,
) -> Path:
    """Serialize the best-performing pipeline to disk.

    Args:
        details: Output from :func:`train_and_evaluate`.
        results_df: Benchmark results DataFrame (sorted by F1 desc).
        output_dir: Directory where the ``.joblib`` file is saved.

    Returns:
        Path to the saved model file.
    """
    best_name: str = results_df.iloc[0]["Model"]
    best_pipeline: Pipeline = details[best_name]["pipeline"]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info(
        "Best model saved -- %s (F1: %.4f) -> %s",
        best_name,
        results_df.iloc[0]["F1 (macro)"],
        model_path,
    )
    return model_path


def load_model(model_path: Path = MODELS_DIR / "best_model.joblib") -> Pipeline:
    """Load a previously saved model pipeline from disk.

    Args:
        model_path: Path to the ``.joblib`` file.

    Returns:
        The deserialized sklearn ``Pipeline``.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train.py first."
        )
    pipeline: Pipeline = joblib.load(model_path)
    logger.info("Model loaded from: %s", model_path)
    return pipeline
