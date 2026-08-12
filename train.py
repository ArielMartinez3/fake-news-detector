#!/usr/bin/env python3
"""Train the Fake News Detector pipeline.

End-to-end orchestrator that:
1. Downloads / loads the dataset
2. Preprocesses text with spaCy
3. Trains and benchmarks 4 classical ML models
4. Generates EDA and evaluation visualizations
5. Saves the best-performing model to disk

Usage::

    python train.py
    python train.py --force-download
"""

import argparse
import logging
import sys
import time

import nltk

from src.config import LOG_DATE_FORMAT, LOG_FORMAT
from src.data.loader import encode_labels, load_dataset, split_data
from src.features.preprocessor import TextPreprocessor
from src.models.trainer import save_best_model, train_and_evaluate
from src.visualization.plots import (
    plot_confusion_matrices,
    plot_label_distribution,
    plot_model_comparison,
    plot_ngrams,
    plot_wordcloud,
)

logger = logging.getLogger("fake_news_detector")


def _setup_logging() -> None:
    """Configure root logger with a clean, timestamped format."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _ensure_nltk_data() -> None:
    """Download required NLTK resources if not already present."""
    for resource in ("stopwords", "punkt", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the Fake News Detector pipeline."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if a local copy exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    _setup_logging()
    args = parse_args()
    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("FAKE NEWS DETECTOR — Training Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. NLTK resources
    # ------------------------------------------------------------------
    _ensure_nltk_data()

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    df = load_dataset(force_download=args.force_download)

    # ------------------------------------------------------------------
    # 3. EDA — class distribution
    # ------------------------------------------------------------------
    from src.config import RAW_LABEL_COL
    plot_label_distribution(df[RAW_LABEL_COL])

    # ------------------------------------------------------------------
    # 4. Encode labels
    # ------------------------------------------------------------------
    df = encode_labels(df)

    # ------------------------------------------------------------------
    # 5. Preprocess text
    # ------------------------------------------------------------------
    preprocessor = TextPreprocessor()
    df = preprocessor.transform(df)

    # ------------------------------------------------------------------
    # 6. EDA — word cloud & n-grams
    # ------------------------------------------------------------------
    from src.config import PROCESSED_TEXT_COL
    plot_wordcloud(df[PROCESSED_TEXT_COL])
    plot_ngrams(df[PROCESSED_TEXT_COL])

    # ------------------------------------------------------------------
    # 7. Train / test split
    # ------------------------------------------------------------------
    x_train, x_test, y_train, y_test = split_data(df)

    # ------------------------------------------------------------------
    # 8. Train & evaluate models
    # ------------------------------------------------------------------
    results_df, details = train_and_evaluate(x_train, x_test, y_train, y_test)

    logger.info("\n%s", results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 9. Evaluation visualizations
    # ------------------------------------------------------------------
    plot_confusion_matrices(details)
    plot_model_comparison(results_df)

    # ------------------------------------------------------------------
    # 10. Save best model
    # ------------------------------------------------------------------
    model_path = save_best_model(details, results_df)

    elapsed = time.perf_counter() - start
    logger.info("=" * 60)
    logger.info("Training complete in %.1f seconds.", elapsed)
    logger.info("Best model saved to: %s", model_path)
    logger.info("Figures saved to: outputs/figures/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
