#!/usr/bin/env python3
"""Predict whether a news article is FAKE or REAL.

Loads the best saved model and runs inference on user-provided text,
either via CLI argument or interactive stdin.

Usage::

    python predict.py --text "Breaking news article text here..."
    python predict.py  # interactive mode
"""

import argparse
import logging
import sys

from src.config import LABEL_MAP_INV, LOG_DATE_FORMAT, LOG_FORMAT
from src.models.trainer import load_model

logger = logging.getLogger("fake_news_detector")


def _setup_logging() -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Predict FAKE / REAL for a news article."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Article text to classify. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a custom .joblib model file.",
    )
    return parser.parse_args()


def predict(text: str, model_path: str | None = None) -> dict[str, str | float]:
    """Run inference on a single text and return the prediction.

    Args:
        text: Raw article text.
        model_path: Optional path to a saved model; uses default if None.

    Returns:
        Dictionary with ``"label"`` (FAKE/REAL) and ``"confidence"`` info.
    """
    from pathlib import Path
    from src.config import MODELS_DIR

    path = Path(model_path) if model_path else MODELS_DIR / "best_model.joblib"
    pipeline = load_model(path)

    prediction = pipeline.predict([text])[0]
    label = LABEL_MAP_INV[prediction]

    # Try to get probability if the model supports it
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text])[0]
            confidence = float(max(proba))
        except AttributeError:
            pass
    elif hasattr(pipeline.named_steps.get("clf", None), "decision_function"):
        try:
            score = pipeline.decision_function([text])[0]
            confidence = abs(float(score))
        except Exception:
            pass

    return {"label": label, "prediction_int": int(prediction), "confidence": confidence}


def main() -> None:
    """Entry point for the prediction CLI."""
    _setup_logging()
    args = parse_args()

    if args.text:
        result = predict(args.text, args.model_path)
        _print_result(result)
    else:
        logger.info("Interactive mode — type an article and press Enter (Ctrl+C to exit).")
        try:
            while True:
                text = input("\n📰 Paste article text: ").strip()
                if not text:
                    continue
                result = predict(text, args.model_path)
                _print_result(result)
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting.")


def _print_result(result: dict) -> None:
    """Pretty-print a prediction result.

    Args:
        result: Dictionary from :func:`predict`.
    """
    tag = "[REAL]" if result["label"] == "REAL" else "[FAKE]"
    print(f"\n{tag}  Prediction: {result['label']}")
    if result["confidence"] is not None:
        print(f"   Confidence score: {result['confidence']:.4f}")
    print()


if __name__ == "__main__":
    main()
