#!/usr/bin/env python3
"""Gradio interactive demo for the Fake News Detector.

Launches a web UI where users can paste a news article and get
an instant FAKE / REAL prediction with confidence scores and
the most influential TF-IDF features.

Usage::

    python app/app.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
import numpy as np

from src.config import LABEL_MAP_INV, LOG_DATE_FORMAT, LOG_FORMAT
from src.models.trainer import load_model

logger = logging.getLogger("fake_news_detector.app")


def _setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def classify_article(text: str) -> tuple[dict[str, float], str]:
    """Classify a news article and return label probabilities + top features.

    Args:
        text: Raw article text pasted by the user.

    Returns:
        Tuple of (label_confidences dict, top_features markdown string).
    """
    if not text or not text.strip():
        return {}, "⚠️ Please paste a news article to classify."

    pipeline = load_model()
    prediction = pipeline.predict([text])[0]
    label = LABEL_MAP_INV[prediction]

    # Build confidence dict for Gradio Label component
    confidences: dict[str, float] = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text])[0]
            for idx, prob in enumerate(proba):
                confidences[LABEL_MAP_INV[idx]] = float(prob)
        except AttributeError:
            confidences[label] = 1.0
    else:
        confidences[label] = 1.0
        other_label = "REAL" if label == "FAKE" else "FAKE"
        confidences[other_label] = 0.0

    # Extract top TF-IDF features
    top_features_md = _get_top_features(pipeline, text)

    return confidences, top_features_md


def _get_top_features(pipeline, text: str, top_k: int = 10) -> str:
    """Extract the most influential TF-IDF features for a prediction.

    Args:
        pipeline: Trained sklearn Pipeline with a TfidfVectorizer step.
        text: Input text.
        top_k: Number of top features to return.

    Returns:
        Markdown-formatted string listing top features.
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        tfidf_vector = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        # Get non-zero features and their weights
        non_zero = tfidf_vector.nonzero()
        weights = []
        for col_idx in non_zero[1]:
            weights.append((feature_names[col_idx], tfidf_vector[0, col_idx]))

        weights.sort(key=lambda x: x[1], reverse=True)
        top = weights[:top_k]

        if not top:
            return "No significant features found."

        lines = ["### 🔍 Top Influential Words", ""]
        lines.append("| Rank | Word | TF-IDF Weight |")
        lines.append("|------|------|---------------|")
        for i, (word, weight) in enumerate(top, 1):
            lines.append(f"| {i} | `{word}` | {weight:.4f} |")

        return "\n".join(lines)
    except Exception as e:
        logger.warning("Could not extract features: %s", e)
        return "Feature extraction not available for this model."


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns:
        Configured Gradio Blocks instance.
    """
    with gr.Blocks(
        title="🔎 Fake News Detector",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🔎 Fake News Detector
            **Paste a news article below** to classify it as **FAKE** or **REAL**
            using a TF-IDF + Machine Learning pipeline.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="📰 News Article",
                    placeholder="Paste the full text of a news article here...",
                    lines=12,
                )
                classify_btn = gr.Button(
                    "🚀 Classify", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="Prediction", num_top_classes=2
                )
                features_output = gr.Markdown(label="Top Features")

        classify_btn.click(
            fn=classify_article,
            inputs=text_input,
            outputs=[label_output, features_output],
        )

        gr.Markdown(
            """
            ---
            *Built with scikit-learn, spaCy, and Gradio.*
            *Model trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).*
            """
        )

    return app


def main() -> None:
    """Launch the Gradio app."""
    _setup_logging()
    logger.info("Starting Gradio app...")
    app = build_app()
    app.launch(share=False)


if __name__ == "__main__":
    main()
