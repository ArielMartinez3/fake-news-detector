"""Visualization utilities for EDA and model evaluation.

Generates word clouds, n-gram frequency charts, confusion matrices,
and model comparison bar charts.  All figures are saved to the
configured output directory.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from src.config import (
    FIGURE_DPI,
    FIGURES_DIR,
    LABEL_MAP_INV,
    TOP_K_NGRAMS,
    WORDCLOUD_HEIGHT,
    WORDCLOUD_WIDTH,
)

logger = logging.getLogger(__name__)

# Use a clean, modern style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "font.size": 11,
})


def _save_figure(fig: plt.Figure, filename: str, output_dir: Path = FIGURES_DIR) -> Path:
    """Persist a matplotlib figure to disk.

    Args:
        fig: The figure to save.
        filename: Target filename (e.g. ``"wordcloud.png"``).
        output_dir: Directory where the file is written.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Figure saved: %s", path)
    return path


def plot_label_distribution(labels: pd.Series) -> Path:
    """Bar chart of the FAKE / REAL class distribution.

    Args:
        labels: Series of string labels (``"FAKE"`` / ``"REAL"``).

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = labels.value_counts()
    colors = ["#e74c3c", "#2ecc71"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title("Class Distribution: FAKE vs REAL", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Articles")
    for i, (label, count) in enumerate(counts.items()):
        ax.text(i, count + 30, str(count), ha="center", fontweight="bold", fontsize=12)
    return _save_figure(fig, "label_distribution.png")


def plot_wordcloud(texts: pd.Series) -> Path:
    """Generate and save a word cloud from preprocessed texts.

    Args:
        texts: Series of preprocessed (cleaned) text documents.

    Returns:
        Path to the saved figure.
    """
    combined = " ".join(texts.dropna())
    wc = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color="white",
        colormap="viridis",
        max_words=200,
    ).generate(combined)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Preprocessed Corpus", fontsize=14, fontweight="bold")
    return _save_figure(fig, "wordcloud.png")


def _get_top_ngrams(
    corpus: pd.Series,
    n: int,
    top_k: int = TOP_K_NGRAMS,
) -> pd.DataFrame:
    """Extract the top-k most frequent n-grams from a text corpus.

    Args:
        corpus: Series of preprocessed text documents.
        n: The *n* in n-gram (1=unigram, 2=bigram, etc.).
        top_k: Number of top n-grams to return.

    Returns:
        DataFrame with columns ``"N-gram"`` and ``"Frequency"``,
        sorted descending by frequency.
    """
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        stop_words="english",
        max_features=top_k * 10,
    )
    matrix = vectorizer.fit_transform(corpus.dropna())
    counts = np.asarray(matrix.sum(axis=0)).flatten()
    features = vectorizer.get_feature_names_out()
    freq = sorted(zip(features, counts), key=lambda x: x[1], reverse=True)[:top_k]
    return pd.DataFrame(freq, columns=["N-gram", "Frequency"])


def plot_ngrams(
    corpus: pd.Series,
    ngram_sizes: list[int] | None = None,
) -> list[Path]:
    """Generate horizontal bar charts for multiple n-gram sizes.

    Args:
        corpus: Series of preprocessed text documents.
        ngram_sizes: List of n-gram sizes to plot (default: 1–4).

    Returns:
        List of paths to saved figures.
    """
    if ngram_sizes is None:
        ngram_sizes = [1, 2, 3]

    names = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams", 4: "4-grams"}
    paths: list[Path] = []

    for n in ngram_sizes:
        ngram_df = _get_top_ngrams(corpus, n)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="Frequency", y="N-gram", data=ngram_df,
            hue="N-gram", palette="viridis", dodge=False, ax=ax, legend=False,
        )
        title = f"Top {TOP_K_NGRAMS} {names.get(n, f'{n}-grams')}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("")
        paths.append(_save_figure(fig, f"ngrams_{n}.png"))

    return paths


def plot_confusion_matrices(
    details: dict[str, dict],
) -> list[Path]:
    """Plot confusion matrices for all trained models in a grid.

    Args:
        details: Output from ``trainer.train_and_evaluate`` containing
            ``"confusion_matrix"`` for each model.

    Returns:
        List of paths to saved figures.
    """
    model_names = list(details.keys())
    n_models = len(model_names)
    cols = 2
    rows = (n_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    labels = list(LABEL_MAP_INV.values())

    paths: list[Path] = []
    for idx, name in enumerate(model_names):
        cm = details[name]["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=axes[idx], cbar=False,
        )
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confusion Matrices — Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    paths.append(_save_figure(fig, "confusion_matrices.png"))
    return paths


def plot_model_comparison(results_df: pd.DataFrame) -> Path:
    """Grouped bar chart comparing model metrics.

    Args:
        results_df: DataFrame from ``trainer.train_and_evaluate`` with
            columns ``Model``, ``Accuracy``, ``F1 (macro)``, etc.

    Returns:
        Path to the saved figure.
    """
    melted = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x="Model", y="Score", hue="Metric", data=melted,
        palette="viridis", ax=ax,
    )
    ax.set_title("Model Benchmark Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, "model_comparison.png")
