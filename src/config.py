"""Centralized configuration for the Fake News Detector pipeline.

All hyperparameters, paths, and constants are defined here to avoid
hardcoded values scattered across the codebase.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

DATASET_URL: str = (
    "https://raw.githubusercontent.com/nacho1907/Bases/"
    "refs/heads/main/fake_or_real_news.csv"
)
LOCAL_DATASET_PATH: Path = DATA_DIR / "fake_or_real_news.csv"

# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
LABEL_MAP: dict[str, int] = {"FAKE": 0, "REAL": 1}
LABEL_MAP_INV: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Columns as they come from the raw CSV
RAW_TEXT_COL: str = "text"
RAW_LABEL_COL: str = "label"
PROCESSED_TEXT_COL: str = "processed_text"

# ---------------------------------------------------------------------------
# NLP / Feature Engineering
# ---------------------------------------------------------------------------
SPACY_MODEL: str = "en_core_web_sm"
TFIDF_MAX_DF: float = 0.90
TFIDF_MIN_DF: int = 2
TFIDF_MAX_FEATURES: int | None = 50_000
NGRAM_RANGE: tuple[int, int] = (1, 2)

# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
PASSIVE_AGGRESSIVE_MAX_ITER: int = 50
SGD_MAX_ITER: int = 1_000

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
TOP_K_NGRAMS: int = 15
WORDCLOUD_WIDTH: int = 1_000
WORDCLOUD_HEIGHT: int = 500
FIGURE_DPI: int = 150

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
