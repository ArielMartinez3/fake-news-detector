"""Dataset loading, validation, and train/test splitting.

Handles downloading the Fake-or-Real news CSV from a remote URL (with
local caching), basic sanity checks, and stratified train/test splitting.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_URL,
    LABEL_MAP,
    LOCAL_DATASET_PATH,
    RANDOM_STATE,
    RAW_LABEL_COL,
    RAW_TEXT_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def load_dataset(
    url: str = DATASET_URL,
    local_path: Path = LOCAL_DATASET_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load the Fake-or-Real news dataset, downloading if needed.

    If a local copy exists and *force_download* is ``False``, the cached
    file is used.  Otherwise the CSV is fetched from *url* and persisted
    to *local_path* for future runs.

    Args:
        url: Remote URL of the CSV file.
        local_path: Where to cache the CSV locally.
        force_download: Re-download even if the local copy exists.

    Returns:
        A ``DataFrame`` with at least columns ``text`` and ``label``.

    Raises:
        ValueError: If the downloaded data does not contain expected columns.
    """
    if local_path.exists() and not force_download:
        logger.info("Loading dataset from local cache: %s", local_path)
        df = pd.read_csv(local_path)
    else:
        logger.info("Downloading dataset from: %s", url)
        df = pd.read_csv(url)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        logger.info("Dataset cached to: %s", local_path)

    # Drop the unnamed index column that comes with this specific CSV
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    _validate_schema(df)
    logger.info(
        "Dataset loaded — %d rows, %d columns, label distribution:\n%s",
        len(df),
        len(df.columns),
        df[RAW_LABEL_COL].value_counts().to_string(),
    )
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Ensure the DataFrame contains the expected columns.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {RAW_TEXT_COL, RAW_LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map string labels (FAKE / REAL) to integers (0 / 1).

    Args:
        df: DataFrame with a ``label`` column containing string labels.

    Returns:
        A copy of the DataFrame with the ``label`` column encoded as int.
    """
    df = df.copy()
    df[RAW_LABEL_COL] = df[RAW_LABEL_COL].map(LABEL_MAP)
    unmapped = df[RAW_LABEL_COL].isna().sum()
    if unmapped > 0:
        logger.warning(
            "%d rows had unmapped labels and will be dropped.", unmapped
        )
        df = df.dropna(subset=[RAW_LABEL_COL])
    df[RAW_LABEL_COL] = df[RAW_LABEL_COL].astype(int)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split on text and encoded labels.

    Args:
        df: DataFrame with ``text``, ``label`` (int-encoded), and
            ``processed_text`` columns.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test)`` where X values
        are the processed text series and y values are int-encoded labels.
    """
    from src.config import PROCESSED_TEXT_COL

    x_train, x_test, y_train, y_test = train_test_split(
        df[PROCESSED_TEXT_COL],
        df[RAW_LABEL_COL],
        test_size=test_size,
        random_state=random_state,
        stratify=df[RAW_LABEL_COL],
    )
    logger.info(
        "Train/test split — train: %d, test: %d (test_size=%.2f)",
        len(x_train),
        len(x_test),
        test_size,
    )
    return x_train, x_test, y_train, y_test
