"""NLP text preprocessing pipeline.

Provides spaCy-based text cleaning: lowercasing, lemmatization,
stopword removal, and non-alphabetic token filtering.  The heavy
resources (spaCy model, stopword set) are loaded once and reused
across all calls for performance.
"""

import logging

import pandas as pd
import spacy
from nltk.corpus import stopwords as nltk_stopwords

from src.config import PROCESSED_TEXT_COL, RAW_TEXT_COL, SPACY_MODEL

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Stateful text preprocessor that loads NLP resources once.

    Attributes:
        nlp: Loaded spaCy language model.
        stop_words: Cached set of English stopwords.
    """

    def __init__(
        self,
        spacy_model: str = SPACY_MODEL,
        language: str = "english",
    ) -> None:
        """Initialize the preprocessor with NLP resources.

        Args:
            spacy_model: Name of the spaCy model to load.
            language: Language for NLTK stopwords.
        """
        logger.info("Loading spaCy model: %s", spacy_model)
        self.nlp: spacy.Language = spacy.load(
            spacy_model, disable=["parser", "ner"]
        )
        self.stop_words: set[str] = set(nltk_stopwords.words(language))
        logger.info(
            "Preprocessor ready — stopwords: %d, spaCy model: %s",
            len(self.stop_words),
            spacy_model,
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize a single text document.

        Applies lowercasing, spaCy lemmatization, stopword removal,
        and filters non-alphabetic tokens.

        Args:
            text: Raw article text.

        Returns:
            Space-separated string of cleaned, lemmatized tokens.
        """
        doc = self.nlp(str(text).lower())
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha and token.text not in self.stop_words
        ]
        return " ".join(tokens)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to the entire DataFrame.

        Adds a ``processed_text`` column with cleaned versions of the
        raw ``text`` column.

        Args:
            df: DataFrame with a ``text`` column.

        Returns:
            The same DataFrame with an additional ``processed_text`` column.
        """
        logger.info("Preprocessing %d documents...", len(df))
        df = df.copy()
        df[PROCESSED_TEXT_COL] = df[RAW_TEXT_COL].apply(self.clean_text)

        empty_count = (df[PROCESSED_TEXT_COL].str.strip() == "").sum()
        if empty_count > 0:
            logger.warning(
                "%d documents resulted in empty text after preprocessing.",
                empty_count,
            )

        logger.info("Preprocessing complete.")
        return df
