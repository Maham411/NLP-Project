"""
corpus.py: the ONE shared corpus for Assignment 2 (Phase 2)

Every part of Assignment 2 (BoW, TF-IDF, N-grams, Word2Vec, fastText) must use
the same documents so the comparisons are fair. Load them with:

    from corpus import load_corpus, documents, tokenized
    df = load_corpus()
    docs = documents(df)        # list of strings, for scikit-learn
    sents = tokenized(df)       # list of token lists, for gensim (Part B)

Source: data/processed/cleaned_dataset.csv (made by the Phase 1 pipeline).
We do NOT clean the text again (the assignment says to keep results comparable).

Two small decisions are made here, once, for the whole team:
  1. Repeated tweet_ids are removed (keep the first copy). Phase 1 found 155
     repeats, some with different labels. Keeping them would count the same
     tweet twice and could leak between train and test sets in Phase 3.
  2. Tweets that became empty after cleaning are removed (they have no words,
     so they would be all zero rows in every representation).

Run from the repository root to see the summary:
    python src/corpus.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "processed" / "cleaned_dataset.csv"

TEXT_COL = "clean_text"
LABEL_COL = "airline_sentiment"

# clean_text is already tokenized by the Phase 1 pipeline (tokens joined by single
# spaces). This pattern tells scikit-learn to split on spaces ONLY, so it uses
# exactly our tokens. The default pattern r"(?u)\b\w\w+\b" would silently drop
# one-letter tokens like "u" and digits, and split tokens like "+2nd".
TOKEN_PATTERN = r"(?u)\S+"


def load_corpus(drop_duplicates: bool = True, drop_empty: bool = True) -> pd.DataFrame:
    """Return the cleaned tweets used by every part of Assignment 2."""
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"{CLEAN_FILE} not found. Run the Phase 1 pipeline first: python src/cleaning_pipeline.py"
        )
    # keep_default_na=False: an empty clean_text stays "" instead of becoming NaN
    df = pd.read_csv(CLEAN_FILE, keep_default_na=False)
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    if drop_duplicates:
        df = df.drop_duplicates(subset="tweet_id", keep="first")
    if drop_empty:
        df = df[df[TEXT_COL] != ""]
    return df.reset_index(drop=True)


def documents(df: pd.DataFrame) -> list[str]:
    """Documents as strings (input for CountVectorizer / TfidfVectorizer)."""
    return df[TEXT_COL].tolist()


def tokenized(df: pd.DataFrame) -> list[list[str]]:
    """Documents as lists of tokens (input for gensim Word2Vec / FastText)."""
    return [text.split() for text in df[TEXT_COL]]


def corpus_summary(df: pd.DataFrame) -> dict:
    raw = pd.read_csv(CLEAN_FILE, keep_default_na=False)
    tokens = [t for doc in tokenized(df) for t in doc]
    return {
        "rows_in_cleaned_file": int(len(raw)),
        "repeated_tweet_ids_removed": int(raw["tweet_id"].duplicated().sum()),
        "empty_after_cleaning_removed": int(len(raw) - raw["tweet_id"].duplicated().sum() - len(df)),
        "documents_used": int(len(df)),
        "total_tokens": len(tokens),
        "vocabulary_size": len(set(tokens)),
        "avg_tokens_per_doc": round(len(tokens) / len(df), 2),
        "labels": {k: int(v) for k, v in df[LABEL_COL].value_counts().items()},
    }


if __name__ == "__main__":
    data = load_corpus()
    for k, v in corpus_summary(data).items():
        print(f"{k:>30}: {v}")
