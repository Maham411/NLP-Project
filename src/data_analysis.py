"""
data_analysis.py: Task 2: Exploration & basic statistics
Member 1: Maham Iqbal (L1F23BSCS0411)
Dataset: Twitter US Airline Sentiment (Kaggle, crowdflower)

Computes on the RAW (uncleaned) tweets:
  1. Basic info (rows, columns, missing values, duplicates)
  2. Number of documents and average document length (words + characters)
  3. Target distribution (negative / neutral / positive) + label confidence
  4. Extra context: tweets per airline, sentiment by airline, negative reasons
  5. Vocabulary size of the whole corpus
  6. Tokens vs types on a random sample (+ vocabulary growth curve)
  7. Zipf's law and the long tail
  8. Real examples of messy text

Run from the repository root:
    python src/data_analysis.py

Outputs:
    report/figures/*.png       graphs for the report
    report/stats.json          all numbers (Members 4 & 5 reuse the "before cleaning" vocab)
    report/messy_examples.txt  real messy tweets found by regex
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import LABEL_COL, ROOT, TEXT_COL, load_raw  # noqa: E402

REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
SEED = 42            # fixed seed -> the same sample every run (reproducibility)
SAMPLE_SIZE = 1000   # number of tweets in the tokens-vs-types sample

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_COLORS = {"negative": "#C44E52", "neutral": "#8C8C8C", "positive": "#55A868"}

# ---------------------------------------------------------------------------
# Baseline tokenizer used for ALL "raw" counts in this file.
# Lowercase, then take runs of letters/digits/underscore.
# (So "@united" -> "united", "#fail" -> "fail", "can't" -> "can", "t".)
# Member 4 must count "vocabulary BEFORE cleaning" with this same function
# so the before/after comparison is fair:  from data_analysis import simple_tokenize
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\w+")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase + regex word tokenizer (baseline, no cleaning)."""
    return TOKEN_RE.findall(str(text).lower())


def _save(fig, save_path: Path | None):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 1. Basic info
# ---------------------------------------------------------------------------
def basic_info(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": {c: int(v) for c, v in df.isnull().sum().items()},
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_tweet_ids": int(df["tweet_id"].duplicated().sum()),
        "n_duplicate_texts": int(df[TEXT_COL].duplicated().sum()),
        "n_unique_texts": int(df[TEXT_COL].nunique()),
    }


# ---------------------------------------------------------------------------
# 2. Document length
# ---------------------------------------------------------------------------
def doc_lengths(df: pd.DataFrame) -> pd.Series:
    """Length of each tweet in words (split on whitespace)."""
    return df[TEXT_COL].astype(str).str.split().str.len()


def char_lengths(df: pd.DataFrame) -> pd.Series:
    """Length of each tweet in characters."""
    return df[TEXT_COL].astype(str).str.len()


def length_stats(lengths: pd.Series, unit: str = "words") -> dict:
    return {
        f"mean_{unit}": round(float(lengths.mean()), 2),
        f"median_{unit}": float(lengths.median()),
        f"std_{unit}": round(float(lengths.std()), 2),
        f"min_{unit}": int(lengths.min()),
        f"max_{unit}": int(lengths.max()),
    }


def plot_length_hist(words: pd.Series, chars: pd.Series, save_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, s, unit in [(axes[0], words, "words"), (axes[1], chars, "characters")]:
        ax.hist(s, bins=40, color="#4C72B0", edgecolor="white")
        ax.axvline(s.mean(), color="#C44E52", linestyle="--", label=f"Mean = {s.mean():.1f}")
        ax.axvline(s.median(), color="#55A868", label=f"Median = {s.median():.0f}")
        ax.set_title(f"Tweet length ({unit})")
        ax.set_xlabel(unit.capitalize())
        ax.set_ylabel("Number of tweets")
        ax.legend()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Target distribution + label confidence
# ---------------------------------------------------------------------------
def target_distribution(df: pd.DataFrame) -> dict:
    counts = df[LABEL_COL].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    pct = (counts / counts.sum() * 100).round(2)
    return {label: {"count": int(counts[label]), "percent": float(pct[label])} for label in counts.index}


def plot_target(df: pd.DataFrame, save_path: Path | None = None):
    counts = df[LABEL_COL].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(counts.index, counts.values, color=[LABEL_COLORS[l] for l in counts.index])
    pct = counts / counts.sum() * 100
    ax.bar_label(bars, labels=[f"{v:,}\n({p:.1f}%)" for v, p in zip(counts.values, pct)])
    ax.set_title("Target distribution (airline_sentiment)")
    ax.set_ylabel("Number of tweets")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, save_path)


def confidence_stats(df: pd.DataFrame) -> dict:
    """How sure the crowd annotators were about each label (1.0 = full agreement)."""
    c = df["airline_sentiment_confidence"]
    return {
        "mean_confidence": round(float(c.mean()), 4),
        "percent_full_agreement_1.0": round(float((c == 1.0).mean() * 100), 2),
        "percent_below_0.5": round(float((c < 0.5).mean() * 100), 2),
        "percent_below_0.6": round(float((c < 0.6).mean() * 100), 2),
    }


# ---------------------------------------------------------------------------
# 4. Extra context: airlines & negative reasons
# ---------------------------------------------------------------------------
def sentiment_by_airline(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["airline"], df[LABEL_COL]).reindex(columns=LABEL_ORDER, fill_value=0)


def plot_sentiment_by_airline(df: pd.DataFrame, save_path: Path | None = None):
    tab = sentiment_by_airline(df)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    left = np.zeros(len(tab))
    for label in LABEL_ORDER:
        ax.barh(tab.index, tab[label], left=left, color=LABEL_COLORS[label], label=label)
        left += tab[label].values
    ax.set_title("Tweets per airline, split by sentiment")
    ax.set_xlabel("Number of tweets")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    return _save(fig, save_path)


def negative_reasons(df: pd.DataFrame) -> pd.Series:
    return df["negativereason"].value_counts()


def plot_negative_reasons(df: pd.DataFrame, save_path: Path | None = None):
    reasons = negative_reasons(df).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(reasons.index, reasons.values, color="#C44E52")
    ax.set_title("Reasons given for negative tweets")
    ax.set_xlabel("Number of tweets")
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Vocabulary (whole corpus)
# ---------------------------------------------------------------------------
def word_counts(texts) -> Counter:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    return counter


def vocab_stats(counter: Counter) -> dict:
    n_tokens = sum(counter.values())
    n_types = len(counter)
    hapax = sum(1 for c in counter.values() if c == 1)       # words seen exactly once
    top100 = sum(c for _, c in counter.most_common(100))
    return {
        "n_tokens": int(n_tokens),
        "vocabulary_size_types": int(n_types),
        "type_token_ratio": round(n_types / n_tokens, 4),
        "hapax_legomena": int(hapax),
        "hapax_percent_of_vocab": round(hapax / n_types * 100, 2),
        "top100_words_percent_of_tokens": round(top100 / n_tokens * 100, 2),
    }


# ---------------------------------------------------------------------------
# 6. Tokens vs types on a sample
# ---------------------------------------------------------------------------
def sample_tokens_types(df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = SEED) -> dict:
    sample = df[TEXT_COL].sample(n=min(n, len(df)), random_state=seed)
    tokens = [tok for text in sample for tok in simple_tokenize(text)]
    return {
        "sample_size_tweets": int(len(sample)),
        "random_state": seed,
        "tokens": len(tokens),
        "types": len(set(tokens)),
        "type_token_ratio": round(len(set(tokens)) / len(tokens), 4),
    }


def type_growth(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """How the number of types grows as we read more tokens (Heaps' law)."""
    shuffled = df[TEXT_COL].sample(frac=1, random_state=seed)
    seen, rows, n_tokens = set(), [], 0
    for i, text in enumerate(shuffled, start=1):
        toks = simple_tokenize(text)
        n_tokens += len(toks)
        seen.update(toks)
        if i % 100 == 0 or i == len(shuffled):
            rows.append((n_tokens, len(seen)))
    return pd.DataFrame(rows, columns=["tokens", "types"])


def plot_type_growth(growth: pd.DataFrame, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(growth["tokens"], growth["types"], color="#4C72B0")
    ax.set_title("Vocabulary growth: types vs tokens")
    ax.set_xlabel("Tokens read")
    ax.set_ylabel("Distinct types")
    ax.grid(alpha=0.3)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Zipf's law and the long tail
# ---------------------------------------------------------------------------
def plot_zipf(counter: Counter, save_path: Path | None = None):
    freqs = np.array(sorted(counter.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, freqs, marker=".", linestyle="none", markersize=2, color="#4C72B0", label="Observed")
    ax.loglog(ranks, freqs[0] / ranks, linestyle="--", color="#C44E52", label="Ideal Zipf: f = f1 / rank")
    ax.set_title("Zipf's law: rank vs frequency (log-log)")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return _save(fig, save_path)


def plot_top_words(counter: Counter, n: int = 20, save_path: Path | None = None):
    words, counts = zip(*counter.most_common(n))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(words[::-1], counts[::-1], color="#4C72B0")
    ax.set_title(f"Top {n} most frequent tokens (raw text)")
    ax.set_xlabel("Frequency")
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Messy text examples
# ---------------------------------------------------------------------------
MESSY_PATTERNS = {
    "@mentions": r"@\w+",
    "#hashtags": r"#\w+",
    "URLs": r"https?://\S+|www\.\S+",
    "HTML entities (&amp; &lt; &gt;)": r"&(?:[a-zA-Z]+|#\d+);",
    "Emojis": r"[\U0001F300-\U0001FAFF☀-➿]",
    "Non-ASCII characters (emojis, accents, encoding issues)": r"[^\x00-\x7F]",
    "ALL-CAPS words (4+ letters)": r"\b[A-Z]{4,}\b",
    "Repeated punctuation (!!, ??, ...)": r"[!?]{2,}|\.{3,}",
    "Elongated words (sooo, pleeease)": r"\b[a-zA-Z]*([a-zA-Z])\1{2,}[a-zA-Z]*\b",
    "Missing space after comma": r"[a-z]{2,},[a-z]{2,}",
    "Flight numbers": r"(?i)\b(?:flight|flt)\s*#?\s*\d{1,4}\b",
    "Times (5:30, 10pm)": r"(?i)\b\d{1,2}(?::\d{2})\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b",
}


def find_messy_examples(df: pd.DataFrame, patterns: dict = MESSY_PATTERNS) -> pd.DataFrame:
    """For each pattern: how many tweets contain it + one real example (row index + full tweet)."""
    texts = df[TEXT_COL].astype(str)
    rows = []
    for name, pattern in patterns.items():
        rx = re.compile(pattern)
        mask = texts.apply(lambda t: rx.search(t) is not None)
        n_match = int(mask.sum())
        example_idx, example, matched = None, "", ""
        if n_match:
            example_idx = int(mask.idxmax())          # first matching row (deterministic)
            example = texts.at[example_idx]
            matched = rx.search(example).group(0)
        rows.append({
            "issue": name,
            "tweets_affected": n_match,
            "percent_of_tweets": round(n_match / len(df) * 100, 2),
            "example_row": example_idx,
            "matched_text": matched,
            "example": example,                       # tweets are short, so show the whole thing
        })
    out = pd.DataFrame(rows)
    out["example_row"] = out["example_row"].astype("Int64")  # whole numbers, <NA> if no match
    return out


def save_messy_examples(messy: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("MESSY TEXT EXAMPLES: Twitter US Airline Sentiment (raw tweets)\n" + "=" * 64 + "\n\n")
        for _, r in messy.iterrows():
            f.write(f"{r['issue']}\n")
            f.write(f"  Tweets affected: {r['tweets_affected']:,} ({r['percent_of_tweets']}%)\n")
            if not pd.isna(r["example_row"]):
                row = int(r["example_row"])
                f.write(f"  Example: row {row} (label: {df.at[row, LABEL_COL]})\n")
                f.write(f"  Matched: {r['matched_text']!r}\n")
                f.write(f"  Tweet:   {r['example']!r}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def print_dict(d: dict, width: int = 28) -> None:
    for k, v in d.items():
        print(f"{k:>{width}}: {v:,}" if isinstance(v, int) else f"{k:>{width}}: {v}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    stats = {}

    print_header("1. BASIC INFO")
    stats["basic_info"] = basic_info(df)
    print_dict({k: v for k, v in stats["basic_info"].items() if k not in ("columns", "missing_values")})
    print("\nColumns:", ", ".join(stats["basic_info"]["columns"]))
    print("\nMissing values per column (non-zero only):")
    for col, n in stats["basic_info"]["missing_values"].items():
        if n:
            print(f"  {col:<32} {n:>6,} ({n / len(df) * 100:.1f}%)")

    print_header("2. DOCUMENT LENGTH")
    words, chars = doc_lengths(df), char_lengths(df)
    stats["length"] = {**length_stats(words, "words"), **length_stats(chars, "chars")}
    print("Number of documents (tweets):", f"{len(df):,}")
    print_dict(stats["length"])
    plt.close(plot_length_hist(words, chars, FIG_DIR / "length_distribution.png"))

    print_header("3. TARGET DISTRIBUTION")
    stats["target"] = target_distribution(df)
    for label, v in stats["target"].items():
        print(f"{label:>12}: {v['count']:>6,} ({v['percent']}%)")
    plt.close(plot_target(df, FIG_DIR / "target_distribution.png"))
    stats["label_confidence"] = confidence_stats(df)
    print("\nLabel confidence (annotator agreement):")
    print_dict(stats["label_confidence"])

    print_header("4. AIRLINES & NEGATIVE REASONS")
    tab = sentiment_by_airline(df)
    print(tab.assign(total=tab.sum(axis=1)).sort_values("total", ascending=False).to_string())
    stats["sentiment_by_airline"] = {a: {k: int(v) for k, v in row.items()} for a, row in tab.iterrows()}
    plt.close(plot_sentiment_by_airline(df, FIG_DIR / "sentiment_by_airline.png"))
    reasons = negative_reasons(df)
    print("\n" + reasons.to_string())
    stats["negative_reasons"] = {k: int(v) for k, v in reasons.items()}
    plt.close(plot_negative_reasons(df, FIG_DIR / "negative_reasons.png"))

    print_header("5. VOCABULARY (whole corpus, raw text)")
    counter = word_counts(df[TEXT_COL])
    stats["vocabulary"] = vocab_stats(counter)
    print_dict(stats["vocabulary"], 32)

    print_header(f"6. TOKENS vs TYPES (random sample of {SAMPLE_SIZE} tweets)")
    stats["sample"] = sample_tokens_types(df)
    print_dict(stats["sample"])
    plt.close(plot_type_growth(type_growth(df), FIG_DIR / "type_growth.png"))

    print_header("7. ZIPF'S LAW / LONG TAIL")
    top20 = counter.most_common(20)
    stats["top20_words"] = [[w, int(c)] for w, c in top20]
    for rank, (w, c) in enumerate(top20, start=1):
        print(f"{rank:>3}. {w:<14} {c:>8,}")
    plt.close(plot_zipf(counter, FIG_DIR / "zipf_plot.png"))
    plt.close(plot_top_words(counter, 20, FIG_DIR / "top20_words.png"))

    print_header("8. MESSY TEXT")
    messy = find_messy_examples(df)
    print(messy[["issue", "tweets_affected", "percent_of_tweets", "example_row"]].to_string(index=False))
    stats["messy_text"] = [
        {
            "issue": r["issue"],
            "tweets_affected": int(r["tweets_affected"]),
            "percent_of_tweets": float(r["percent_of_tweets"]),
            "example_row": None if pd.isna(r["example_row"]) else int(r["example_row"]),
        }
        for _, r in messy.iterrows()
    ]
    save_messy_examples(messy, df, REPORT_DIR / "messy_examples.txt")

    with open(REPORT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print_header("DONE")
    print(f"Figures   -> {FIG_DIR}")
    print(f"Stats     -> {REPORT_DIR / 'stats.json'}")
    print(f"Examples  -> {REPORT_DIR / 'messy_examples.txt'}")


if __name__ == "__main__":
    main()