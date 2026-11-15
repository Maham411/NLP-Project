"""
count_representations.py: Assignment 2, Part A (A1 to A3): count based representations
Member 1: Maham Iqbal (L1F23BSCS0411)

  A1. Bag of Words  : document term count matrix (CountVectorizer), shape and sparsity
  A2. TF-IDF        : TF-IDF matrix over the same corpus, words with HIGH weight and
                      common words pushed toward ZERO, with a real example
  A3. N-grams       : unigrams, bigrams and trigrams, how the number of features grows
                      with n and what that does to sparsity

All parts use the same cleaned corpus from src/corpus.py.

Run from the repository root:
    python src/count_representations.py

Outputs:
    report/part_a/a1_bow_stats.json
    report/part_a/a2_idf_lowest.csv, a2_idf_highest.csv, a2_example_tweet.csv
    report/part_a/a3_ngram_growth.csv, a3_top_bigrams.csv, a3_top_trigrams.csv
    report/part_a/part_a_summary.json
    report/figures/part_a_top_words.png
    report/figures/part_a_idf.png
    report/figures/part_a_ngram_growth.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import LABEL_COL, ROOT, TEXT_COL, TOKEN_PATTERN, documents, load_corpus  # noqa: E402

OUT_DIR = ROOT / "report" / "part_a"
FIG_DIR = ROOT / "report" / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sparsity_stats(X) -> dict:
    """Shape, number of non zero cells and sparsity (% of cells that are zero)."""
    n_docs, n_features = X.shape
    total = n_docs * n_features
    nnz = int(X.nnz)
    sparse_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    return {
        "n_documents": int(n_docs),
        "n_features": int(n_features),
        "total_cells": int(total),
        "non_zero_cells": nnz,
        "zero_cells": int(total - nnz),
        "sparsity_percent": round((1 - nnz / total) * 100, 4),
        "avg_non_zero_per_doc": round(nnz / n_docs, 2),
        "sparse_matrix_MB": round(sparse_bytes / 1e6, 2),
        "dense_matrix_MB_if_float64": round(total * 8 / 1e6, 1),
    }


def _save(fig, path: Path | None):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# A1. Bag of Words
# ---------------------------------------------------------------------------
def build_bow(docs: list[str], ngram_range=(1, 1), min_df: int = 1):
    """Document term COUNT matrix. Rows = tweets, columns = vocabulary words,
    cell = how many times the word appears in the tweet."""
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False,
                          ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(docs)          # scipy sparse CSR matrix
    return vec, X


def term_frequencies(vec, X) -> pd.DataFrame:
    """Total count and document frequency (number of tweets containing it) per term."""
    total = np.asarray(X.sum(axis=0)).ravel()
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    return (pd.DataFrame({"term": vec.get_feature_names_out(), "total_count": total, "doc_freq": doc_freq})
            .sort_values("total_count", ascending=False).reset_index(drop=True))


def show_bow_row(vec, X, docs: list[str], row: int) -> pd.DataFrame:
    """The non zero cells of one BoW row, so you can see what a row looks like."""
    r = X.getrow(row)
    terms = vec.get_feature_names_out()
    return pd.DataFrame({"term": terms[r.indices], "count": r.data}).sort_values("count", ascending=False)


def plot_top_words(freq: pd.DataFrame, n: int = 20, path: Path | None = None):
    top = freq.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["term"], top["total_count"], color="#4C72B0")
    ax.set_title(f"A1. Top {n} words in the Bag of Words matrix")
    ax.set_xlabel("Total count in the corpus")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# A2. TF-IDF
# ---------------------------------------------------------------------------
def build_tfidf(docs: list[str], ngram_range=(1, 1), min_df: int = 1):
    """TF-IDF with scikit-learn defaults (written out so we can explain each one):
         tf  = raw count of the term in the tweet
         idf = ln((1 + n) / (1 + df)) + 1     (smooth_idf=True)
         each row is then divided by its length so it has length 1 (norm="l2")."""
    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=ngram_range,
                          min_df=min_df, smooth_idf=True, sublinear_tf=False, norm="l2")
    X = vec.fit_transform(docs)
    return vec, X


def idf_table(tfidf_vec, bow_freq: pd.DataFrame) -> pd.DataFrame:
    """IDF of every term, joined with its document frequency from the BoW matrix."""
    idf = pd.DataFrame({"term": tfidf_vec.get_feature_names_out(), "idf": tfidf_vec.idf_})
    return idf.merge(bow_freq[["term", "doc_freq", "total_count"]], on="term").sort_values("idf")


def tfidf_example(tfidf_vec, X_tfidf, bow_vec, X_bow, idf: pd.DataFrame, row: int) -> pd.DataFrame:
    """Every word of one tweet with its count, document frequency, idf and final weight."""
    terms = tfidf_vec.get_feature_names_out()
    r = X_tfidf.getrow(row)
    out = pd.DataFrame({"term": terms[r.indices], "tfidf_weight": r.data.round(4)})
    counts = dict(zip(bow_vec.get_feature_names_out()[X_bow.getrow(row).indices], X_bow.getrow(row).data))
    out["count_in_tweet"] = out["term"].map(counts)
    out = out.merge(idf[["term", "doc_freq", "idf"]], on="term")
    out["idf"] = out["idf"].round(4)
    return out[["term", "count_in_tweet", "doc_freq", "idf", "tfidf_weight"]].sort_values("tfidf_weight", ascending=False)


def pick_example_row(df: pd.DataFrame, idf: pd.DataFrame, common_n: int = 15) -> int:
    """Pick a real tweet that contains BOTH very common words and rare words,
    so the example shows high and low weights side by side. Deterministic
    (always the same tweet for the same data)."""
    common = set(idf.head(common_n)["term"])                    # lowest idf = most common
    rare = set(idf[idf["doc_freq"] <= 5]["term"])
    for i, text in enumerate(df[TEXT_COL]):
        toks = text.split()
        if 7 <= len(toks) <= 12 and len(set(toks)) == len(toks):
            n_common = len(set(toks) & common)
            n_rare = len(set(toks) & rare)
            if n_common >= 3 and n_rare >= 1:
                return i
    return 0


def mean_tfidf(tfidf_vec, X_tfidf, bow_freq: pd.DataFrame) -> pd.DataFrame:
    """Average TF-IDF weight of each word over the tweets that contain it."""
    sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
    out = pd.DataFrame({"term": tfidf_vec.get_feature_names_out(), "sum_weight": sums})
    out = out.merge(bow_freq[["term", "doc_freq"]], on="term")
    out["mean_weight_when_present"] = (out["sum_weight"] / out["doc_freq"]).round(4)
    return out


def rare_word_sample(idf: pd.DataFrame, n: int = 12, seed: int = 42) -> pd.DataFrame:
    """A fixed random sample of real words (letters only, 5+ letters) that appear in
    only one tweet. Thousands of words share the top IDF, so we show a sample of
    readable ones instead of numbers like '600' that just happen to sort first."""
    pool = idf[(idf["doc_freq"] == 1) & idf["term"].str.fullmatch(r"[a-z]{5,}")]
    return pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("term")


def plot_idf(idf: pd.DataFrame, n: int = 12, path: Path | None = None):
    low = idf.head(n)
    high = rare_word_sample(idf, n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    axes[0].barh(low["term"][::-1], low["idf"][::-1], color="#C44E52")
    axes[0].set_title(f"Lowest IDF: {n} most common words")
    axes[1].barh(high["term"], high["idf"], color="#55A868")
    axes[1].set_title(f"Highest IDF: words in only 1 tweet")
    for ax in axes:
        ax.set_xlabel("IDF")
    fig.suptitle("A2. Inverse document frequency (common words get low weight)")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# A3. N-grams
# ---------------------------------------------------------------------------
NGRAM_SETTINGS = [
    ("unigrams (1,1)", (1, 1)),
    ("bigrams only (2,2)", (2, 2)),
    ("trigrams only (3,3)", (3, 3)),
    ("uni + bi (1,2)", (1, 2)),
    ("uni + bi + tri (1,3)", (1, 3)),
]


def ngram_growth(docs: list[str], settings=NGRAM_SETTINGS) -> pd.DataFrame:
    """For each n-gram setting: feature count, sparsity and how many features
    appear in only one tweet. Also the feature count if we keep only features
    seen in at least 2 tweets (min_df=2), to show the usual way of controlling growth."""
    rows = []
    for name, rng in settings:
        vec, X = build_bow(docs, ngram_range=rng)
        s = sparsity_stats(X)
        doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
        _, X2 = build_bow(docs, ngram_range=rng, min_df=2)
        rows.append({
            "setting": name,
            "ngram_range": str(rng),
            "features": s["n_features"],
            "non_zero_cells": s["non_zero_cells"],
            "sparsity_percent": s["sparsity_percent"],
            "avg_non_zero_per_doc": s["avg_non_zero_per_doc"],
            "features_in_only_1_tweet": int((doc_freq == 1).sum()),
            "percent_in_only_1_tweet": round((doc_freq == 1).mean() * 100, 2),
            "features_with_min_df_2": int(X2.shape[1]),
            "sparsity_percent_min_df_2": sparsity_stats(X2)["sparsity_percent"],
        })
    return pd.DataFrame(rows)


def top_ngrams(docs: list[str], n: int, k: int = 20) -> pd.DataFrame:
    vec, X = build_bow(docs, ngram_range=(n, n))
    return term_frequencies(vec, X).head(k)


def plot_ngram_growth(growth: pd.DataFrame, path: Path | None = None):
    g = growth[growth["setting"].str.startswith(("unigrams", "uni +"))].copy()
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(g))
    w = 0.38
    b1 = ax1.bar(x - w / 2, g["features"], w, color="#4C72B0", label="all features")
    b2 = ax1.bar(x + w / 2, g["features_with_min_df_2"], w, color="#8EA9DB", label="features with min_df = 2")
    ax1.bar_label(b1, labels=[f"{v:,}" for v in g["features"]], fontsize=8)
    ax1.bar_label(b2, labels=[f"{v:,}" for v in g["features_with_min_df_2"]], fontsize=8)
    ax1.set_xticks(x, ["n = 1\n(1,1)", "n up to 2\n(1,2)", "n up to 3\n(1,3)"])
    ax1.set_ylabel("Number of features (columns)")
    ax1.set_ylim(0, g["features"].max() * 1.18)
    ax2 = ax1.twinx()
    ax2.plot(x, g["sparsity_percent"], color="#C44E52", marker="o", label="sparsity % (all features)")
    for xi, v in zip(x, g["sparsity_percent"]):
        ax2.annotate(f"{v:.3f}%", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#C44E52")
    ax2.set_ylabel("Sparsity (% zero cells)")
    lo = g["sparsity_percent"].min()
    ax2.set_ylim(lo - (100 - lo) * 0.6, 100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)
    ax1.set_title("A3. Feature count and sparsity as n grows")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def header(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_corpus()
    docs = documents(df)
    summary = {"documents": len(docs)}
    print(f"Corpus: {len(docs):,} cleaned tweets (src/corpus.py)")

    # ---------------- A1 ----------------
    header("A1. BAG OF WORDS")
    bow_vec, X_bow = build_bow(docs)
    a1 = sparsity_stats(X_bow)
    for k, v in a1.items():
        print(f"{k:>28}: {v:,}" if isinstance(v, int) else f"{k:>28}: {v}")
    freq = term_frequencies(bow_vec, X_bow)
    print("\nTop 10 words:", ", ".join(f"{t} ({c})" for t, c in freq.head(10)[["term", "total_count"]].values))
    print(f"\nExample row 1: {docs[1]!r}")
    print(show_bow_row(bow_vec, X_bow, docs, 1).to_string(index=False))
    summary["A1_bow"] = a1
    with open(OUT_DIR / "a1_bow_stats.json", "w") as f:
        json.dump(a1, f, indent=2)
    plt.close(plot_top_words(freq, 20, FIG_DIR / "part_a_top_words.png"))

    # ---------------- A2 ----------------
    header("A2. TF-IDF")
    tfidf_vec, X_tfidf = build_tfidf(docs)
    a2 = sparsity_stats(X_tfidf)
    print(f"Shape: {X_tfidf.shape}  (same vocabulary as BoW)   sparsity: {a2['sparsity_percent']}%")
    idf = idf_table(tfidf_vec, freq)
    print("\nLowest IDF (most common words, pushed toward zero):")
    print(idf.head(10)[["term", "doc_freq", "idf"]].round(4).to_string(index=False))
    print("\nHighest IDF (rare words, get the biggest boost), a sample of real words:")
    print(rare_word_sample(idf, 8)[["term", "doc_freq", "idf"]].round(4).to_string(index=False))
    n_max = int((idf["idf"] == idf["idf"].max()).sum())
    print(f"({n_max:,} words share the maximum IDF because each appears in only one tweet)")
    idf.head(20).round(4).to_csv(OUT_DIR / "a2_idf_lowest.csv", index=False)
    idf.tail(20).round(4).to_csv(OUT_DIR / "a2_idf_highest.csv", index=False)

    row = pick_example_row(df, idf)
    example = tfidf_example(tfidf_vec, X_tfidf, bow_vec, X_bow, idf, row)
    print(f"\nExample tweet (row {row}, tweet_id {df.at[row, 'tweet_id']}, label {df.at[row, LABEL_COL]}):")
    print(f"  raw  : {df.at[row, 'text']!r}")
    print(f"  clean: {docs[row]!r}")
    print(example.to_string(index=False))
    example.to_csv(OUT_DIR / "a2_example_tweet.csv", index=False)

    mt = mean_tfidf(tfidf_vec, X_tfidf, freq)
    flight_w = mt.loc[mt["term"] == "flight", "mean_weight_when_present"]
    summary["A2_tfidf"] = {
        **a2,
        "idf_min": round(float(idf["idf"].min()), 4),
        "idf_max": round(float(idf["idf"].max()), 4),
        "words_with_max_idf": n_max,
        "example_row": row,
        "example_tweet_id": int(df.at[row, "tweet_id"]),
        "mean_weight_of_flight_when_present": float(flight_w.iloc[0]) if len(flight_w) else None,
    }
    plt.close(plot_idf(idf, 12, FIG_DIR / "part_a_idf.png"))

    # ---------------- A3 ----------------
    header("A3. N-GRAMS")
    growth = ngram_growth(docs)
    print(growth.drop(columns=["ngram_range"]).to_string(index=False))
    growth.to_csv(OUT_DIR / "a3_ngram_growth.csv", index=False)
    bi, tri = top_ngrams(docs, 2), top_ngrams(docs, 3)
    print("\nTop 10 bigrams :", ", ".join(f"{t} ({c})" for t, c in bi.head(10)[["term", "total_count"]].values))
    print("Top 10 trigrams:", ", ".join(f"{t} ({c})" for t, c in tri.head(10)[["term", "total_count"]].values))
    bi.to_csv(OUT_DIR / "a3_top_bigrams.csv", index=False)
    tri.to_csv(OUT_DIR / "a3_top_trigrams.csv", index=False)
    tf13, X13 = build_tfidf(docs, ngram_range=(1, 3))
    print(f"\nTF-IDF with (1,3) has the same shape as BoW (1,3): {X13.shape}")
    summary["A3_ngrams"] = growth.drop(columns=["ngram_range"]).to_dict("records")
    plt.close(plot_ngram_growth(growth, FIG_DIR / "part_a_ngram_growth.png"))

    with open(OUT_DIR / "part_a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    header("DONE")
    print(f"Tables  -> {OUT_DIR}")
    print(f"Figures -> {FIG_DIR / 'part_a_*.png'}")


if __name__ == "__main__":
    main()
