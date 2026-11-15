"""
hand_worked_a4.py: Assignment 2, Part A4: hand worked TF-IDF weight and bigram
Member 1: Maham Iqbal (L1F23BSCS0411)

We take a TINY toy corpus of 4 real tweets from our cleaned data, compute one
TF-IDF weight and one bigram step by step, and check that the library gives
the same numbers. The same arithmetic is done on paper (CamScanner copy).

We use the method taught in class (Lecture 7 and the TF-IDF Excel example):
    TF(t, d)     = count of t in d / number of words in d
    DF(t)        = number of documents that contain t
    IDF(t)       = log10(N / DF(t))            (no smoothing, base 10)
    TF-IDF(t, d) = TF(t, d) x IDF(t)           (no normalisation)
    P(next | previous) = count(pair) / count(previous)

For comparison we also show scikit-learn's DEFAULT TfidfVectorizer formula
(smoothed natural log idf + 1, then L2 normalisation), which A2 uses.

Run from the repository root:
    python src/hand_worked_a4.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import TEXT_COL, TOKEN_PATTERN, load_corpus  # noqa: E402

# Four short real tweets (picked by tweet_id so the choice never changes).
# They share the words "flight" and "book" so the IDF values are different.
TOY_TWEET_IDS = [
    570038941497192448,   # D1  "book flight hawaii"
    569842553966571520,   # D2  "learn book flight"
    569209566715469824,   # D3  "good flight attendant"
    570065037231927296,   # D4  "flight voucher"
]
TFIDF_TERM, TFIDF_DOC = "hawaii", 0          # TF-IDF of "hawaii" in D1
BIGRAM, BIGRAM_FIRST = "book flight", "book"  # P(flight | book)
DOC_NAMES = ["D1", "D2", "D3", "D4"]


def toy_corpus():
    df = load_corpus()
    rows = df.set_index("tweet_id").loc[TOY_TWEET_IDS]
    return rows["text"].tolist(), rows[TEXT_COL].tolist()


# ---------------------------------------------------------------------------
# Class method (Lecture 7): TF = count / length, IDF = log10(N / DF)
# ---------------------------------------------------------------------------
def hand_tfidf(docs: list[str], term: str, doc_i: int) -> tuple[float, list[str]]:
    """TF-IDF of one term in one document, written out the way we do it in class."""
    tokens = [d.split() for d in docs]
    n = len(tokens)
    doc = tokens[doc_i]
    name = DOC_NAMES[doc_i]
    count = doc.count(term)
    df_t = sum(term in d for d in tokens)
    tf = count / len(doc)
    idf = math.log10(n / df_t)
    w = tf * idf
    rows = [
        (f"1. Document length of {name}", f"{len(doc)} words"),
        (f"2. Count('{term}', {name})", f"{count}"),
        (f"3. TF('{term}', {name})", f"{count} / {len(doc)} = {tf:.6f}"),
        ("4. Number of documents N", f"{n}"),
        (f"5. DF('{term}')", f"{df_t} document(s)"),
        ("6. N / DF", f"{n} / {df_t} = {n / df_t:g}"),
        (f"7. IDF('{term}')", f"log10({n / df_t:g}) = {idf:.6f}"),
        (f"8. TF-IDF('{term}', {name})", f"{tf:.6f} x {idf:.6f} = {w:.6f}"),
    ]
    steps = [
        f"Target: TF-IDF of '{term}' in {name} = {' '.join(doc)!r}",
        "Rules (class method): TF = count / doc length,  IDF = log10(N / DF),  TF-IDF = TF x IDF",
    ] + [f"{label:<30} = {value}" for label, value in rows]
    for other in sorted(set(doc)):
        if other != term:
            o_tf = doc.count(other) / len(doc)
            o_df = sum(other in d for d in tokens)
            o_idf = math.log10(n / o_df)
            steps.append(f"   (compare '{other}': TF = {doc.count(other)}/{len(doc)}, DF = {o_df}, "
                         f"IDF = log10({n}/{o_df}) = {o_idf:.6f}, TF-IDF = {o_tf * o_idf:.6f})")
    return w, steps


def library_counts(docs: list[str]):
    """Word counts from scikit-learn's CountVectorizer (the library part)."""
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)
    X = vec.fit_transform(docs).toarray()
    return vec, X


def library_tfidf_matrix(docs: list[str]) -> pd.DataFrame:
    """Class formula applied to the library's count matrix.
    TF  = counts / row sums (document lengths)
    IDF = log10(N / DF). We take DF from TfidfVectorizer(smooth_idf=False), whose
          idf_ is ln(N / DF) + 1. So (idf_ - 1) / ln(10) = log10(N / DF)."""
    vec, counts = library_counts(docs)
    tf = counts / counts.sum(axis=1, keepdims=True)
    sk = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, smooth_idf=False, norm=None)
    sk.fit(docs)
    assert list(sk.get_feature_names_out()) == list(vec.get_feature_names_out())
    idf = (sk.idf_ - 1) / math.log(10)
    return pd.DataFrame(tf * idf, columns=vec.get_feature_names_out(), index=DOC_NAMES[: len(docs)])


def library_tfidf(docs: list[str], term: str, doc_i: int) -> float:
    return float(library_tfidf_matrix(docs).iloc[doc_i][term])


def class_tables(docs: list[str]) -> dict:
    """The same tables as the class Excel workbook: counts, TF, IDF and TF-IDF."""
    vec, counts = library_counts(docs)
    cols = vec.get_feature_names_out()
    n = len(docs)
    df_t = (counts > 0).sum(axis=0)
    counts_df = pd.DataFrame(counts, columns=cols, index=DOC_NAMES[:n])
    counts_df["doc length"] = counts.sum(axis=1)
    tf = pd.DataFrame(counts / counts.sum(axis=1, keepdims=True), columns=cols, index=DOC_NAMES[:n])
    idf = pd.DataFrame({"DF": df_t, "N / DF": n / df_t, "IDF = log10(N/DF)": np.log10(n / df_t)}, index=cols)
    return {"counts": counts_df, "tf": tf.round(6), "idf": idf.round(6), "tfidf": library_tfidf_matrix(docs).round(6)}


# ---------------------------------------------------------------------------
# For comparison: scikit-learn's DEFAULT TfidfVectorizer (used in A2)
# ---------------------------------------------------------------------------
def sklearn_default_steps(docs: list[str], term: str, doc_i: int) -> tuple[float, list[str]]:
    n = len(docs)
    tokens = [d.split() for d in docs]
    doc = tokens[doc_i]
    raw = {}
    steps = ["scikit-learn default: tf = raw count, idf = ln((1 + N) / (1 + DF)) + 1, then divide the row by its length"]
    for t in sorted(set(doc)):
        df_t = sum(t in d for d in tokens)
        raw[t] = doc.count(t) * (math.log((1 + n) / (1 + df_t)) + 1)
        steps.append(f"  '{t}': idf = ln({1 + n}/{1 + df_t}) + 1 = {raw[t] / doc.count(t):.4f}")
    norm = math.sqrt(sum(v * v for v in raw.values()))
    w = raw[term] / norm
    steps.append(f"  row length = {norm:.4f},  weight('{term}') = {raw[term]:.4f} / {norm:.4f} = {w:.4f}")
    return w, steps


def library_sklearn_default(docs: list[str], term: str, doc_i: int) -> float:
    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)
    X = vec.fit_transform(docs)
    return float(X[doc_i, vec.vocabulary_[term]])


# ---------------------------------------------------------------------------
# Bigram probability (Lecture 7): P(next | previous) = count(pair) / count(previous)
# ---------------------------------------------------------------------------
def hand_bigram(docs: list[str], bigram: str, first: str) -> tuple[int, float, list[str]]:
    steps = []
    tokens = [d.split() for d in docs]
    w1, w2 = bigram.split()
    total = 0
    steps.append("Bigrams (n = 2) of each document:")
    for i, d in enumerate(tokens):
        pairs = [f"{a} {b}" for a, b in zip(d, d[1:])]
        c = pairs.count(bigram)
        total += c
        steps.append(f"  {DOC_NAMES[i]} {' '.join(d)!r}: {' · '.join(pairs)}   ->  count('{bigram}') = {c}")
    c_first = sum(d.count(first) for d in tokens)
    prob = total / c_first
    steps.append(f"{'count(' + bigram + ')':<20} = {total}")
    steps.append(f"{'count(' + first + ')':<20} = {c_first}")
    steps.append(f"P({w2} | {w1}) = count({bigram}) / count({w1}) = {total} / {c_first} = {prob:.4f}")
    return total, prob, steps


def library_bigram(docs: list[str], bigram: str, first: str) -> tuple[int, float]:
    bi = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(2, 2))
    Xb = bi.fit_transform(docs)
    uni = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)
    Xu = uni.fit_transform(docs)
    count = int(np.asarray(Xb[:, bi.vocabulary_[bigram]].sum()))
    c_first = int(np.asarray(Xu[:, uni.vocabulary_[first]].sum()))
    return count, count / c_first


def main() -> None:
    raw_docs, docs = toy_corpus()
    print("TOY CORPUS (4 real tweets from our cleaned data), N = 4")
    for i, (r, d) in enumerate(zip(raw_docs, docs)):
        print(f"  {DOC_NAMES[i]}: {d!r:<28} raw: {r!r}")

    print("\n--- TF-IDF BY HAND (class method) ---")
    w, steps = hand_tfidf(docs, TFIDF_TERM, TFIDF_DOC)
    print("\n".join(steps))
    lib = library_tfidf(docs, TFIDF_TERM, TFIDF_DOC)
    print(f"Library (CountVectorizer + TfidfVectorizer idf): {lib:.6f}   ->  {'MATCH' if abs(lib - w) < 1e-9 else 'DIFFERENT'}")

    print("\nFull TF-IDF table (class method):")
    print(class_tables(docs)["tfidf"].to_string())

    print("\n--- BIGRAM BY HAND ---")
    c, p, steps = hand_bigram(docs, BIGRAM, BIGRAM_FIRST)
    print("\n".join(steps))
    lc, lp = library_bigram(docs, BIGRAM, BIGRAM_FIRST)
    ok = lc == c and abs(lp - p) < 1e-12
    print(f"Library (CountVectorizer ngram_range=(2,2)): count = {lc}, P = {lp:.4f}   ->  {'MATCH' if ok else 'DIFFERENT'}")

    print("\n--- FOR COMPARISON: scikit-learn DEFAULT TfidfVectorizer (used in A2) ---")
    w2, steps = sklearn_default_steps(docs, TFIDF_TERM, TFIDF_DOC)
    print("\n".join(steps))
    lib2 = library_sklearn_default(docs, TFIDF_TERM, TFIDF_DOC)
    print(f"Library (TfidfVectorizer default): {lib2:.4f}   ->  {'MATCH' if abs(lib2 - w2) < 1e-9 else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
