"""
hand_worked_a4.py: Assignment 2, Part A4: hand worked TF-IDF weight and bigram
Member 1: Maham Iqbal (L1F23BSCS0411)

We take a TINY toy corpus of 4 real tweets from our cleaned data, compute one
TF-IDF weight and one bigram step by step, and check that scikit-learn gives
the same numbers. Do the same arithmetic on paper and attach the scan.

Run from the repository root:
    python src/hand_worked_a4.py
"""

import math
import sys
from pathlib import Path

import numpy as np
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
TFIDF_TERM, TFIDF_DOC = "hawaii", 0          # weight of "hawaii" in D1
BIGRAM, BIGRAM_FIRST = "book flight", "book"  # bigram count and P(flight | book)


def toy_corpus():
    df = load_corpus()
    rows = df.set_index("tweet_id").loc[TOY_TWEET_IDS]
    return rows["text"].tolist(), rows[TEXT_COL].tolist()


def hand_tfidf(docs: list[str], term: str, doc_i: int) -> tuple[float, list[str]]:
    """TF-IDF exactly as scikit-learn computes it (default settings), written out."""
    steps = []
    n = len(docs)
    tokens = [d.split() for d in docs]
    doc = tokens[doc_i]
    steps.append(f"Corpus size n = {n}.  Target: weight of '{term}' in D{doc_i + 1} = {' '.join(doc)!r}")
    steps.append("Formula (scikit-learn default): idf(t) = ln((1 + n) / (1 + df(t))) + 1,  "
                 "tfidf = tf x idf,  then divide by the length of the whole row (L2 norm)")

    raw = {}
    for t in sorted(set(doc)):
        tf = doc.count(t)
        df_t = sum(t in d for d in tokens)
        idf = math.log((1 + n) / (1 + df_t)) + 1
        raw[t] = tf * idf
        steps.append(f"  '{t}': tf = {tf}, df = {df_t}, idf = ln({1 + n}/{1 + df_t}) + 1 = "
                     f"{math.log((1 + n) / (1 + df_t)):.4f} + 1 = {idf:.4f},  tf x idf = {tf * idf:.4f}")

    sq = " + ".join(f"{v:.4f}^2" for v in raw.values())
    norm = math.sqrt(sum(v * v for v in raw.values()))
    steps.append(f"Row length = sqrt({sq}) = sqrt({sum(v * v for v in raw.values()):.4f}) = {norm:.4f}")
    weight = raw[term] / norm
    steps.append(f"tfidf('{term}', D{doc_i + 1}) = {raw[term]:.4f} / {norm:.4f} = {weight:.4f}")
    for t in raw:
        if t != term:
            steps.append(f"  (for comparison: '{t}' = {raw[t]:.4f} / {norm:.4f} = {raw[t] / norm:.4f})")
    return weight, steps


def library_tfidf(docs: list[str], term: str, doc_i: int) -> float:
    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)   # same defaults
    X = vec.fit_transform(docs)
    return float(X[doc_i, vec.vocabulary_[term]])


def hand_bigram(docs: list[str], bigram: str, first: str) -> tuple[int, float, list[str]]:
    steps = []
    tokens = [d.split() for d in docs]
    w1, w2 = bigram.split()
    total = 0
    for i, d in enumerate(tokens):
        pairs = [f"{a} {b}" for a, b in zip(d, d[1:])]
        c = pairs.count(bigram)
        total += c
        steps.append(f"  D{i + 1} {' '.join(d)!r}: bigrams = {pairs}  ->  '{bigram}' appears {c} time(s)")
    steps.append(f"Count('{bigram}') in the whole toy corpus = {total}")
    c_first = sum(d.count(first) for d in tokens)
    prob = total / c_first
    steps.append(f"Count('{first}') = {c_first}")
    steps.append(f"Bigram probability P({w2} | {w1}) = Count('{bigram}') / Count('{w1}') = {total} / {c_first} = {prob:.4f}")
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
    print("TOY CORPUS (4 real tweets from our cleaned data)")
    for i, (r, d) in enumerate(zip(raw_docs, docs), start=1):
        print(f"  D{i}: {d!r:<28} raw: {r!r}")

    print("\n--- TF-IDF BY HAND ---")
    w, steps = hand_tfidf(docs, TFIDF_TERM, TFIDF_DOC)
    print("\n".join(steps))
    lib = library_tfidf(docs, TFIDF_TERM, TFIDF_DOC)
    print(f"Library (TfidfVectorizer): {lib:.4f}   ->  {'MATCH' if abs(lib - w) < 1e-9 else 'DIFFERENT'}")

    print("\n--- BIGRAM BY HAND ---")
    c, p, steps = hand_bigram(docs, BIGRAM, BIGRAM_FIRST)
    print("\n".join(steps))
    lc, lp = library_bigram(docs, BIGRAM, BIGRAM_FIRST)
    ok = lc == c and abs(lp - p) < 1e-12
    print(f"Library (CountVectorizer ngram_range=(2,2)): count = {lc}, P = {lp:.4f}   ->  {'MATCH' if ok else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
