import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from corpus import ROOT, documents, load_corpus, tokenized
from vectorize import (COUNT_BASED, LEARNED, MODEL_DIR, _count_vectorizer, _word_vectors,
                       available_representations, vectorize_many)

B_DIR = ROOT / "report" / "part_b"
C_DIR = ROOT / "report" / "part_c"
FIG_DIR = ROOT / "report" / "figures"

RELATED_PAIRS = [("bag", "luggage"), ("delay", "late"), ("great", "awesome"),
                 ("rude", "unhelpful"), ("voucher", "compensation")]
UNRELATED_PAIRS = [("bag", "thank"), ("pilot", "refund"), ("snow", "happyface")]
NEIGHBOUR_WORDS = ["bag", "rude", "voucher", "layover", "turbulence"]
UNSEEN_WORDS = ["bagagge", "custmer", "refundd", "flihgt", "wifii", "covid"]


def corpus_stats():
    df = load_corpus()
    docs = documents(df)
    counts = Counter(t for s in tokenized(df) for t in s)
    return docs, counts


def matrix_profile(name, docs):
    start = time.perf_counter()
    X = vectorize_many(docs, name, already_clean=True)
    seconds = time.perf_counter() - start
    if hasattr(X, "nnz"):
        nonzero = X.nnz
        stored_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
        all_zero_rows = int((X.getnnz(axis=1) == 0).sum())
    else:
        nonzero = int(np.count_nonzero(X))
        stored_mb = X.nbytes / 1e6
        all_zero_rows = int((~X.any(axis=1)).sum())
    cells = X.shape[0] * X.shape[1]
    return {
        "representation": name,
        "type": "count-based" if name in COUNT_BASED else "learned",
        "dimensions": int(X.shape[1]),
        "dimension_set_by": "vocabulary size" if name in COUNT_BASED else "vector_size hyperparameter",
        "sparsity_percent": round(100 * (1 - nonzero / cells), 3),
        "avg_nonzero_per_tweet": round(nonzero / X.shape[0], 1),
        "tweets_with_all_zero_vector": all_zero_rows,
        "stored_MB": round(stored_mb, 2),
        "seconds_to_vectorize_corpus": round(seconds, 2),
    }


def coverage_profile(name, counts):
    total = sum(counts.values())
    if name in COUNT_BASED:
        vocab = _count_vectorizer(name).vocabulary_
        known = {w for w in counts if w in vocab}
    elif name == "fasttext":
        known = set(counts)
    else:
        known = {w for w in counts if w in _word_vectors(name).key_to_index}
    covered = sum(counts[w] for w in known)
    return {
        "representation": name,
        "corpus_word_types": len(counts),
        "types_with_a_vector": len(known) if name != "fasttext" else len(counts),
        "token_coverage_percent": round(100 * covered / total, 2),
        "types_dropped": len(counts) - len(known),
    }


def unseen_word_table(names):
    rows = []
    for word in UNSEEN_WORDS:
        row = {"word": word}
        for name in names:
            vec = vectorize_many([word], name, already_clean=True)
            nonzero = vec.nnz if hasattr(vec, "nnz") else int(np.count_nonzero(vec))
            row[name] = "vector" if nonzero else "all zeros"
        if "fasttext" in names:
            row["fasttext_nearest"] = ", ".join(w for w, _ in _word_vectors("fasttext").most_similar(word, topn=3))
        rows.append(row)
    return pd.DataFrame(rows)


def count_word_vectors(name, docs):
    vectorizer = _count_vectorizer(name)
    columns = normalize(vectorizer.transform(docs).T.tocsr())
    return vectorizer.vocabulary_, columns


def word_similarity_table(names, docs):
    count_vectors = {n: count_word_vectors(n, docs) for n in names if n in COUNT_BASED}
    rows = []
    for kind, pairs in (("related", RELATED_PAIRS), ("unrelated", UNRELATED_PAIRS)):
        for a, b in pairs:
            row = {"pair": f"{a} / {b}", "kind": kind}
            for name in names:
                if name in COUNT_BASED:
                    vocab, cols = count_vectors[name]
                    row[name] = round(float(cols[vocab[a]].multiply(cols[vocab[b]]).sum()), 3)
                else:
                    row[name] = round(float(_word_vectors(name).similarity(a, b)), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def training_time(name):
    from gensim.models import FastText, Word2Vec
    cls = FastText if name == "fasttext" else Word2Vec
    saved = cls.load(str(MODEL_DIR / LEARNED[name]))
    params = dict(vector_size=saved.vector_size, window=saved.window, min_count=saved.min_count,
                  epochs=saved.epochs, sg=saved.sg, negative=saved.negative, hs=saved.hs,
                  sample=saved.sample, seed=42, workers=1)
    sents = tokenized(load_corpus())
    start = time.perf_counter()
    cls(sents, **params)
    return round(time.perf_counter() - start, 2), params


def cbow_vs_skipgram(counts):
    timing = []
    for name in ("cbow", "skipgram"):
        seconds, params = training_time(name)
        timing.append({"model": name, "training_seconds": seconds,
                       **{k: v for k, v in params.items() if k not in ("seed", "workers")}})
    timing = pd.DataFrame(timing)
    ratio = round(timing.loc[1, "training_seconds"] / timing.loc[0, "training_seconds"], 2)

    neighbours = []
    for word in NEIGHBOUR_WORDS:
        neighbours.append({
            "word": word,
            "count_in_corpus": counts[word],
            "cbow_top5": ", ".join(w for w, _ in _word_vectors("cbow").most_similar(word, topn=5)),
            "skipgram_top5": ", ".join(w for w, _ in _word_vectors("skipgram").most_similar(word, topn=5)),
        })
    return timing, ratio, pd.DataFrame(neighbours)


def plot_similarity(sim, names, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sim))
    width = 0.8 / len(names)
    colours = {"bow": "#C9CED6", "tfidf": "#8C96A6", "cbow": "#4C72B0", "skipgram": "#55A868", "fasttext": "#DD8452"}
    for i, name in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * width, sim[name], width, label=name, color=colours.get(name))
    n_related = (sim["kind"] == "related").sum()
    ax.axvline(n_related - 0.5, color="#999999", linestyle="--", linewidth=1)
    ax.text(n_related / 2 - 0.5, 1.02, "related pairs", ha="center", fontsize=9, color="#555555")
    ax.text(n_related + (len(sim) - n_related) / 2 - 0.5, 1.02, "unrelated pairs", ha="center", fontsize=9, color="#555555")
    ax.set_xticks(x, sim["pair"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(min(0, sim[names].min().min()) - 0.05, 1.1)
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.legend(ncol=len(names), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("C4. Count-based word vectors miss synonyms that rarely share a tweet")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def header(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    for d in (B_DIR, C_DIR):
        d.mkdir(parents=True, exist_ok=True)
    names = available_representations()
    missing = [n for n in LEARNED if n not in names]
    print(f"Representations compared: {', '.join(names)}")
    if missing:
        print(f"Not available yet (no model in models/): {', '.join(missing)}")
    docs, counts = corpus_stats()
    summary = {"documents": len(docs), "representations": names}

    header("C4. DIMENSIONALITY, SPARSITY AND SIZE")
    profile = pd.DataFrame([matrix_profile(n, docs) for n in names])
    print(profile.drop(columns=["dimension_set_by"]).to_string(index=False))
    profile.to_csv(C_DIR / "c4_dimensionality_sparsity.csv", index=False)
    summary["c4_dimensionality_sparsity"] = profile.to_dict("records")

    header("C4. OOV HANDLING")
    cover = pd.DataFrame([coverage_profile(n, counts) for n in names])
    print(cover.to_string(index=False))
    unseen = unseen_word_table(names)
    print("\nWords never seen in the corpus (misspellings and a new word):")
    print(unseen.to_string(index=False))
    cover.to_csv(C_DIR / "c4_vocabulary_coverage.csv", index=False)
    unseen.to_csv(C_DIR / "c4_unseen_words.csv", index=False)
    summary["c4_vocabulary_coverage"] = cover.to_dict("records")
    summary["c4_unseen_words"] = unseen.to_dict("records")

    header("C4. WHAT EACH REPRESENTATION CAPTURES (word-pair cosine similarity)")
    sim = word_similarity_table(names, docs)
    print(sim.to_string(index=False))
    sim.to_csv(C_DIR / "c4_word_similarity.csv", index=False)
    plot_similarity(sim, names, FIG_DIR / "c4_word_similarity.png")
    summary["c4_word_similarity"] = sim.to_dict("records")

    if {"cbow", "skipgram"} <= set(names):
        header("B3. CBOW VS SKIP-GRAM")
        timing, ratio, neighbours = cbow_vs_skipgram(counts)
        print(timing.to_string(index=False))
        print(f"\nSkip-gram took {ratio}x as long as CBOW with the same settings.")
        print("\nNearest neighbours, frequent words first, rare words last:")
        print(neighbours.to_string(index=False))
        timing.to_csv(B_DIR / "b3_training_time.csv", index=False)
        neighbours.to_csv(B_DIR / "b3_neighbours.csv", index=False)
        summary["b3_training_time"] = timing.to_dict("records")
        summary["b3_skipgram_to_cbow_time_ratio"] = ratio
        summary["b3_neighbours"] = neighbours.to_dict("records")

    with open(C_DIR / "part_b3_c4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    header("DONE")
    print(f"Tables  -> {B_DIR.relative_to(ROOT)}/b3_*.csv and {C_DIR.relative_to(ROOT)}/c4_*.csv")
    print(f"Figure  -> {(FIG_DIR / 'c4_word_similarity.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
