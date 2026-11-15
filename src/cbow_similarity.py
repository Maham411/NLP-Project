"""
cbow_similarity.py: Part C1 + C2 (similarity checks with the CBOW model)

Loads the saved CBOW model (made by cbow_train.py) and
  C1: prints the top-N most similar words for some airline-related query words
  C2: prints the cosine similarity of two similar pairs and two unrelated pairs

Run from the repository root:
    python src/cbow_train.py
    python src/cbow_similarity.py
"""

import sys
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ROOT

MODEL_PATH = ROOT / "models" / "word2vec_cbow.model"
OUT_DIR = ROOT / "report" / "part_c"

TOP_N = 8
# The corpus is already lemmatised by Phase 1, so we use words like "delay" and "cancel".
QUERY_WORDS = ["delay", "bag", "cancel", "service", "refund"]
SIMILAR_PAIRS = [("delay", "cancel"), ("bag", "luggage")]
UNRELATED_PAIRS = [("pilot", "refund"), ("luggage", "love")]


def main():
    model = Word2Vec.load(str(MODEL_PATH))
    wv = model.wv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # C1: top-N neighbours
    rows = []
    print(f"C1: top {TOP_N} most similar words (CBOW)\n")
    for word in QUERY_WORDS:
        if word not in wv.key_to_index:
            print(f"  {word}: not in vocabulary")
            continue
        neighbours = wv.most_similar(word, topn=TOP_N)
        print(f"  {word}: " + ", ".join(f"{w} ({s:.2f})" for w, s in neighbours))
        for rank, (w, s) in enumerate(neighbours, start=1):
            rows.append({"query": word, "rank": rank, "neighbour": w, "cosine": round(float(s), 4)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "c1_cbow_neighbours.csv", index=False)

    # C2: cosine similarity of word pairs
    pair_rows = []
    print("\nC2: cosine similarity (CBOW)\n")
    for kind, pairs in (("similar", SIMILAR_PAIRS), ("unrelated", UNRELATED_PAIRS)):
        for a, b in pairs:
            score = float(wv.similarity(a, b))
            print(f"  {kind:>9}: {a} - {b} = {score:.3f}")
            pair_rows.append({"expected": kind, "word_1": a, "word_2": b, "cosine": round(score, 4)})
    pd.DataFrame(pair_rows).to_csv(OUT_DIR / "c2_cbow_cosine.csv", index=False)


if __name__ == "__main__":
    main()
