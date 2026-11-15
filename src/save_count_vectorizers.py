import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ROOT, documents, load_corpus
from count_representations import build_bow, build_tfidf
from vectorize import MODEL_DIR, SAVED_VECTORIZERS


def main():
    docs = documents(load_corpus())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fitting on {len(docs):,} tweets from the shared corpus (src/corpus.py)")
    for name, build in (("bow", build_bow), ("tfidf", build_tfidf)):
        vectorizer, X = build(docs)
        path = MODEL_DIR / SAVED_VECTORIZERS[name]
        joblib.dump(vectorizer, path)
        size_kb = path.stat().st_size / 1024
        print(f"  {name:<6} vocabulary={len(vectorizer.vocabulary_):,}  matrix={X.shape}  "
              f"-> {path.relative_to(ROOT)} ({size_kb:.0f} KB)")
    print("Phase 3 loads these files through src/vectorize.py, nothing is refitted there.")


if __name__ == "__main__":
    main()
