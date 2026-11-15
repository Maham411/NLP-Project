"""
cbow_train.py: Part B1 + D1 (Word2Vec CBOW)

Trains a Word2Vec model with the CBOW variant (sg=0) on the shared corpus
and saves it to models/word2vec_cbow.model so Phase 3 can load it without retraining.

Run from the repository root:
    python src/cbow_train.py
"""

import json
import sys
from pathlib import Path

from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ROOT, load_corpus, tokenized

MODEL_PATH = ROOT / "models" / "word2vec_cbow.model"
INFO_PATH = ROOT / "report" / "part_b" / "b1_cbow_info.json"

# Hyperparameters. The Skip-gram model (B2) must use the SAME values,
# only sg changes (sg=0 is CBOW, sg=1 is Skip-gram).
PARAMS = dict(
    vector_size=100,  # length of each word vector
    window=5,         # words looked at on each side of the centre word
    min_count=2,      # ignore words that appear fewer than 2 times
    epochs=20,        # passes over the corpus
    seed=42,          # same result every run
    workers=1,        # one thread, so the seed really gives the same result
)


def main():
    sentences = tokenized(load_corpus())   # list of token lists, as gensim needs
    print(f"Training CBOW on {len(sentences):,} tweets")

    model = Word2Vec(sentences, sg=0, **PARAMS)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))   # D1: save once, load later with Word2Vec.load()

    info = {
        "model": "Word2Vec CBOW (sg=0)",
        "tweets_used": len(sentences),
        "vocabulary_size": len(model.wv),
        **PARAMS,
        "saved_to": str(MODEL_PATH.relative_to(ROOT)),
    }
    INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    INFO_PATH.write_text(json.dumps(info, indent=2))

    for key, value in info.items():
        print(f"  {key:>16}: {value}")


if __name__ == "__main__":
    main()
