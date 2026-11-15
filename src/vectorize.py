import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from corpus import ROOT, documents, load_corpus

MODEL_DIR = ROOT / "models"
COUNT_BASED = ("bow", "tfidf")
LEARNED = {
    "cbow": "word2vec_cbow.model",
    "skipgram": "word2vec_skipgram.model",
    "fasttext": "fasttext.model",
}
REPRESENTATIONS = COUNT_BASED + tuple(LEARNED)


def available_representations():
    return list(COUNT_BASED) + [name for name, file in LEARNED.items() if (MODEL_DIR / file).exists()]


def _check(representation):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}, choose from {REPRESENTATIONS}")
    if representation in LEARNED and not (MODEL_DIR / LEARNED[representation]).exists():
        raise FileNotFoundError(
            f"{representation} model not found at models/{LEARNED[representation]}. "
            f"Available now: {available_representations()}"
        )


@lru_cache(maxsize=None)
def _count_vectorizer(representation):
    from count_representations import build_bow, build_tfidf
    build = build_bow if representation == "bow" else build_tfidf
    vectorizer, _ = build(documents(load_corpus()))
    return vectorizer


@lru_cache(maxsize=None)
def _word_vectors(representation):
    if representation == "fasttext":
        from gensim.models import FastText
        return FastText.load(str(MODEL_DIR / LEARNED[representation])).wv
    from gensim.models import Word2Vec
    return Word2Vec.load(str(MODEL_DIR / LEARNED[representation])).wv


def preprocess(raw_text):
    from cleaning_pipeline import clean_text
    return clean_text(raw_text).split()


def dimension(representation):
    _check(representation)
    if representation in COUNT_BASED:
        return len(_count_vectorizer(representation).vocabulary_)
    return _word_vectors(representation).vector_size


def _embed(tokens, representation):
    wv = _word_vectors(representation)
    if representation == "fasttext":
        known = tokens
    else:
        known = [t for t in tokens if t in wv.key_to_index]
    if not known:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.mean([wv[t] for t in known], axis=0).astype(np.float32)


def vectorize(raw_text, representation="tfidf", already_clean=False):
    _check(representation)
    tokens = str(raw_text).split() if already_clean else preprocess(raw_text)
    if representation in COUNT_BASED:
        row = _count_vectorizer(representation).transform([" ".join(tokens)])
        return row.toarray().ravel()
    return _embed(tokens, representation)


def vectorize_many(raw_texts, representation="tfidf", already_clean=False):
    _check(representation)
    token_lists = [str(t).split() if already_clean else preprocess(t) for t in raw_texts]
    if representation in COUNT_BASED:
        return _count_vectorizer(representation).transform([" ".join(t) for t in token_lists])
    return np.vstack([_embed(t, representation) for t in token_lists])


def coverage(raw_text, representation, already_clean=False):
    _check(representation)
    tokens = str(raw_text).split() if already_clean else preprocess(raw_text)
    if representation in COUNT_BASED:
        vocab = _count_vectorizer(representation).vocabulary_
        known = [t for t in tokens if t in vocab]
    elif representation == "fasttext":
        known = tokens
    else:
        known = [t for t in tokens if t in _word_vectors(representation).key_to_index]
    return {"tokens": tokens, "known": known, "unknown": [t for t in tokens if t not in known]}


def main():
    examples = [
        "@united my flight was Cancelled Flightled again!!! worst customer service ever :(",
        "@JetBlue thanks for the upgrade, the crew was AMAZING \U0001F60D",
        "@SouthwestAir what time does the gate open for flight 1234?",
        "@AmericanAir my bagagge is lost and nobody is helping",
    ]
    names = available_representations()
    print(f"Available representations: {', '.join(names)}")
    missing = [n for n in REPRESENTATIONS if n not in names]
    if missing:
        print(f"Missing (model file not in models/ yet): {', '.join(missing)}")
    for text in examples:
        print(f"\nraw   : {text}")
        print(f"tokens: {preprocess(text)}")
        for name in names:
            vec = vectorize(text, name)
            nonzero = int(np.count_nonzero(vec))
            unknown = coverage(text, name)["unknown"]
            print(f"  {name:<9} dim={len(vec):<5} non-zero={nonzero:<4} "
                  f"norm={np.linalg.norm(vec):.3f}  unknown words={unknown}")
    batch = vectorize_many(examples, names[0])
    print(f"\nvectorize_many({len(examples)} tweets, {names[0]!r}) -> {type(batch).__name__} {batch.shape}")


if __name__ == "__main__":
    main()
