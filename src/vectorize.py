import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from corpus import ROOT

MODEL_DIR = ROOT / "models"
COUNT_BASED = ("bow", "tfidf")
SAVED_VECTORIZERS = {
    "bow": "bow_vectorizer.joblib",
    "tfidf": "tfidf_vectorizer.joblib",
}
LEARNED = {
    "cbow": ("word2vec_cbow.model", "cbow.model"),
    "skipgram": ("word2vec_skipgram.model", "skipgram.model"),
    "fasttext": ("fasttext.model",),
}
MODEL_FILES = {**{k: (v,) for k, v in SAVED_VECTORIZERS.items()}, **LEARNED}
REPRESENTATIONS = COUNT_BASED + tuple(LEARNED)
SAVE_HINT = {
    "bow": "python src/save_count_vectorizers.py",
    "tfidf": "python src/save_count_vectorizers.py",
    "cbow": "the Part B1 training script",
    "skipgram": "the Part B2 training script",
    "fasttext": "the Part B4 training script",
}


def model_path(representation):
    candidates = [MODEL_DIR / name for name in MODEL_FILES[representation]]
    return next((p for p in candidates if p.exists()), candidates[0])


def available_representations():
    return [name for name in REPRESENTATIONS if model_path(name).exists()]


def _check(representation):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}, choose from {REPRESENTATIONS}")
    if not model_path(representation).exists():
        names = " or ".join(f"models/{n}" for n in MODEL_FILES[representation])
        raise FileNotFoundError(
            f"{representation} is not saved at {names} yet. "
            f"Run {SAVE_HINT[representation]} first. Available now: {available_representations()}"
        )


@lru_cache(maxsize=None)
def _count_vectorizer(representation):
    import joblib
    return joblib.load(model_path(representation))


@lru_cache(maxsize=None)
def _word_vectors(representation):
    if representation == "fasttext":
        from gensim.models import FastText
        return FastText.load(str(model_path(representation))).wv
    from gensim.models import Word2Vec
    return Word2Vec.load(str(model_path(representation))).wv


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
        print(f"Missing (not saved in models/ yet): {', '.join(missing)}")
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
