import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import documents, load_corpus
from vectorize import (COUNT_BASED, REPRESENTATIONS, _count_vectorizer, available_representations, dimension,
                       model_path, vectorize, vectorize_many)

RAW = "@united my flight was Cancelled Flightled again!!! worst customer service ever :("
NAMES = available_representations()


def as_array(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def test_every_available_representation_returns_a_vector_of_its_dimension():
    for name in NAMES:
        vec = vectorize(RAW, name)
        assert vec.ndim == 1 and len(vec) == dimension(name), name
        assert np.isfinite(vec).all() and np.any(vec), name


def test_same_input_gives_same_vector():
    for name in NAMES:
        assert np.array_equal(vectorize(RAW, name), vectorize(RAW, name)), name


def test_raw_and_already_clean_text_agree():
    from cleaning_pipeline import clean_text
    cleaned = clean_text(RAW)
    for name in NAMES:
        assert np.allclose(vectorize(RAW, name), vectorize(cleaned, name, already_clean=True)), name


def test_count_based_vectors_match_the_part_a_matrix():
    from count_representations import build_bow, build_tfidf
    docs = documents(load_corpus())
    for name, build in (("bow", build_bow), ("tfidf", build_tfidf)):
        if name not in NAMES:
            continue
        _, X = build(docs)
        for row in (0, 5, 100):
            assert np.allclose(vectorize(docs[row], name, already_clean=True), X.getrow(row).toarray().ravel())


def test_batch_matches_single_documents():
    texts = [RAW, "@JetBlue thanks for the upgrade!", "@united where is my bag??"]
    for name in NAMES:
        batch = as_array(vectorize_many(texts, name))
        assert batch.shape == (3, dimension(name)), name
        for i, text in enumerate(texts):
            assert np.allclose(batch[i], vectorize(text, name)), name


def test_empty_document_gives_zero_vector():
    for name in NAMES:
        vec = vectorize("@united http://t.co/abc", name)
        assert len(vec) == dimension(name) and not np.any(vec), name


def test_unseen_word_handling():
    for name in NAMES:
        vec = vectorize("custmer", name, already_clean=True)
        if name == "fasttext":
            assert np.any(vec)
        else:
            assert not np.any(vec), name


def test_unknown_representation_is_rejected():
    try:
        vectorize(RAW, "glove")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_available_means_saved_on_disk():
    assert set(NAMES) <= set(REPRESENTATIONS)
    for name in REPRESENTATIONS:
        assert (name in NAMES) == model_path(name).exists(), name


def test_count_based_load_the_saved_vectorizer_without_refitting():
    import joblib
    for name in COUNT_BASED:
        if name not in NAMES:
            continue
        saved = joblib.load(model_path(name))
        assert _count_vectorizer(name).vocabulary_ == saved.vocabulary_, name


def test_missing_model_gives_a_clear_error():
    missing = [n for n in REPRESENTATIONS if n not in NAMES]
    if not missing:
        return
    try:
        vectorize(RAW, missing[0])
    except FileNotFoundError as e:
        assert "models/" in str(e)
        return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    print(f"Testing with: {', '.join(NAMES)}")
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {test.__name__} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
