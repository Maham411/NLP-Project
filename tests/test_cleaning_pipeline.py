import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleaning_pipeline import build_clean_dataset, clean_text, ensure_nltk_data

ensure_nltk_data()


def test_urls_and_mentions_are_removed():
    out = clean_text("@united where is my bag?? http://t.co/abc123")
    assert "http" not in out and "united" not in out
    assert "bag" in out.split()


def test_html_entities_are_decoded():
    assert "amp" not in clean_text("@JetBlue food &amp; drinks were great")


def test_negations_survive():
    assert "not" in clean_text("@united I don't think so").split()
    assert "not" in clean_text("@united you cant be serious").split()
    assert "never" in clean_text("@united never flying with you again").split()


def test_hashtag_word_is_kept():
    assert "fail" in clean_text("@united #fail").split()


def test_emoji_and_emoticons_become_words():
    out = clean_text("@VirginAmerica I ❤️ you :) but not this :(").split()
    assert "emoji_red_heart" in out
    assert "happyface" in out and "sadface" in out


def test_elongation_and_repeated_marks():
    out = clean_text("@united sooooo late!!!").split()
    assert "soo" in out
    assert out.count("!") == 1


def test_lemmatization():
    out = clean_text("@united flights were delayed and bags cancelled").split()
    assert {"flight", "delay", "bag", "cancel"} <= set(out)


def test_unicode_forms_give_same_result():
    composed = "@united café was nice"
    decomposed = "@united café was nice"
    assert clean_text(composed) == clean_text(decomposed)


def test_cleaning_is_stable():
    once = clean_text("@united Flight 1234 delayed 3 hrs!!! :( #worst")
    assert clean_text(once) == once


def test_labels_and_rows_are_preserved():
    raw = pd.DataFrame({
        "tweet_id": [1, 2, 3],
        "airline": ["United", "Delta", "Virgin America"],
        "airline_sentiment": ["negative", "positive", "neutral"],
        "airline_sentiment_confidence": [1.0, 0.7, 0.5],
        "negativereason": ["Late Flight", None, None],
        "text": ["@united late again", "@Delta thanks!", "@VirginAmerica http://t.co/x"],
    })
    out = build_clean_dataset(raw)
    assert len(out) == 3
    assert out["airline_sentiment"].tolist() == raw["airline_sentiment"].tolist()
    assert out["tweet_id"].tolist() == [1, 2, 3]
    assert out.loc[2, "clean_text"] == ""


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
