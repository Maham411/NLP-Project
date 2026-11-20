"""Tokenization: Regex vs NLTK vs spaCy on airline tweets."""
import re
import glob
import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_nlp = spacy.blank("en")  # spaCy rule-based tokenizer, no model download needed

# Order matters: URLs, @mentions, #hashtags, emoticons, words, numbers, any symbol
_TOKEN_RE = re.compile(
    r"https?://\S+"
    r"|[@#]\w+"
    r"|[:;=][-o]?[)(DPp/\\|]"
    r"|[A-Za-z]+(?:['\u2019-][A-Za-z]+)*"
    r"|\d+(?:[.,:]\d+)*"
    r"|[^\w\s]"
)


def regex_tokenize(text):
    return _TOKEN_RE.findall(text)


def nltk_tokenize(text):
    return word_tokenize(text)


def spacy_tokenize(text):
    return [t.text for t in _nlp(text)]


TOKENIZERS = {"Regex": regex_tokenize, "NLTK": nltk_tokenize, "spaCy": spacy_tokenize}


def load_tweets():
    """Find Tweets.csv anywhere in the repo and load it. Falls back to samples."""
    files = glob.glob("**/Tweets.csv", recursive=True) + glob.glob("../**/Tweets.csv", recursive=True)
    if files:
        df = pd.read_csv(files[0])
        print("Loaded:", files[0], "| rows:", len(df))
        return df
    print("Tweets.csv not found - using sample tweets")
    return pd.DataFrame({"text": [
        "@VirginAmerica I can't get my bag, flight was delayed 3 hours!!!",
        "@united thanks for nothing :( http://t.co/abc123 #fail",
        "@AmericanAir won't answer, been on hold since 5:30pm... #neveragain",
        "@SouthwestAir you're the best! Landed 10 mins early :) #LUVSWA",
        "@JetBlue Why is my co-worker's flight $1,200?? Seriously.",
        "@Delta lost my luggage AGAIN. Can't believe it!!!",
    ]})


def compare(texts):
    rows = []
    for t in texts:
        row = {"tweet": t}
        for name, fn in TOKENIZERS.items():
            toks = fn(t)
            row[name] = toks
            row[f"{name}_n"] = len(toks)
        rows.append(row)
    return pd.DataFrame(rows)


def show_differences(texts):
    for t in texts:
        out = {n: fn(t) for n, fn in TOKENIZERS.items()}
        if len({tuple(v) for v in out.values()}) > 1:
            print("\nTWEET:", t)
            for n, toks in out.items():
                print(f"  {n:6}({len(toks):2}):", toks)


if __name__ == "__main__":
    df = load_tweets()
    show_differences(df["text"].head(8).tolist())
    s = df["text"].sample(min(1000, len(df)), random_state=42)
    print("\nAvg tokens per tweet (sample):")
    for n, fn in TOKENIZERS.items():
        print(f"  {n}: {s.map(lambda x: len(fn(x))).mean():.2f}")
