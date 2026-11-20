import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.tokenize import TweetTokenizer, word_tokenize

DATA_PATH = Path("Tweets.csv")
OUT_DIR = Path("outputs")
SAMPLE_SIZE = 1000
SEED = 42

for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

REGEX_PATTERN = re.compile(
    r"https?://\S+"
    r"|@\w+"
    r"|#\w+"
    r"|\d+(?:[:.,]\d+)*"
    r"|\w+(?:'\w+)?"
    r"|[^\w\s]"
)

_tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)

_spacy_nlp = None
try:
    import spacy
    _spacy_nlp = spacy.blank("en")
except Exception:
    _spacy_nlp = None


def regex_tokenize(text):
    return REGEX_PATTERN.findall(text)


def nltk_tokenize(text):
    return word_tokenize(text)


def tweet_tokenize(text):
    return _tweet_tokenizer.tokenize(text)


def spacy_tokenize(text):
    return [t.text for t in _spacy_nlp.make_doc(text)]


TOKENIZERS = {
    "regex": regex_tokenize,
    "nltk_word": nltk_tokenize,
    "nltk_tweet": tweet_tokenize,
}
if _spacy_nlp is not None:
    TOKENIZERS["spacy"] = spacy_tokenize


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df


def tokenizer_stats(texts, tokenizer):
    all_tokens = []
    per_text = []
    for t in texts:
        toks = tokenizer(t)
        per_text.append(len(toks))
        all_tokens.extend(toks)
    types = set(all_tokens)
    return {
        "total_tokens": len(all_tokens),
        "vocab_size": len(types),
        "tokens_per_text": sum(per_text) / len(per_text),
        "type_token_ratio": len(types) / len(all_tokens),
    }


CHALLENGE_TEXTS = {
    "URL": "@united check this http://t.co/abc123 now",
    "Mention + hashtag": "@AmericanAir worst flight ever #fail #neveragain",
    "Contraction": "I can't believe they didn't fix it, it's 2 hrs late",
    "Emoticon / emoji": "thanks for nothing :( \U0001F620 @Delta",
    "Time / number": "Flight 3412 delayed until 3:45pm, lost $1,200 bag",
    "Elongation / repeats": "sooo happyyyy!!! best airline eveeer",
}


def challenge_table():
    rows = []
    for name, text in CHALLENGE_TEXTS.items():
        row = {"case": name, "text": text}
        for tname, tok in TOKENIZERS.items():
            row[tname] = tok(text)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_on_sample(df):
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    texts = sample["text"].tolist()
    stats = {name: tokenizer_stats(texts, tok) for name, tok in TOKENIZERS.items()}
    stats_df = pd.DataFrame(stats).T
    return sample, stats_df


def pattern_preservation(texts):
    url_re = re.compile(r"https?://\S+")
    mention_re = re.compile(r"@\w+")
    hashtag_re = re.compile(r"#\w+")
    emo_re = re.compile(r"[:;=]['\-]?[()DPp/]")
    rows = []
    for name, tok in TOKENIZERS.items():
        kept = {"url": 0, "mention": 0, "hashtag": 0, "emoticon": 0}
        total = {"url": 0, "mention": 0, "hashtag": 0, "emoticon": 0}
        for t in texts:
            toks = set(tok(t))
            for key, rx in (("url", url_re), ("mention", mention_re), ("hashtag", hashtag_re), ("emoticon", emo_re)):
                for m in rx.findall(t):
                    total[key] += 1
                    if m in toks:
                        kept[key] += 1
        row = {"tokenizer": name}
        for key in kept:
            row[key + "_kept_%"] = round(100 * kept[key] / total[key], 1) if total[key] else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print("Tweets.csv not found in the current folder")
        sys.exit(1)

    df = load_data()
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print()

    sample, stats_df = compare_on_sample(df)
    print("=== Tokenizer comparison on", len(sample), "random tweets ===")
    print(stats_df.round(3).to_string())
    print()

    pres = pattern_preservation(sample["text"].tolist())
    print("=== % of special patterns kept as a single token ===")
    print(pres.to_string(index=False))
    print()

    print("=== 5 example tweets ===")
    for text in sample["text"].head(5):
        print("TEXT      :", text)
        for name, tok in TOKENIZERS.items():
            print(f"{name:<10}:", tok(text))
        print()

    ch = challenge_table()
    print("=== Tokenization challenges ===")
    for _, row in ch.iterrows():
        print("CASE:", row["case"], "|", row["text"])
        for name in TOKENIZERS:
            print(f"  {name:<10}:", row[name])
        print()

    reg = sample["text"].apply(regex_tokenize)
    nl = sample["text"].apply(nltk_tokenize)
    diff = (reg.apply(tuple) != nl.apply(tuple)).mean() * 100
    print(f"Regex vs NLTK word_tokenize produce different output on {diff:.1f}% of tweets")

    out = sample[["text"]].copy()
    for name, tok in TOKENIZERS.items():
        out[name] = out["text"].apply(tok)
    out.to_csv(OUT_DIR / "tokenization_results.csv", index=False)
    stats_df.to_csv(OUT_DIR / "tokenizer_stats.csv")
    pres.to_csv(OUT_DIR / "pattern_preservation.csv", index=False)

    ax = stats_df[["tokens_per_text"]].plot(kind="bar", legend=False, figsize=(6, 4))
    ax.set_ylabel("Average tokens per tweet")
    ax.set_title("Average tokens per tweet by tokenizer")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "avg_tokens_per_tweet.png", dpi=150)
    plt.close()

    ax = stats_df[["vocab_size"]].plot(kind="bar", legend=False, figsize=(6, 4), color="tab:orange")
    ax.set_ylabel("Vocabulary size (types)")
    ax.set_title("Vocabulary size by tokenizer")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "vocab_size_by_tokenizer.png", dpi=150)
    plt.close()

    print("Saved results to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
