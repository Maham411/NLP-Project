import argparse
import hashlib
import html
import importlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import emoji
import nltk
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from data_analysis import simple_tokenize
from load_data import LABEL_COL, RAW_FILE, ROOT, TEXT_COL, load_raw

OUTPUT_FILE = ROOT / "data" / "processed" / "cleaned_dataset.csv"
SUMMARY_FILE = ROOT / "report" / "pipeline_summary.json"
EXAMPLES_FILE = ROOT / "report" / "pipeline_examples.txt"

CLEAN_COL = "clean_text"
KEEP_COLUMNS = [
    "tweet_id",
    "airline",
    LABEL_COL,
    "airline_sentiment_confidence",
    "negativereason",
    TEXT_COL,
]

NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}

QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
})

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUMBER_RE = re.compile(r"[$£€]?\d+(?:[.,:/-]\d+)*(?:am|pm|st|nd|rd|th|hrs?|mins?|k|s)?\b")
HAPPY_RE = re.compile(r"(?<!\w)(?:[:;=]-?(?:\)+|\]|d\b|p\b)|\(+-?[:;=]|<3+)")
SAD_RE = re.compile(r"(?<!\w)(?:[:;=]'?-?(?:\(+|\[|/(?!/)|\\|\|))")
SKIN_TONE_RE = re.compile(r"_(?:medium-light|medium-dark|light|medium|dark)_skin_tone")
ELONGATED_RE = re.compile(r"([a-z])\1{2,}")
REPEATED_MARK_RE = re.compile(r"([!?])\1+")
NOT_ALLOWED_RE = re.compile(r"[^a-z_!?\s]")
SPACES_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z]+(?:_[a-z]+)*|[!?]")

NEGATION_FIXES = [
    (re.compile(r"\bwon'?t\b"), "will not"),
    (re.compile(r"\bcan'?t\b|\bcannot\b"), "can not"),
    (re.compile(r"\bain'?t\b"), "is not"),
    (re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should|must|need)n'?t\b"), r"\1 not"),
    (re.compile(r"n't\b"), " not"),
]
SUFFIX_RE = re.compile(r"'(?:s|re|ll|ve|m|d)\b")

KEEP_STOPWORDS = {
    "no", "not", "nor", "never", "against", "but",
    "very", "too", "more", "most", "off", "out", "down",
}


def ensure_nltk_data():
    for path, name in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def decode_entities(text):
    return html.unescape(str(text))


def _normalize_unicode(text):
    return unicodedata.normalize("NFC", text).translate(QUOTE_MAP)


def _case_fold(text):
    return text.lower()


def _emoji_to_word(chars, data):
    name = SKIN_TONE_RE.sub("", data["en"].strip(":").lower())
    name = re.sub(r"[^a-z]+", "_", name).strip("_")
    return f" emoji_{name} "


def _remove_noise(text):
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=_emoji_to_word)
    text = HAPPY_RE.sub(" happyface ", text)
    text = SAD_RE.sub(" sadface ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = NUMBER_RE.sub(" ", text)
    for pattern, replacement in NEGATION_FIXES:
        text = pattern.sub(replacement, text)
    text = SUFFIX_RE.sub("", text)
    text = ELONGATED_RE.sub(r"\1\1", text)
    text = REPEATED_MARK_RE.sub(r"\1", text)
    text = NOT_ALLOWED_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text):
    return TOKEN_RE.findall(text)


_stopwords = None
_lemmatizer = None


def _remove_stopwords(tokens):
    global _stopwords
    if _stopwords is None:
        from nltk.corpus import stopwords
        _stopwords = set(stopwords.words("english")) - KEEP_STOPWORDS
    return [t for t in tokens if t not in _stopwords and (len(t) > 1 or t in "!?")]


def _lemmatize_tokens(tokens):
    global _lemmatizer
    if _lemmatizer is None:
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    out = []
    for tok in tokens:
        if "_" in tok or tok in ("!", "?", "happyface", "sadface"):
            out.append(tok)
            continue
        lemma = _lemmatizer.lemmatize(tok, "v")
        if lemma == tok and not tok.endswith("ss"):
            lemma = _lemmatizer.lemmatize(tok, "n")
        out.append(lemma)
    return out


def _pick(module_name, func_name, fallback):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, func_name), f"{module_name}.{func_name}"
    except (ImportError, AttributeError):
        return fallback, "cleaning_pipeline (fallback)"


normalize_unicode, _src_unicode = _pick("normalization", "normalize_unicode", _normalize_unicode)
case_fold, _src_case = _pick("normalization", "case_fold", _case_fold)
remove_noise, _src_noise = _pick("noise_removal", "remove_noise", _remove_noise)
remove_stopwords, _src_stop = _pick("normalization", "remove_stopwords", _remove_stopwords)
lemmatize_tokens, _src_lemma = _pick("normalization", "lemmatize_tokens", _lemmatize_tokens)

PIPELINE_STEPS = [
    ("1. decode HTML entities", "cleaning_pipeline"),
    ("2. unicode normalization (NFC)", _src_unicode),
    ("3. case folding", _src_case),
    ("4. noise removal", _src_noise),
    ("5. tokenization", "cleaning_pipeline"),
    ("6. stop-word removal (negations kept)", _src_stop),
    ("7. lemmatization", _src_lemma),
]


def clean_tokens(text):
    text = decode_entities(text)
    text = normalize_unicode(text)
    text = case_fold(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)


def clean_text(text):
    return " ".join(clean_tokens(text))


def build_clean_dataset(raw):
    ensure_nltk_data()
    missing = [c for c in KEEP_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")

    df = raw[KEEP_COLUMNS].copy().reset_index(drop=True)
    df["airline_sentiment_confidence"] = df["airline_sentiment_confidence"].astype(float).round(4)
    df[CLEAN_COL] = df[TEXT_COL].map(clean_text)
    df["n_tokens"] = df[CLEAN_COL].str.split().str.len().fillna(0).astype(int)

    if len(df) != len(raw) or not df[LABEL_COL].equals(raw[LABEL_COL].reset_index(drop=True)):
        raise RuntimeError("labels changed during cleaning")
    return df


def to_csv_text(df):
    return df.to_csv(index=False, lineterminator="\n")


def save_dataset(df, path=OUTPUT_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = to_csv_text(df)
    path.write_bytes(content.encode("utf-8"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def vocab_stats(token_lists):
    counts = Counter(t for toks in token_lists for t in toks)
    return {"tokens": int(sum(counts.values())), "types": int(len(counts))}


def summarize(raw, df, sha):
    before = vocab_stats(raw[TEXT_COL].map(simple_tokenize))
    after = vocab_stats(df[CLEAN_COL].str.split())
    drop = round(100 * (1 - after["types"] / before["types"]), 2)
    return {
        "rows_in": int(len(raw)),
        "rows_out": int(len(df)),
        "labels_preserved": bool(df[LABEL_COL].tolist() == raw[LABEL_COL].tolist()),
        "label_counts": {k: int(v) for k, v in df[LABEL_COL].value_counts().items()},
        "empty_after_cleaning": int((df["n_tokens"] == 0).sum()),
        "avg_tokens_after": round(float(df["n_tokens"].mean()), 2),
        "vocabulary_before": before,
        "vocabulary_after": after,
        "vocabulary_reduction_percent": drop,
        "steps": {name: src for name, src in PIPELINE_STEPS},
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha,
    }


def write_examples(df, path=EXAMPLES_FILE, n=12, seed=42):
    picks = pd.concat([
        df[df[TEXT_COL].map(lambda s: re.search(p, s) is not None)].sample(1, random_state=seed)
        for p in [r"http", r"&amp;", r"n't|nt\b", r"[:;]-?[)(DPp]", r"(\w)\1{2,}", r"#\w+", r"[^\x00-\x7f]"]
    ])
    rest = df.drop(picks.index).sample(n - len(picks), random_state=seed)
    lines = ["BEFORE / AFTER: full cleaning pipeline", "=" * 40, ""]
    for _, row in pd.concat([picks, rest]).iterrows():
        lines += [
            f"label : {row[LABEL_COL]}",
            f"before: {row[TEXT_COL]!r}",
            f"after : {row[CLEAN_COL]!r}",
            "",
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def load_input(path):
    if path is None:
        return load_raw()
    return pd.read_csv(path, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clean the airline tweets and save them as CSV.")
    parser.add_argument("--input", default=None, help=f"raw CSV (default: {RAW_FILE.relative_to(ROOT)})")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="where to write the cleaned CSV")
    args = parser.parse_args(argv)

    print("Cleaning pipeline steps:")
    for name, src in PIPELINE_STEPS:
        print(f"  {name:<40} <- {src}")

    raw = load_input(args.input)
    print(f"\nLoaded {len(raw):,} raw tweets")
    df = build_clean_dataset(raw)
    sha = save_dataset(df, args.output)
    shown = Path(args.output).resolve()
    shown = shown.relative_to(ROOT) if shown.is_relative_to(ROOT) else shown
    print(f"Saved {len(df):,} rows -> {shown}")

    summary = summarize(raw, df, sha)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_examples(df)

    b, a = summary["vocabulary_before"], summary["vocabulary_after"]
    print(f"Labels preserved: {summary['labels_preserved']}  {summary['label_counts']}")
    print(f"Vocabulary: {b['types']:,} types before -> {a['types']:,} after "
          f"({summary['vocabulary_reduction_percent']}% smaller)")
    print(f"Tokens:     {b['tokens']:,} before -> {a['tokens']:,} after")
    print(f"Empty tweets after cleaning: {summary['empty_after_cleaning']}")
    print(f"SHA-256: {sha}")
    print(f"\nSummary  -> {SUMMARY_FILE.relative_to(ROOT)}")
    print(f"Examples -> {EXAMPLES_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
