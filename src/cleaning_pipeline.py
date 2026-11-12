import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import emoji
import pandas as pd
import spacy

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from data_analysis import simple_tokenize
from load_data import LABEL_COL, RAW_FILE, ROOT, TEXT_COL, load_raw
from noise_removal import remove_noise
from normalization import nlp, normalize_text, normalize_unicode

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

QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
})

HAPPY_RE = re.compile(r"[:;=]-?\)+|[:;=]-?\]|(?<!\w)[:;=]-?[dp]\b|\(+-?[:;=]|<3+")
SAD_RE = re.compile(r"[:;=]'?-?(?:\(+|\[|\\|\|)|(?<!\w)[:;=]'?-?/(?!/)")
SKIN_TONE_RE = re.compile(r"_(?:medium-light|medium-dark|light|medium|dark)_skin_tone")
ELONGATED_RE = re.compile(r"([a-z])\1{2,}")
NEGATION_FIXES = [
    (re.compile(r"\bwon'?t\b"), "will not"),
    (re.compile(r"\bcan'?t\b|\bcannot\b"), "can not"),
    (re.compile(r"\bain'?t\b"), "is not"),
    (re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should|must|need)n'?t\b"), r"\1 not"),
    (re.compile(r"n't\b"), " not"),
]

PIPELINE_STEPS = [
    ("1. decode HTML entities", "cleaning_pipeline"),
    ("2. unicode NFC + standard quotes", "normalization.normalize_unicode (Part 4)"),
    ("3. case folding", "cleaning_pipeline"),
    ("4. noise removal (URLs, mentions, numbers)", "noise_removal.remove_noise (Part 3)"),
    ("5. keep sentiment cues", "cleaning_pipeline"),
    ("6. tokenize + stop words + lemmatize", "normalization.normalize_text (Part 4)"),
]


def decode_entities(text):
    return html.unescape(str(text))


def standard_quotes(text):
    return text.translate(QUOTE_MAP)


def case_fold(text):
    return text.casefold()


def _emoji_to_words(chars, data):
    name = SKIN_TONE_RE.sub("", data["en"].strip(":").lower())
    return " " + re.sub(r"[^a-z]+", " ", name).strip() + " "


def keep_sentiment_cues(text):
    text = emoji.replace_emoji(text, replace=_emoji_to_words)
    text = HAPPY_RE.sub(" happyface ", text)
    text = SAD_RE.sub(" sadface ", text)
    for pattern, replacement in NEGATION_FIXES:
        text = pattern.sub(replacement, text)
    text = ELONGATED_RE.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    text = decode_entities(text)
    text = standard_quotes(normalize_unicode(text))
    text = case_fold(text)
    text = remove_noise(text)
    text = keep_sentiment_cues(text)
    return normalize_text(text)


def build_clean_dataset(raw):
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
        "versions": {
            "pandas": pd.__version__,
            "spacy": spacy.__version__,
            "spacy_model": f"{nlp.meta['name']} {nlp.meta['version']}",
            "emoji": emoji.__version__,
        },
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha,
    }


def write_examples(df, path=EXAMPLES_FILE, n=12, seed=42):
    picks = pd.concat([
        df[df[TEXT_COL].map(lambda s: re.search(p, s) is not None)].sample(1, random_state=seed)
        for p in [r"http", r"&amp;", r"n't|n’t", r"[:;]-?[)(DPp]", r"(\w)\1{2,}", r"#\w+", r"[\U0001F300-\U0001FAFF☀-➿]"]
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
        print(f"  {name:<44} <- {src}")

    raw = load_input(args.input)
    print(f"\nLoaded {len(raw):,} raw tweets (spaCy takes about a minute)")
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
