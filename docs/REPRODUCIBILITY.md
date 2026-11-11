# Reproducibility guide (Phase 1)

This page explains how anyone can go from a fresh clone to the exact same `data/processed/cleaned_dataset.csv` that is committed in the repo, and how to prove it.

## 1. From a fresh clone to the cleaned CSV

```bash
git clone https://github.com/Maham411/NLP-Project.git
cd NLP-Project

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ('stopwords', 'wordnet', 'omw-1.4')]"
```

Put the raw data in place (either option):

- `python src/load_data.py` (needs a Kaggle API token, see README section 5), or
- download `Tweets.csv` from Kaggle by hand and save it as `data/raw/Tweets.csv`.

Then run the pipeline, the tests and the reproducibility check:

```bash
python src/cleaning_pipeline.py
python tests/test_cleaning_pipeline.py
python src/verify_reproducibility.py
```

Expected end of `verify_reproducibility.py`:

```
  [PASS] row count matches raw data
  [PASS] labels identical to raw data
  [PASS] labels identical to committed CSV
  [PASS] tweet ids in the same order
  [PASS] no missing labels
  [PASS] byte-identical output (SHA-256)

REPRODUCIBLE
```

## 2. What the pipeline does, in order

`clean_text(text)` in `src/cleaning_pipeline.py` is the one function that cleans a tweet. Every row goes through the same steps in the same order:

| # | Step | Why here |
|---|------|----------|
| 1 | Decode HTML entities (`&amp;` → `&`, `&lt;3` → `<3`) | The dataset stores 720 tweets with escaped HTML. It must be undone first, otherwise `amp` and `lt` end up as words. |
| 2 | Unicode normalization (NFC) + curly quotes → straight quotes | Has to happen before any regex or tokenizer so that `café` / `cafe´` and `won’t` / `won't` look identical. |
| 3 | Case folding (lowercase) | Capitalisation in tweets is inconsistent (`DELAYED`, `Delayed`, `delayed`). Shouting is partly kept through `!`, emojis and emoticons. |
| 4 | Noise removal | URLs, @mentions and numbers/times are removed; `#` is dropped but the hashtag word is kept; emojis and `:)` / `:(` become words; `don't` / `dont` → `do not`; `sooo` → `soo`; `!!!` → `!`. |
| 5 | Tokenization | Regex `[a-z]+(?:_[a-z]+)*|[!?]`, so emoji words like `emoji_red_heart` stay one token and `!` / `?` survive. |
| 6 | Stop-word removal | NLTK English list **minus** negations and sentiment-bearing words (`no, not, nor, never, against, but, very, too, more, most, off, out, down`). |
| 7 | Lemmatization (WordNet) | `flights → flight`, `delayed → delay`, `cancelled → cancel`. Chosen over stemming because the tokens stay readable for error analysis in Phase 2. |

## 3. Output

`data/processed/cleaned_dataset.csv`, one row per raw tweet (14,640 rows, same order as `Tweets.csv`):

| Column | Meaning |
|--------|---------|
| `tweet_id` | original tweet id |
| `airline` | airline the tweet is addressed to |
| `airline_sentiment` | **label**, copied unchanged from the raw data |
| `airline_sentiment_confidence` | annotator agreement |
| `negativereason` | reason given for negative tweets (empty otherwise) |
| `text` | raw tweet, untouched |
| `clean_text` | output of the pipeline, tokens joined by single spaces |
| `n_tokens` | number of tokens in `clean_text` |

No rows are dropped. Duplicates (36 exact duplicate rows found in Part 1) and the 11 tweets that become empty after cleaning are left in so the file lines up 1:1 with the raw data; Phase 2 decides how to split and de-duplicate.

Phase 2 can load it directly:

```python
import pandas as pd
df = pd.read_csv("data/processed/cleaned_dataset.csv", keep_default_na=False)
X, y = df["clean_text"], df["airline_sentiment"]
```

The run also writes:

- `report/pipeline_summary.json` – row counts, label counts, vocabulary before/after, SHA-256 of the output
- `report/pipeline_examples.txt` – before/after examples for the report

## 4. Results of the committed run

| | Before cleaning | After cleaning |
|---|---|---|
| Tweets | 14,640 | 14,640 |
| Tokens | 268,713 | 142,537 |
| Vocabulary (types) | 15,089 | 8,987 (−40.4%) |
| negative / neutral / positive | 9,178 / 3,099 / 2,363 | 9,178 / 3,099 / 2,363 |

"Before" is counted with `simple_tokenize` from `src/data_analysis.py` (Part 1), so both numbers use the same counting rule.

SHA-256 of the committed CSV: `74dc89589c75e440b3167a8a9d3cf6742f652ce72a8647d58a449a76eefaac04`

## 5. How the teammates' functions plug in

`cleaning_pipeline.py` looks for these functions first and only uses its own built-in versions when a module or function is missing. The run prints which one was used for every step.

| Step | Module (owner) | Function it looks for |
|------|----------------|-----------------------|
| Unicode normalization | `src/normalization.py` (Member 4) | `normalize_unicode(text: str) -> str` |
| Case folding | `src/normalization.py` (Member 4) | `case_fold(text: str) -> str` |
| Noise removal | `src/noise_removal.py` (Member 3) | `remove_noise(text: str) -> str` |
| Stop-word removal | `src/normalization.py` (Member 4) | `remove_stopwords(tokens: list[str]) -> list[str]` |
| Lemmatization / stemming | `src/normalization.py` (Member 4) | `lemmatize_tokens(tokens: list[str]) -> list[str]` |

After Members 3 and 4 merge their work, re-run `python src/cleaning_pipeline.py`, commit the new `cleaned_dataset.csv` and `report/pipeline_summary.json`, and run `verify_reproducibility.py` again.

## 6. Things that can break reproducibility

- **Different raw file.** The run must use the original Kaggle `Tweets.csv` (14,640 rows). Some copies on the internet were re-saved in Excel and have rounded tweet ids; those will fail the check.
- **Library versions.** The committed CSV was produced with Python 3.13, pandas 3.0.5, nltk 3.10.4, emoji 2.16.0, and gives the same bytes with pandas 2.3 and 3.1. `airline_sentiment_confidence` is rounded to 4 decimals because some pandas versions parse a few of those values with float noise (0.352 → 0.35200000000000004). Newer emoji releases can rename a few emojis; newer NLTK data can change the stop-word list. If the check fails, `verify_reproducibility.py` lists the tweets that differ.
- **Line endings.** `.gitattributes` keeps CSV files as LF on Windows, and the verify script ignores `\r\n` anyway.
- **Editing the CSV by hand.** Don't. Always regenerate it with the script.
