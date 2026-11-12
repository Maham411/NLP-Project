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
python -m spacy download en_core_web_sm
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

The pipeline takes about a minute because spaCy processes every tweet. Expected end of `verify_reproducibility.py`:

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

`clean_text(text)` in `src/cleaning_pipeline.py` is the one function that cleans a tweet. It calls the functions written in Part 3 (`src/noise_removal.py`) and Part 4 (`src/normalization.py`) and adds three small steps of its own around them.

| # | Step | Code | Why here |
|---|------|------|----------|
| 1 | Decode HTML entities (`&amp;` → `&`, `&lt;3` → `<3`) | `cleaning_pipeline.decode_entities` | 720 tweets store escaped HTML. It is undone first so `amp` and `lt` never become words and `&lt;3` can be read as a heart. |
| 2 | Unicode NFC, then curly quotes → straight quotes | `normalization.normalize_unicode` (Part 4) + `cleaning_pipeline.standard_quotes` | Must run before any regex or tokenizer so `café` / `cafe´` match. Straight quotes matter because spaCy only turns `n't` into `not` when the apostrophe is straight: without this, `didn’t like it` became `like`. |
| 3 | Case folding (`casefold`) | `cleaning_pipeline.case_fold` | Capitalisation in tweets is inconsistent (`DELAYED`, `Delayed`), which splits one word into several types. Matches the choice made in Part 4. |
| 4 | Noise removal | `noise_removal.remove_noise` (Part 3) | Removes URLs, @mentions and numbers, keeps the hashtag word without `#`, squeezes whitespace. URLs must go before step 5, or `http://` would be read as the emoticon `:/`. |
| 5 | Keep sentiment cues | `cleaning_pipeline.keep_sentiment_cues` | Emojis become their names (`👍` → `thumbs up`), `:)` / `:(` become `happyface` / `sadface`, `dont` / `cant` / `won't` become `do not` / `can not` / `will not`, and `sooooo` → `soo`. spaCy would otherwise drop emojis and emoticons, which are strong sentiment signals. |
| 6 | Tokenization, stop-word removal, lemmatization | `normalization.normalize_text` (Part 4) | spaCy `en_core_web_sm` tokenizes, drops stop words and punctuation but keeps negations (`not`, `no`, `never`, …) and intensifiers (`very`, `really`, `but`, …), and lemmatizes (`flights` → `flight`, `delayed` → `delay`). |

## 3. Output

`data/processed/cleaned_dataset.csv`, one row per raw tweet (14,640 rows, same order as `Tweets.csv`):

| Column | Meaning |
|--------|---------|
| `tweet_id` | original tweet id |
| `airline` | airline the tweet is addressed to |
| `airline_sentiment` | **label**, copied unchanged from the raw data |
| `airline_sentiment_confidence` | annotator agreement (rounded to 4 decimals, as in the source) |
| `negativereason` | reason given for negative tweets (empty otherwise) |
| `text` | raw tweet, untouched |
| `clean_text` | output of the pipeline, tokens joined by single spaces |
| `n_tokens` | number of tokens in `clean_text` |

No rows are dropped. Duplicates (36 exact duplicate rows found in Part 1) and the 57 tweets that become empty after cleaning are left in so the file lines up 1:1 with the raw data; Phase 2 decides how to split and de-duplicate.

Phase 2 can load it directly:

```python
import pandas as pd
df = pd.read_csv("data/processed/cleaned_dataset.csv", keep_default_na=False)
X, y = df["clean_text"], df["airline_sentiment"]
```

The run also writes:

- `report/pipeline_summary.json`: row counts, label counts, vocabulary before/after, library versions, SHA-256 of the output
- `report/pipeline_examples.txt`: before/after examples for the report

## 4. Results of the committed run

| | Before cleaning | After cleaning |
|---|---|---|
| Tweets | 14,640 | 14,640 |
| Tokens | 268,713 | 119,888 |
| Vocabulary (types) | 15,089 | 8,961 (−40.6%) |
| negative / neutral / positive | 9,178 / 3,099 / 2,363 | 9,178 / 3,099 / 2,363 |
| Empty after cleaning | 0 | 57 |

"Before" is counted with `simple_tokenize` from `src/data_analysis.py` (Part 1), so both numbers use the same counting rule.

SHA-256 of the committed CSV: `af10013ff447a66431f309c825db528e0db6b9689f68369bd935a50998e15335`

## 5. Things that can break reproducibility

- **Different raw file.** The run must use the original Kaggle `Tweets.csv` (14,640 rows). Some copies online were re-saved in Excel and have rounded tweet ids; those fail the check.
- **spaCy model version.** Lemmas and stop words come from `en_core_web_sm`. The committed CSV was produced with spaCy 3.8 and `en_core_web_sm` 3.8.0. A different model version can lemmatize a few words differently. `report/pipeline_summary.json` records the exact versions used.
- **Other library versions.** Also produced with pandas 3.0.5 and emoji 2.16.0. `airline_sentiment_confidence` is rounded to 4 decimals because some pandas versions parse a few values with float noise (0.352 → 0.35200000000000004). If the check fails, `verify_reproducibility.py` lists the tweets that differ.
- **Line endings.** `.gitattributes` keeps CSV files as LF on Windows, and the verify script ignores `\r\n` anyway.
- **Editing the CSV by hand.** Don't. Always regenerate it with the script.
