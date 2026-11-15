# US Airline Tweet Sentiment Analysis — NLP Semester Project

**Course:** Introduction to NLP\
**Phase 1 (Assignment 1):** Data Acquisition, Text Processing & Cleaning\
**Phase 2 (Assignment 2):** Word Representations\
**Team name:** 5 Siphahi

| # | Name | Roll No. | GitHub | Role (Phase 1) |
|---|------|----------|--------|----------------|
| 1 | Maham Iqbal | L1F23BSCS0411 | Maham411 | Data loading & dataset analysis |
| 2 | Minahil Malik | L1F23BSCS0431 | minahiltariiqq | Tokenization techniques |
| 3 | Tamazar Fatima | L1F23BSCS1221 | tamazarfatima | Regular expressions & noise removal |
| 4 | Muhammad Hadi | L1F23BSCS0118 | muhammadhadi0594 | Text normalization |
| 5 | Sameer Javed Khan | L1F23BSCS0355 | sameer-javed-khan | Cleaning pipeline & reproducibility |

> **How this README is organised**\
> **Part 1: Assignment 1 (Phase 1)** is in sections 1 to 9.\
> **Part 2: Assignment 2 (Phase 2)** is at the end, in sections 10 to 15.

---

# Part 1: Assignment 1 (Phase 1)

## 1. Project overview

We are building a **3-class sentiment classifier** that predicts whether a tweet written to a US airline is **negative**, **neutral** or **positive**.

Phase 1 turns the raw tweets into a clean, well-documented, analysis-ready corpus using a reproducible cleaning pipeline. Later phases will load the cleaned output directly.

---

## 2. Dataset

| Field | Details |
|-------|---------|
| **Name** | Twitter US Airline Sentiment |
| **Source** | Kaggle — https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| **Original creator** | Figure Eight (formerly CrowdFlower), *Data for Everyone* library |
| **Collected** | February 2015 |
| **Files** | `Tweets.csv` (we use this) and `database.sqlite` (same data, not used) |
| **Size** | 14,640 tweets, 15 columns |
| **Airlines covered** | American, Delta, Southwest, United, US Airways, Virgin America |
| **Key columns** | `text` (raw tweet), `airline_sentiment` (label), `airline_sentiment_confidence`, `negativereason`, `airline`, `tweet_id` |
| **Labels** | `negative`, `neutral`, `positive` (imbalanced: most tweets are negative) |
| **Licence** | CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0) |
| **Prediction target** | `airline_sentiment`: 3-class classification |

**Why this dataset:** it is real, labelled social-media text, which makes it a good fit for this phase. The tweets contain @mentions, #hashtags, URLs, HTML entities (`&amp;`), emojis, elongated words ("sooo"), ALL-CAPS shouting, flight numbers and times. That gives the tokenization, regex and cleaning stages real problems to solve.

**Labels were crowd-sourced.** Each tweet has an `airline_sentiment_confidence` score showing how much the human annotators agreed.

**Citation:**
> Figure Eight / CrowdFlower (2015). *Twitter US Airline Sentiment.* Data for Everyone. Available on Kaggle: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment (CC BY-NC-SA 4.0).

> ⚠️ **The raw dataset is NOT committed to this repository.** Use the download step below to get it.

---

## 3. Repository structure

```
.
├── data/
│   ├── raw/                          # Tweets.csv goes here (git-ignored)
│   └── processed/
│       ├── cleaned_dataset.csv       # pipeline output (committed)
│       └── normalized_tweets.csv
├── notebook/
│   ├── data_analysis.ipynb           # Task 2: exploration
│   ├── tokenization_comparison.ipynb # Task 3: tokenization (to be merged)
│   └── normalization_examples.ipynb  # Task 4: normalization
├── src/
│   ├── load_data.py                  # Task 1: downloads / loads the raw data
│   ├── data_analysis.py              # Task 2: exploration & statistics
│   ├── tokenization.py               # Task 3: tokenizer comparison (to be merged)
│   ├── regex_patterns.py             # Task 3: regular expressions
│   ├── noise_removal.py              # Task 3: noise removal
│   ├── noise_removal_analysis.py
│   ├── noise_removal_examples.py
│   ├── normalization.py              # Task 4: normalization
│   ├── normalization_analysis.py
│   ├── normalization_examples.py
│   ├── cleaning_pipeline.py          # Task 4–5: single reusable cleaning pipeline
│   └── verify_reproducibility.py     # Task 5: fresh run vs committed output check
├── tests/
│   └── test_cleaning_pipeline.py
├── docs/
│   └── REPRODUCIBILITY.md            # fresh clone guide + pipeline details
├── report/
│   ├── figures/                      # graphs produced by the scripts
│   ├── Member_3_Tamazar_Fatima/      # report section (Member 3)
│   ├── Member_5_Sameer_Javed/        # report section (Member 5)
│   ├── stats.json                    # Task 2 statistics
│   ├── messy_examples.txt            # Task 2 messy text examples
│   ├── pipeline_summary.json         # vocab before/after, label counts, output hash
│   └── pipeline_examples.txt         # before/after cleaning examples
├── requirements.txt
├── .gitignore
└── README.md
```

The files added in Assignment 2 are listed in [section 11](#11-new-files-in-assignment-2).

---

## 4. Environment setup

**Requirements:** Python 3.11+

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model and NLTK data
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 5. Getting the data

**Option A — Kaggle API**

1. Create a Kaggle API token: Kaggle → *Settings* → *API* → *Create New Token*. This downloads `kaggle.json`.
2. Place it at `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (macOS / Linux).
3. Run:

```bash
python src/load_data.py
```

**Option B — Manual download**

Download the dataset from the Kaggle link above, unzip it, and place `Tweets.csv` at `data/raw/Tweets.csv`.

---

## 6. How to run

```bash
# Task 1 — download / check the raw data
python src/load_data.py

# Task 2 — exploration & statistics (writes report/figures, report/stats.json, report/messy_examples.txt)
python src/data_analysis.py

# Task 4–5 — full cleaning pipeline (writes data/processed/cleaned_dataset.csv,
#            report/pipeline_summary.json and report/pipeline_examples.txt)
python src/cleaning_pipeline.py

# Tests for the cleaning function
python tests/test_cleaning_pipeline.py

# Check that a fresh run reproduces the committed cleaned_dataset.csv exactly
python src/verify_reproducibility.py
```

Options: `python src/cleaning_pipeline.py --input path/to/Tweets.csv --output path/to/out.csv`

Notebook version of Task 2: open `notebook/data_analysis.ipynb` and run *Kernel → Restart & Run All*.

---

## 7. Cleaning pipeline & reproducibility

`src/cleaning_pipeline.py` holds one function, `clean_text(text)`, that runs every tweet through the Part 3 and Part 4 code in a fixed order:

1. decode HTML entities → 2. Unicode NFC (`normalize_unicode`, Part 4) + straight quotes → 3. case folding → 4. noise removal (`remove_noise`, Part 3) → 5. keep sentiment cues (emojis, `:)` / `:(`, negations, `sooo` → `soo`) → 6. tokenization, stop words and lemmatization with spaCy (`normalize_text`, Part 4)

The cleaned file `data/processed/cleaned_dataset.csv` has one row per raw tweet, the label column `airline_sentiment` copied unchanged, the raw `text` and the new `clean_text`.

| | Before | After |
|---|---|---|
| Vocabulary (types) | 15,089 | 8,961 |
| Tokens | 268,713 | 119,888 |
| Rows / labels | 14,640 | 14,640 (unchanged) |

Step-by-step justification, the output schema and troubleshooting are in **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

---

## 8. Team workflow

This workflow is used for both assignments.

- `main` is protected. Nobody commits to it directly.
- Each member works on their **own branch** named `<name>-<last 4 digits of roll no>`, for example `maham-0411` or `sameer-0355`.
- Before starting new work: `git checkout main` → `git pull` → `git checkout -b <your-branch>`, so the branch starts from the latest `main`.
- Commit small and often with clear messages (e.g. `Add regex for @mentions`), then push the branch and open a **pull request** into `main`.
- Another team member reviews the pull request before it is merged.
- The raw dataset (`data/raw/Tweets.csv`) is never committed. Share it outside Git and place it in `data/raw/`.
- Submissions are tagged on `main`: **`assignment-1`** for Phase 1 and **`assignment-2`** for Phase 2.

---

## 9. Contributions (Assignment 1)

| Member | Contribution |
|--------|--------------|
| Maham Iqbal | Data loading & dataset analysis (Task 1–2): wrote `src/load_data.py` (dataset download/loading) and `src/data_analysis.py` + `notebooks/data_analysis.ipynb` (exploration). Computed dataset statistics (size, missing values, duplicates, tweet length, target distribution, label confidence, airline breakdown), vocabulary size, tokens vs types, Zipf's law / long-tail analysis, and identified messy-text examples. Produced all exploration figures in `report/figures/` and wrote the dataset & exploration sections of the report. |
| Minahil Malik | _to be filled_ |
| Tamazur Fatima | Regular expressions & noise removal (Task 3): wrote src/regex_patterns.py, src/noise_removal.py, src/noise_removal_examples.py, and src/noise_removal_analysis.py. Implemented URL, mention, and number removal; hashtag symbol removal while preserving hashtag words; HTML entity decoding; and whitespace cleanup. Tested the regex patterns and generated noise-removal examples. |
| Muhammad Hadi | Text normalization (Task 4): wrote src/normalization.py, src/normalization_analysis.py, and src/normalization_examples.py, and created notebook/normalization_examples.ipynb. Implemented Unicode NFC normalization, case folding, selective stopword removal with negation preservation, spaCy lemmatization, vocabulary comparison, and normalized dataset generation and verification. |
| Sameer Javed | Cleaning pipeline & reproducibility (Task 4–5): wrote `src/cleaning_pipeline.py` (one reusable `clean_text` function that runs every step in a fixed order and plugs in the Member 3/4 functions), generated `data/processed/cleaned_dataset.csv` with labels preserved, wrote `tests/test_cleaning_pipeline.py` and `src/verify_reproducibility.py` (fresh-run vs committed output check), added `docs/REPRODUCIBILITY.md` and the run instructions in this README, and coordinated the report formatting and screenshots. |

---

# Part 2: Assignment 2 (Phase 2)

## 10. Assignment 2 overview

Phase 2 turns the cleaned tweets from Phase 1 into numbers in several ways and compares them. We continue in the same repository and the submission is tagged **`assignment-2`**.

| Part | Topic |
|------|-------|
| A | Count based representations: A1 Bag of Words, A2 TF-IDF, A3 n-grams, A4 hand worked numerical |
| B | Learned embeddings: Word2Vec (CBOW and Skip-gram) and fastText |
| C | Probing and comparison |
| D | Package for the next phase |

---

## 11. New files in Assignment 2

```
.
├── notebook/
│   └── part_a_count_representations.ipynb   # Part A: BoW, TF-IDF, n-grams, A4
├── src/
│   ├── corpus.py                     # shared corpus used by every part of Assignment 2
│   ├── count_representations.py      # Part A1–A3: Bag of Words, TF-IDF, n-grams
│   ├── hand_worked_a4.py             # Part A4: hand worked TF-IDF and bigram check
│   ├── representation_comparison.py  # Part B3 + C4: CBOW vs Skip-gram, count vs learned
│   ├── save_count_vectorizers.py     # Part D1: saves the BoW and TF-IDF vectorizers once
│   └── vectorize.py                  # Part D2: raw tweet -> vector (Phase 3 interface)
├── tests/
│   └── test_vectorize.py             # Part D2: tests for every available representation
├── models/                           # Part D1: everything Phase 3 loads, saved once
│   ├── bow_vectorizer.joblib         #   fitted CountVectorizer (committed)
│   ├── tfidf_vectorizer.joblib       #   fitted TfidfVectorizer (committed)
│   ├── word2vec_cbow.model           #   Part B1
│   ├── word2vec_skipgram.model       #   Part B2
│   └── fasttext.model                #   Part B4
├── report/
│   ├── figures/
│   │   ├── part_a_top_words.png      # A1: most frequent words
│   │   ├── part_a_idf.png            # A2: lowest and highest IDF words
│   │   ├── part_a_ngram_growth.png   # A3: feature count and sparsity as n grows
│   │   └── c4_word_similarity.png    # C4: word-pair similarity, count-based vs learned
│   └── part_a/
│       ├── a1_bow_stats.json         # A1: shape and sparsity
│       ├── a2_idf_lowest.csv         # A2: most common words (lowest IDF)
│       ├── a2_idf_highest.csv        # A2: rarest words (highest IDF)
│       ├── a2_example_tweet.csv      # A2: TF-IDF weights of one real tweet
│       ├── a3_ngram_growth.csv       # A3: features and sparsity for each n-gram setting
│       ├── a3_top_bigrams.csv
│       ├── a3_top_trigrams.csv
│       ├── part_a_summary.json       # all Part A numbers in one file
│       └── A4_hand_worked_scan.pdf   # A4: handwritten calculation (CamScanner)
│   ├── part_b/
│   │   ├── b3_training_time.csv      # B3: CBOW vs Skip-gram training time, same settings
│   │   └── b3_neighbours.csv         # B3: nearest neighbours, frequent vs rare words
│   └── part_c/
│       ├── c4_dimensionality_sparsity.csv  # C4: size, sparsity, all-zero tweets
│       ├── c4_vocabulary_coverage.csv      # C4: corpus words each model can represent
│       ├── c4_unseen_words.csv             # C4: misspellings / new words (OOV)
│       ├── c4_word_similarity.csv          # C4: what each representation captures
│       └── part_b3_c4_summary.json         # all B3 and C4 numbers in one file
└── requirements.txt                  # adds scikit-learn and gensim
```

---

## 12. Shared corpus

Every part of Assignment 2 uses the **same corpus**, loaded with `src/corpus.py`, so the comparisons are fair. It reads `data/processed/cleaned_dataset.csv` from Phase 1 and the text is **not** cleaned again. Two things are removed, once, for the whole team:

| Step | Rows |
|------|------|
| Rows in `cleaned_dataset.csv` | 14,640 |
| Repeated tweet IDs removed (keep the first copy) | 155 |
| Tweets that became empty after cleaning removed | 56 |
| **Documents used in Assignment 2** | **14,429** |

```python
from corpus import load_corpus, documents, tokenized
df = load_corpus()
docs = documents(df)      # strings, for scikit-learn (Part A)
sents = tokenized(df)     # token lists, for gensim Word2Vec / fastText (Part B)
```

`corpus.TOKEN_PATTERN` (`(?u)\S+`) makes scikit-learn split on spaces only, so it uses exactly the tokens made by the Phase 1 pipeline.

---

## 13. Setup and how to run (Assignment 2)

Install the new packages (scikit-learn and gensim were added to `requirements.txt`):

```bash
pip install -r requirements.txt
```

The cleaned dataset is already in the repository, so `Tweets.csv` is **not** needed for Assignment 2.

```bash
# Shared corpus (prints a summary)
python src/corpus.py

# Part A1–A3: Bag of Words, TF-IDF and n-grams
# (writes report/part_a/*.csv, report/part_a/part_a_summary.json and report/figures/part_a_*.png)
python src/count_representations.py

# Part A4: hand worked TF-IDF weight and bigram (class method), checked against scikit-learn
python src/hand_worked_a4.py

# Part B3 + C4: CBOW vs Skip-gram and count-based vs learned representations
# (writes report/part_b/b3_*.csv, report/part_c/c4_*.csv and report/figures/c4_word_similarity.png)
python src/representation_comparison.py

# Part D1: save the fitted BoW and TF-IDF vectorizers to models/ (already committed, so
# only needed if the corpus changes)
python src/save_count_vectorizers.py

# Part D2: turn raw tweets into vectors, then test it with every available representation
python src/vectorize.py
python tests/test_vectorize.py
```

**Part D2, the Phase 3 interface.** Give it a raw tweet and the name of a representation:

```python
from vectorize import vectorize, vectorize_many, available_representations

available_representations()          # ['bow', 'tfidf', 'cbow', 'skipgram', 'fasttext']
vec = vectorize("@united my bag is lost :(", "fasttext")       # numpy array, shape (100,)
X = vectorize_many(list_of_raw_tweets, "tfidf")                # sparse matrix for bow / tfidf
X = vectorize_many(df["clean_text"], "cbow", already_clean=True)  # skip re-cleaning the corpus
```

The raw tweet goes through the same Phase 1 cleaning (`clean_text`) as the training corpus. Nothing is trained or refitted when vectorizing: `bow` and `tfidf` load the vectorizers saved by `src/save_count_vectorizers.py` (`models/bow_vectorizer.joblib`, `models/tfidf_vectorizer.joblib`), and `cbow`, `skipgram` and `fasttext` load `models/word2vec_cbow.model` (or `cbow.model`), `models/word2vec_skipgram.model` (or `skipgram.model`) and `models/fasttext.model`. A representation is listed as available only once its file is in `models/`. A tweet vector under a learned model is the mean of its word vectors; words the model has never seen are skipped by Word2Vec, while fastText builds them from character n-grams.

Notebook version of Part A: open `notebook/part_a_count_representations.ipynb` and run *Kernel → Restart Kernel and Run All Cells*.

---

## 14. Work division and results

| Part | Member | Files |
|------|--------|-------|
| A | **Maham Iqbal** (L1F23BSCS0411) | `src/corpus.py`, `src/count_representations.py`, `src/hand_worked_a4.py`, `notebook/part_a_count_representations.ipynb` |
| B3, C4, D1 (BoW / TF-IDF), D2 | **Sameer Javed Khan** (L1F23BSCS0355) | `src/representation_comparison.py`, `src/save_count_vectorizers.py`, `src/vectorize.py`, `tests/test_vectorize.py`, `models/*_vectorizer.joblib` |
| B | _to be filled_ | |
| C | _to be filled_ | |
| D | _to be filled_ | |

### Part A results

| Task | Result |
|------|--------|
| A1 Bag of Words | 14,429 × 8,961 matrix, 99.912% of cells are zero (1.42 MB sparse vs about 1,034 MB as a full table) |
| A2 TF-IDF | Same shape and vocabulary. Most common words: *flight* (idf 2.32) and *not* (idf 2.44). 4,591 words appear in only one tweet (idf 9.88) |
| A3 N-grams | 8,961 unigrams → 74,529 with bigrams → 157,601 with trigrams. 88% of these appear in only one tweet. min_df = 2 keeps 18,598 |
| A4 Hand worked | TF-IDF(hawaii, D1) = 1/3 × log₁₀(4) = 0.200687 and P(flight \| book) = 2 / 2 = 1.0, both matching the library |

---

## 15. Contributions (Assignment 2)

| Member | Contribution |
|--------|--------------|
| Maham Iqbal | Part A, count based representations (A1 to A4): wrote `src/corpus.py` (the shared Assignment 2 corpus that every part uses: removes repeated tweet IDs and empty tweets), `src/count_representations.py` (A1 Bag of Words with shape and sparsity, A2 TF-IDF with IDF tables and a worked example tweet, A3 unigram / bigram / trigram feature growth and sparsity with a min_df comparison) and `src/hand_worked_a4.py` (A4 TF-IDF weight and bigram probability with the class method, checked against scikit-learn). Built `notebook/part_a_count_representations.ipynb`, produced the Part A tables in `report/part_a/` and the figures `report/figures/part_a_*.png`, added scikit-learn and gensim to `requirements.txt` and did the A4 calculation by hand (`report/part_a/A4_hand_worked_scan.pdf`). |
| Minahil Malik | _to be filled_ |
| Tamazur Fatima | _to be filled_ |
| Hadi Faisal | _to be filled_ |
| Sameer Javed | B3, C4 and D2: wrote `src/representation_comparison.py` (B3 CBOW vs Skip-gram: training time under identical settings and nearest neighbours for frequent vs rare words; C4 count-based vs learned representations: dimensionality, sparsity, vocabulary coverage, misspelled and unseen words, and word-pair similarity with `report/figures/c4_word_similarity.png`), `src/save_count_vectorizers.py` (D1 for the count-based side: saves the fitted BoW and TF-IDF vectorizers to `models/` once), `src/vectorize.py` (D2 Phase 3 interface: `vectorize()` takes a raw tweet, runs the Phase 1 cleaning and returns its vector under `bow`, `tfidf`, `cbow`, `skipgram` or `fasttext`, loading saved files only; `vectorize_many()` for batches) and `tests/test_vectorize.py` (11 tests run against every available representation). |
