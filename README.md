# US Airline Tweet Sentiment Analysis — NLP Semester Project

**Course:** Introduction to NLP
**Phase:** 1 — Data Acquisition, Text Processing & Cleaning
**Team name:** 5 Siphahi

| # | Name | Roll No. | GitHub | Role (Phase 1) |
|---|------|----------|--------|----------------|
| 1 | Maham Iqbal | L1F23BSCS0411 | Maham411 | Data loading & dataset analysis |
| 2 | Minahil Malik | L1F23BSCS0431 | minahiltariiqq | Tokenization techniques |
| 3 | Tamazur Fatima | L1F23BSCS1221 | tamazarfatima | Regular expressions & noise removal |
| 4 | Hadi Faisal | L1F23BSCS0118 | muhammadhadi0594 | Text normalization |
| 5 | Sameer Javed | L1F23BSCS0355 | sameer-javed-khan | Cleaning pipeline & reproducibility |

---

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
│   ├── raw/                  # Tweets.csv goes here (git-ignored)
│   └── processed/            # cleaned output from the pipeline
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── tokenization_comparison.ipynb
│   └── normalization_examples.ipynb
├── src/
│   ├── load_data.py          # downloads / loads the raw data
│   ├── data_analysis.py      # Task 2: exploration & statistics
│   ├── tokenization.py
│   ├── regex_patterns.py
│   ├── noise_removal.py
│   ├── normalization.py
│   └── cleaning_pipeline.py  # single reusable cleaning pipeline
├── report/
│   ├── figures/              # graphs produced by the scripts
│   └── ...                   # PDF report + screenshots
├── requirements.txt
├── .gitignore
└── README.md
```

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
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
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

# Tasks 3–5 — to be completed
python src/cleaning_pipeline.py   # → data/processed/cleaned_tweets.csv
```

Notebook version of Task 2: open `notebooks/data_analysis.ipynb` and run *Kernel → Restart & Run All*.

---

## 7. Team workflow

- `main` is protected. Nobody commits to it directly.
- Each member works on their own branch (e.g. `feature/data-analysis`) and opens a **pull request** to merge.
- Commit small and often, with clear messages (e.g. `Add regex for @mentions`).
- The final Phase 1 submission is tagged **`assignment-1`**.

---

## 8. Contributions

| Member | Contribution |
|--------|--------------|
| Maham Iqbal | | Maham Iqbal | Data loading & dataset analysis (Task 1–2): wrote `src/load_data.py` (dataset download/loading) and `src/data_analysis.py` + `notebooks/data_analysis.ipynb` (exploration). Computed dataset statistics (size, missing values, duplicates, tweet length, target distribution, label confidence, airline breakdown), vocabulary size, tokens vs types, Zipf's law / long-tail analysis, and identified messy-text examples. Produced all exploration figures in `report/figures/` and wrote the dataset & exploration sections of the report. ||
| Minahil Malik | _to be filled_ |
| Tamazur Fatima | _to be filled_ |
| Hadi Faisal | _to be filled_ |
| Sameer Javed | _to be filled_ |
