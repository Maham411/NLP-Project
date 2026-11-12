# IMDB Movie Review Sentiment Analysis — NLP Semester Project

**Course:** Introduction to NLP
**Phase:** 1 — Data Acquisition, Text Processing & Cleaning
**Team name:** _<Team Name>_

| # | Name | Roll No. | GitHub | Role (Phase 1) |
|---|------|----------|--------|----------------|
| 1 | Maham Iqbal | L1F23BSCS0411 | Maham411 | Data loading & dataset analysis |
| 2 | Minahil Malik | L1F23BSCS0431 | minahiltariiqq | Tokenization techniques |
| 3 | _<Name>_ | _<Roll No.>_ | @_<username>_ | Regular expressions & noise removal |
| 4 | _<Name>_ | _<Roll No.>_ | @_<username>_ | Text normalization |
| 5 | _<Name>_ | _<Roll No.>_ | @_<username>_ | Cleaning pipeline & reproducibility |

---

## 1. Project overview

We are building a **binary sentiment classifier** that predicts whether an IMDB movie review is **positive** or **negative**.

Phase 1 turns the raw reviews into a clean, well-documented, analysis-ready corpus using a reproducible cleaning pipeline. Later phases will load the cleaned output directly.

---

## 2. Dataset

| Field | Details |
|-------|---------|
| **Name** | IMDB Dataset of 50K Movie Reviews |
| **Source** | Kaggle — https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews |
| **Original source** | Stanford AI Lab, Large Movie Review Dataset — http://ai.stanford.edu/~amaas/data/sentiment/ |
| **File** | `IMDB Dataset.csv` (≈ 66 MB) |
| **Size** | 50,000 reviews (49,582 unique — 418 duplicates) |
| **Columns** | `review` (raw review text), `sentiment` (label) |
| **Labels** | `positive`, `negative` (balanced, 25,000 each) |
| **Licence** | Other (as listed on Kaggle) — data originates from the Stanford Large Movie Review Dataset; use requires citing the original paper below. |
| **Prediction target** | `sentiment` — binary classification (positive vs. negative) |

**Why this dataset:** it is large, labelled, balanced, and contains real messy text, including HTML line breaks (`<br /><br />`), inconsistent spacing and punctuation, mixed case, accented characters, numbers, prices and ratings. That gives the tokenization and cleaning stages real problems to solve.

**Citation:**
> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis.* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

> ⚠️ **The raw dataset is NOT committed to this repository.** Use the download step below to get it.

---

## 3. Repository structure (planned)

```
.
├── data/
│   ├── raw/                  # raw CSV goes here (git-ignored)
│   └── processed/            # cleaned output from the pipeline
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── tokenization_comparison.ipynb
│   └── normalization_examples.ipynb
├── src/
│   ├── load_data.py          # downloads / loads the raw data
│   ├── data_analysis.py
│   ├── tokenization.py
│   ├── regex_patterns.py
│   ├── noise_removal.py
│   ├── normalization.py
│   └── cleaning_pipeline.py  # single reusable cleaning pipeline
├── report/                   # PDF report + screenshots
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

**Option A — Kaggle API (recommended)**

1. Create a Kaggle API token: Kaggle → *Settings* → *API* → *Create New Token*. This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
3. Run:

```bash
python src/load_data.py
```

This downloads and unzips `IMDB Dataset.csv` into `data/raw/`.

**Option B — Manual download**

Download the CSV from the Kaggle link above and place it at `data/raw/IMDB Dataset.csv`.

---

## 6. How to run

> _To be completed as the pipeline is built._

```bash
# Exploration & statistics
python src/data_analysis.py

# Full cleaning pipeline → writes data/processed/cleaned_imdb.csv
python src/cleaning_pipeline.py
```

---

## 7. Team workflow

- `main` is protected. Nobody commits to it directly.
- Each member works on their own branch (e.g. `feature/tokenization`) and opens a **pull request** to merge.
- Commit small and often, with clear messages (e.g. `Add regex for HTML tag removal`).
- The final Phase 1 submission is tagged **`assignment-1`**.

---

## 8. Contributions

| Member | Contribution |
|--------|--------------|
| Maham Iqbal | _to be filled_ |
| Minahil Malik | _to be filled_ |
| _<Name 3>_ | _to be filled_ |
| _<Name 4>_ | _to be filled_ |
| _<Name >_ | _to be filled_ |
