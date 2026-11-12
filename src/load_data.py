"""
load_data.py: download (if needed) and load the raw Twitter US Airline Sentiment dataset.

Dataset: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
Licence: CC BY-NC-SA 4.0

Usage (run from the repository root):
    python src/load_data.py

The raw CSV is saved to data/raw/ and is NOT committed to Git (see .gitignore).
"""

import sys
from pathlib import Path

import pandas as pd

# Paths are built from this file's location, so the script works
# no matter which folder you run it from.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "Tweets.csv"
KAGGLE_DATASET = "crowdflower/twitter-airline-sentiment"

TEXT_COL = "text"                  # the raw tweet
LABEL_COL = "airline_sentiment"    # prediction target: negative / neutral / positive

MANUAL_INSTRUCTIONS = f"""
Could not download the dataset automatically.

Option A: set up the Kaggle API:
  1. Kaggle -> Settings -> API -> "Create New Token" (downloads kaggle.json)
  2. Put kaggle.json in:  C:\\Users\\<you>\\.kaggle\\   (Windows)
                          ~/.kaggle/                 (macOS / Linux)
  3. Run this script again.

Option B: download manually:
  1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}
  2. Click "Download", unzip it
  3. Put "Tweets.csv" here:  {RAW_FILE}
"""


def download_raw(force: bool = False) -> Path:
    """Download the dataset from Kaggle into data/raw/ (skipped if it already exists)."""
    if RAW_FILE.exists() and not force:
        print(f"Raw data already present: {RAW_FILE}")
        return RAW_FILE

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {KAGGLE_DATASET} from Kaggle ...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True, quiet=False)
    except Exception as err:  # missing package, missing kaggle.json, no internet, ...
        print(f"Error: {err}")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    if not RAW_FILE.exists():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)
    # The zip also contains database.sqlite (same data). We only use Tweets.csv.
    print(f"Saved to {RAW_FILE}")
    return RAW_FILE


def load_raw() -> pd.DataFrame:
    """Return the raw dataset as a DataFrame (all 15 original columns)."""
    if not RAW_FILE.exists():
        download_raw()
    df = pd.read_csv(RAW_FILE, encoding="utf-8")

    expected = {TEXT_COL, LABEL_COL, "airline", "tweet_id"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {expected}, found {list(df.columns)}")
    return df


if __name__ == "__main__":
    download_raw()
    data = load_raw()
    print(f"\nLoaded {len(data):,} rows x {data.shape[1]} columns")
    print(data[[TEXT_COL, LABEL_COL, "airline"]].head())