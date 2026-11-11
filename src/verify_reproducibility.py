import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cleaning_pipeline import CLEAN_COL, OUTPUT_FILE, PIPELINE_STEPS, build_clean_dataset, to_csv_text
from load_data import LABEL_COL, ROOT, load_raw


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if not OUTPUT_FILE.exists():
        print(f"No committed output found at {OUTPUT_FILE.relative_to(ROOT)}")
        print("Run  python src/cleaning_pipeline.py  first.")
        return 1

    committed_text = OUTPUT_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    committed = pd.read_csv(OUTPUT_FILE, keep_default_na=False, dtype=str)

    raw = load_raw()
    fresh = build_clean_dataset(raw)
    fresh_text = to_csv_text(fresh)

    checks = []
    checks.append(("row count matches raw data", len(fresh) == len(raw) == len(committed)))
    checks.append(("labels identical to raw data", fresh[LABEL_COL].tolist() == raw[LABEL_COL].tolist()))
    checks.append(("labels identical to committed CSV", committed[LABEL_COL].tolist() == fresh[LABEL_COL].tolist()))
    checks.append(("tweet ids in the same order", committed["tweet_id"].tolist() == fresh["tweet_id"].astype(str).tolist()))
    checks.append(("no missing labels", fresh[LABEL_COL].notna().all()))
    checks.append(("byte-identical output (SHA-256)", committed_text == fresh_text))

    print("Pipeline steps used:")
    for name, src in PIPELINE_STEPS:
        print(f"  {name:<40} <- {src}")
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n  committed SHA-256: {sha256(committed_text)}")
    print(f"  fresh     SHA-256: {sha256(fresh_text)}")

    if committed_text != fresh_text and len(committed) == len(fresh):
        diff = committed[CLEAN_COL] != fresh[CLEAN_COL].fillna("")
        print(f"\n  {int(diff.sum())} of {len(fresh)} cleaned tweets differ. First few:")
        for i in diff[diff].index[:5]:
            print(f"    row {i}: committed={committed.at[i, CLEAN_COL]!r}")
            print(f"    {' ' * len(f'row {i}: ')}fresh    ={fresh.at[i, CLEAN_COL]!r}")

    passed = all(ok for _, ok in checks)
    print("\nREPRODUCIBLE" if passed else "\nNOT REPRODUCIBLE")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
