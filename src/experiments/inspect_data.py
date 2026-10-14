"""Inspect the text-classification datasets without training a model.

This module is intentionally read-only: it prints a compact data profile and
never rewrites the supplied CSV files.
"""

from __future__ import annotations

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import re
import sys
from pathlib import Path

import pandas as pd


from common.config import DATA_DIR
TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data_unlabeled.csv"
PREDICTION_EXAMPLE_PATH = DATA_DIR / "predictions.csv"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labeled training set and unlabeled submission test set."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def normalize_text(text: str) -> str:
    """Normalize case and whitespace for conservative duplicate checks."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def text_statistics(texts: pd.Series) -> pd.DataFrame:
    """Return robust character- and word-count summaries."""
    clean_texts = texts.fillna("").astype(str)
    measures = {
        "characters": clean_texts.str.len(),
        "words": clean_texts.str.split().str.len(),
    }
    rows = []
    for name, values in measures.items():
        quantiles = values.quantile([0, 0.25, 0.5, 0.75, 0.95, 1.0])
        rows.append(
            {
                "measure": name,
                "min": int(quantiles.loc[0]),
                "p25": int(quantiles.loc[0.25]),
                "median": int(quantiles.loc[0.5]),
                "p75": int(quantiles.loc[0.75]),
                "p95": int(quantiles.loc[0.95]),
                "max": int(quantiles.loc[1.0]),
            }
        )
    return pd.DataFrame(rows)


def header_coverage(texts: pd.Series) -> pd.Series:
    """Measure how often email/newsgroup-style metadata appears in the text."""
    patterns = {
        "From": r"(?im)^From\s*:",
        "Subject": r"(?im)^Subject\s*:",
        "Newsgroups": r"(?im)^Newsgroups\s*:",
        "Organization": r"(?im)^Organization\s*:",
        "Lines": r"(?im)^Lines\s*:",
        "quoted_reply": r"(?m)^>\s?",
    }
    clean_texts = texts.fillna("").astype(str)
    return pd.Series(
        {
            name: clean_texts.str.contains(pattern, regex=True).mean()
            for name, pattern in patterns.items()
        },
        name="share",
    )


def preview_text(text: str, limit: int = 140) -> str:
    """Prefer a Subject line; otherwise show the first non-empty text line."""
    text = str(text)
    subject = re.search(r"(?im)^Subject\s*:\s*(.+)$", text)
    if subject:
        candidate = f"Subject: {subject.group(1).strip()}"
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = lines[0] if lines else "<empty>"
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate[:limit]


def print_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print schema, completeness, uniqueness, and distribution checks."""
    print("\n=== 1. Dataset shape and schema ===")
    print(f"train: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"test:  {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
    print(f"train columns: {train_df.columns.tolist()}")
    print(f"test columns:  {test_df.columns.tolist()}")

    print("\n=== 2. Missing and empty values ===")
    for name, frame in (("train", train_df), ("test", test_df)):
        missing = frame.isna().sum()
        string_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_string_dtype(frame[column].dtype)
        ]
        empty = frame[string_columns].apply(
            lambda column: column.fillna("").astype(str).str.strip().eq("").sum()
        )
        print(f"{name} missing:\n{missing.to_string()}")
        print(f"{name} empty strings:\n{empty.to_string()}")

    print("\n=== 3. Label distribution ===")
    label_counts = train_df["target"].value_counts().sort_index()
    label_table = pd.DataFrame(
        {
            "count": label_counts,
            "share_pct": (label_counts / len(train_df) * 100).round(2),
        }
    )
    print(label_table.to_string())
    print(
        "largest/smallest class ratio: "
        f"{label_counts.max() / label_counts.min():.3f}"
    )

    print("\n=== 4. Text-length distribution ===")
    print("train:\n" + text_statistics(train_df["text"]).to_string(index=False))
    print("test:\n" + text_statistics(test_df["text"]).to_string(index=False))
    for name, frame in (("train", train_df), ("test", test_df)):
        lengths = frame["text"].fillna("").astype(str).str.len()
        long_counts = [
            f">{threshold:,}: {int((lengths > threshold).sum())} "
            f"({(lengths > threshold).mean():.2%})"
            for threshold in (5_000, 10_000, 50_000)
        ]
        print(f"{name} long-text counts: " + ", ".join(long_counts))

    print("\n=== 5. Duplicate and overlap checks ===")
    train_norm = train_df["text"].map(normalize_text)
    test_norm = test_df["text"].map(normalize_text)
    train_duplicates = int(train_norm.duplicated().sum())
    test_duplicates = int(test_norm.duplicated().sum())
    overlap = len(set(train_norm) & set(test_norm))
    print(f"normalized duplicate train rows: {train_duplicates}")
    print(f"normalized duplicate test rows:  {test_duplicates}")
    print(f"exact normalized texts shared by train and test: {overlap}")

    print("\n=== 6. Email/newsgroup metadata coverage ===")
    coverage = pd.concat(
        {
            "train_pct": (header_coverage(train_df["text"]) * 100).round(2),
            "test_pct": (header_coverage(test_df["text"]) * 100).round(2),
        },
        axis=1,
    )
    print(coverage.to_string())

    print("\n=== 7. One example per numeric class ===")
    examples = (
        train_df.groupby("target", group_keys=False)
        .sample(n=1, random_state=42)
        .sort_values("target")
    )
    for row in examples.itertuples(index=False):
        print(f"class {row.target}: {preview_text(row.text)}")


def inspect_prediction_example(expected_rows: int) -> None:
    """Check whether the supplied predictions file is a real submission."""
    if not PREDICTION_EXAMPLE_PATH.exists():
        return
    predictions = pd.read_csv(PREDICTION_EXAMPLE_PATH, header=None)
    counts = predictions.iloc[:, 0].value_counts(dropna=False).sort_index()
    print("\n=== 8. Supplied predictions.csv example ===")
    print(f"shape: {predictions.shape}")
    print(f"expected rows: {expected_rows}")
    print("value counts:\n" + counts.to_string())


def main() -> None:
    """Run all read-only checks."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train_df, test_df = load_datasets()
    print_dataset_summary(train_df, test_df)
    inspect_prediction_example(expected_rows=len(test_df))


if __name__ == "__main__":
    main()
