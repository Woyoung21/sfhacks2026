# app/test/batch_eval.py
"""
Batch evaluator for the router Classifier.

Input CSV format (NO HEADER):
question,expected_label

- question may contain commas if it's quoted properly by CSV rules.
- expected_label must be one of: search, local, cloud (case-insensitive).

Usage:
  python -m app.test.batch_eval path/to/dataset.csv
  OR
  python app/test/batch_eval.py path/to/dataset.csv

Output:
  - Prints summary stats
  - Writes misclassified lines to: misclassified_<inputname>.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    from app.router.classifier import Classifier
except ModuleNotFoundError:
    # Allows running directly from repo root without -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.router.classifier import Classifier


LABEL_MAP = {
    "search": "Search",
    "local": "Local",
    "cloud": "Cloud",
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def normalize_expected(label: str) -> str:
    s = (label or "").strip().lower()
    if s not in LABEL_MAP:
        raise ValueError(f"Invalid expected label '{label}'. Must be one of: search, local, cloud.")
    return LABEL_MAP[s]


def make_output_path(input_csv: Path) -> Path:
    # misclassified_datasetname.txt next to the input file
    stem = input_csv.stem
    return input_csv.with_name(f"misclassified_{stem}.txt")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m app.test.batch_eval <path/to/dataset.csv>")
        return 2

    input_path = Path(sys.argv[1]).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return 2

    clf = Classifier()

    total = 0
    correct = 0

    # Confusion-ish stats
    expected_counts = Counter()
    predicted_counts = Counter()
    correct_by_label = Counter()

    # Track wrong examples
    misclassified = []

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < 2:
                print(f"Skipping line {line_num}: expected 2 columns, got {len(row)} -> {row}")
                continue

            question = (row[0] or "").strip()
            expected_raw = (row[1] or "").strip()

            if not question:
                print(f"Skipping line {line_num}: empty question")
                continue

            try:
                expected = normalize_expected(expected_raw)
            except ValueError as e:
                print(f"Skipping line {line_num}: {e}")
                continue

            predicted = clf.classify(question)

            total += 1
            expected_counts[expected] += 1
            predicted_counts[predicted] += 1

            if predicted == expected:
                correct += 1
                correct_by_label[expected] += 1
            else:
                # Save a readable record
                misclassified.append(
                    f"[line {line_num}] expected={REVERSE_LABEL_MAP.get(expected, expected)} "
                    f"predicted={REVERSE_LABEL_MAP.get(predicted, predicted)} :: {question}"
                )

    if total == 0:
        print("No valid rows found.")
        return 1

    accuracy = correct / total
    print(f"Evaluated: {total}")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {accuracy:.2%}")

    print("\nPer-label accuracy:")
    for label in ["Search", "Local", "Cloud"]:
        n = expected_counts[label]
        if n == 0:
            print(f"  {label}: (no examples)")
            continue
        acc = correct_by_label[label] / n
        print(f"  {label}: {acc:.2%} ({correct_by_label[label]}/{n})")

    print("\nPrediction distribution:")
    for label in ["Search", "Local", "Cloud"]:
        print(f"  {label}: {predicted_counts[label]}")

    out_path = make_output_path(input_path)
    with out_path.open("w", encoding="utf-8") as out:
        for line in misclassified:
            out.write(line + "\n")

    print(f"\nWrote misclassified examples to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

