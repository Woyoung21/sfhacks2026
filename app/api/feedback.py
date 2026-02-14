"""
feedback.py

Collect simple user feedback after an action item returns a result.

Appends a row to a CSV with:
  timestamp, question, response, classification, feedback

Usage patterns:
  1) Import and call collect_feedback(...)
  2) CLI mode (manual test)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal


FeedbackValue = Literal["up", "down"]


@dataclass
class FeedbackRecord:
    timestamp_utc: str
    question: str
    response: str
    classification: str
    feedback: FeedbackValue


def _utc_now_iso() -> str:
    # ISO 8601 UTC timestamp like: 2026-02-14T23:59:59Z
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_feedback(raw: str) -> Optional[FeedbackValue]:
    s = (raw or "").strip().lower()

    # Accept lots of quick inputs
    if s in {"up", "u", "y", "yes", "1", "👍", "+", "thumbs up", "thumbsup"}:
        return "up"
    if s in {"down", "d", "n", "no", "0", "👎", "-", "thumbs down", "thumbsdown"}:
        return "down"
    return None


def prompt_for_feedback(
    prompt: str = "Feedback? (👍/👎) [up/down]: "
) -> FeedbackValue:
    while True:
        raw = input(prompt)
        val = _normalize_feedback(raw)
        if val is not None:
            return val
        print("Please enter 'up'/'down' (or 👍/👎).")


def append_feedback_csv(
    csv_path: Path,
    record: FeedbackRecord,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header only if file doesn't exist or is empty
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp_utc", "question", "response", "classification", "feedback"])
        writer.writerow([record.timestamp_utc, record.question, record.response, record.classification, record.feedback])


def collect_feedback(
    question: str,
    response: str,
    classification: str,
    csv_path: str | Path = "app/db/feedback_log.csv",
    interactive: bool = True,
    feedback: Optional[FeedbackValue] = None,
) -> FeedbackRecord:
    """
    Call this after your router takes an action and returns a result.

    If interactive=True, prompt user for thumbs up/down.
    If interactive=False, you must pass feedback="up" or "down".
    """
    csv_path = Path(csv_path)

    if interactive:
        fb = prompt_for_feedback()
    else:
        if feedback not in {"up", "down"}:
            raise ValueError("If interactive=False, feedback must be 'up' or 'down'.")
        fb = feedback

    record = FeedbackRecord(
        timestamp_utc=_utc_now_iso(),
        question=question,
        response=response,
        classification=classification,
        feedback=fb,
    )

    append_feedback_csv(csv_path, record)
    return record


# --- Optional CLI test mode ---
if __name__ == "__main__":
    print("feedback.py CLI test mode")
    q = input("Question: ").strip()
    c = input("Classification (Search/Local/Cloud): ").strip() or "Unknown"
    r = input("Response: ").strip()

    rec = collect_feedback(q, r, c, csv_path="app/db/feedback_log.csv", interactive=True)
    print("\nSaved feedback:")
    print(rec)

