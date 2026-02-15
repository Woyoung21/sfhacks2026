"""

feedback.py

User feedback integration for routing reinforcement. 
Update VectorDB routing_history feedback (authoritative)

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from app.db.vectordb import VectorStore

FeedbackValue = Literal["up", "down"]
VectorDBFeedbackValue = Literal["thumbs_up", "thumbs_down"]


@dataclass(frozen=True)
class FeedbackRecord:
    timestamp_utc: str
    classification: str
    feedback: FeedbackValue
    vectordb_entry_id: int
    vectordb_feedback: VectorDBFeedbackValue
    updated: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_vectordb_feedback(val: FeedbackValue) -> VectorDBFeedbackValue:
    return "thumbs_up" if val == "up" else "thumbs_down"


_STORE: Optional[VectorStore] = None


async def _get_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
        await _STORE.setup()
    return _STORE


async def close_store() -> None:
    """Optional: call this on app shutdown to cleanly close the DB connection."""
    global _STORE
    if _STORE is not None:
        await _STORE.close()
        _STORE = None


async def record_feedback(
    *,
    entry_id: int,
    classification: str,
    feedback: FeedbackValue,
) -> FeedbackRecord:
    """
    Update VectorDB routing_history feedback for a previously logged query.

    Inputs:
      - entry_id: ID returned by VectorStore.log_query()
      - classification: "Search"/"Local"/"Cloud" (or whatever label you track)
      - feedback: "up" or "down"

    Output:
      - FeedbackRecord (good for stdout logging / metrics)

    Notes:
      - If entry_id < 0, VectorStore.log_query() likely failed; we do nothing.
    """
    vectordb_feedback = _to_vectordb_feedback(feedback)

    if entry_id < 0:
        return FeedbackRecord(
            timestamp_utc=_utc_now_iso(),
            classification=classification,
            feedback=feedback,
            vectordb_entry_id=entry_id,
            vectordb_feedback=vectordb_feedback,
            updated=False,
        )

    store = await _get_store()

    # update_feedback() in vectordb.py already guards entry_id < 0,
    # but we handled that above. This will no-op on connection issues.
    await store.update_feedback(entry_id, vectordb_feedback)

    return FeedbackRecord(
        timestamp_utc=_utc_now_iso(),
        classification=classification,
        feedback=feedback,
        vectordb_entry_id=entry_id,
        vectordb_feedback=vectordb_feedback,
        updated=True,
    )

