"""
Per-request logging for routing outcomes.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class RequestLog:
    timestamp: str
    query: str
    mode: str
    tier_name: str
    tier_used: int
    cached: bool
    was_escalated: bool
    routing_reason: str
    model_info: str
    latency_ms: float
    energy_kwh: float


class RequestLogger:
    """Writes JSONL logs and keeps a small in-memory tail."""

    def __init__(self, log_path: str = "logs/requests.jsonl", tail_size: int = 200):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._tail = deque(maxlen=tail_size)
        self._lock = Lock()

    def log(self, *, query: str, mode: str, result: Any) -> None:
        entry = RequestLog(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            query=query,
            mode=mode,
            tier_name=getattr(result, "tier_name", ""),
            tier_used=int(getattr(result, "tier_used", 0)),
            cached=bool(getattr(result, "cached", False)),
            was_escalated=bool(getattr(result, "was_escalated", False)),
            routing_reason=getattr(result, "routing_reason", ""),
            model_info=getattr(result, "model_info", ""),
            latency_ms=float(getattr(result, "latency_ms", 0.0)),
            energy_kwh=float(getattr(result, "energy_kwh", 0.0)),
        )

        payload = asdict(entry)
        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            self._tail.append(payload)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._tail)[-limit:]
