"""
Live metrics tracker for API/dashboard consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class LiveMetrics:
    total_requests: int = 0
    cache_hits: int = 0
    escalations: int = 0
    tier_counts: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    total_latency_ms: float = 0.0
    total_energy_kwh: float = 0.0

    def to_dict(self) -> dict:
        avg_latency = 0.0
        if self.total_requests > 0:
            avg_latency = self.total_latency_ms / self.total_requests
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "escalations": self.escalations,
            "tier_counts": dict(self.tier_counts),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "total_energy_kwh": round(self.total_energy_kwh, 6),
        }


class MetricsTracker:
    def __init__(self):
        self._metrics = LiveMetrics()
        self._lock = Lock()

    def record(self, result: Any) -> None:
        tier = int(getattr(result, "tier_used", 0))
        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.cache_hits += 1 if bool(getattr(result, "cached", False)) else 0
            self._metrics.escalations += 1 if bool(getattr(result, "was_escalated", False)) else 0
            self._metrics.total_latency_ms += float(getattr(result, "latency_ms", 0.0))
            self._metrics.total_energy_kwh += float(getattr(result, "energy_kwh", 0.0))
            self._metrics.tier_counts[tier] = self._metrics.tier_counts.get(tier, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return self._metrics.to_dict()
