"""
Bounded threshold tuning helpers for routing decisions.

This module keeps threshold math explicit and safe (bounded) so future
adaptive tuning cannot drift into unstable ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["eco", "performance"]


@dataclass
class ThresholdConfig:
    """
    Thresholds over a 0-100 complexity score.

    - score <= search_max: Tier 1 (Search)
    - score <= local_max:  Tier 2 (Local)
    - else:                Tier 3 (Cloud)
    """

    search_max: int = 35
    local_max: int = 75
    min_bound: int = 5
    max_bound: int = 95

    def clamp(self) -> None:
        self.search_max = max(self.min_bound, min(self.search_max, self.max_bound))
        self.local_max = max(self.min_bound, min(self.local_max, self.max_bound))
        if self.local_max <= self.search_max:
            self.local_max = min(self.max_bound, self.search_max + 1)


class AdaptiveThresholds:
    """Small bounded adjustments based on mode + carbon modifier."""

    def __init__(self, config: ThresholdConfig | None = None):
        self.cfg = config or ThresholdConfig()
        self.cfg.clamp()

    def adjusted_local_max(self, mode: Mode, carbon_modifier: int) -> int:
        """
        Adjust the local/cloud boundary with bounded shifts.
        Positive output makes Cloud harder to reach.
        """
        shift = 0
        if mode == "eco":
            shift += 8
        else:
            shift -= 4

        shift += max(0, carbon_modifier)
        adjusted = self.cfg.local_max + shift
        return max(self.cfg.min_bound, min(adjusted, self.cfg.max_bound))

    def pick_tier(self, complexity_score: int, mode: Mode, carbon_modifier: int) -> int:
        """Map score -> tier using adjusted thresholds."""
        score = max(0, min(100, int(complexity_score)))
        local_max = self.adjusted_local_max(mode, carbon_modifier)
        if score <= self.cfg.search_max:
            return 1
        if score <= local_max:
            return 2
        return 3
