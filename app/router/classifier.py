from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import Dict, Optional, List, Tuple, Pattern, Any


class Classifier:
    """
    Rule-based classifier that routes text into: Search, Local, Cloud
    using CSV keyword/phrase weights plus a few high-leverage intent gates.

    Key design:
      - Intent gates first (transform, proof/reasoning, explicit lookup).
      - Then weighted phrase matching (longest phrases first).
      - If no scores, default to Search (your preference).
    """

    # --- High-leverage intent gates (normalized text is lowercase + cleaned) ---
    TRANSFORM_GATES = [
        "rewrite", "rephrase", "paraphrase", "shorten", "make this shorter",
        "summarize", "summary", "bullet points", "turn these notes into",
        "proofread", "fix grammar", "edit this email", "rewrite this email",
        "make this email shorter", "make this paragraph shorter",
    ]

    REASONING_GATES = [
        "prove", "theorem", "formal proof", "complex theorem", "derive", "show that",
        "rigorous", "proof",
    ]

    # Search should only win when it looks like *retrieval/lookup*
    LOOKUP_GATES = [
        "what time is it", "time in", "time zone",
        "weather", "forecast",
        "near me", "opening hours", "closing time", "open now", "hours",
        "address", "phone number",
        "how far", "distance",
        "population", "definition", "meaning of",
        "who is", "when did", "where is",
        "exchange rate", "stock price",
    ]

    def __init__(
        self,
        search_path: Optional[str] = None,
        local_path: Optional[str] = None,
        cloud_path: Optional[str] = None,
    ):
        base_db = Path(__file__).resolve().parent.parent / "db"
        self.search_path = Path(search_path) if search_path else base_db / "search_weights.csv"
        self.local_path = Path(local_path) if local_path else base_db / "local_weights.csv"
        self.cloud_path = Path(cloud_path) if cloud_path else base_db / "cloud_weights.csv"

        self.weights = {
            "Search": self._load_csv_weights(self.search_path),
            "Local": self._load_csv_weights(self.local_path),
            "Cloud": self._load_csv_weights(self.cloud_path),
        }
        self._normalize_loaded_phrases()

        # Precompile matchers (longest phrases first)
        self._compiled: Dict[str, List[Tuple[str, float, Pattern[str], int]]] = {}
        for cat, mapping in self.weights.items():
            compiled = []
            for phrase, w in mapping.items():
                pat = self._compile_phrase_pattern(phrase)
                compiled.append((phrase, float(w), pat, len(phrase)))
            compiled.sort(key=lambda t: t[3], reverse=True)  # longest first
            self._compiled[cat] = compiled

    def _load_csv_weights(self, path: Path) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                phrase = (row[0] or "").strip()
                if not phrase or phrase.startswith("#"):
                    continue
                # ignore common header mistakes
                if phrase.strip().lower() in {"keyword", "phrase"}:
                    continue

                weight = 1.0
                if len(row) > 1:
                    try:
                        weight = float(row[1])
                    except Exception:
                        weight = 1.0

                # If somebody provided [0,1] normalize to your scale
                if 0.0 < weight < 1.0:
                    weight = weight * 100.0

                # Clamp to sane range
                if weight < 0:
                    weight = 0.0

                out[phrase] = float(weight)
        return out

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        # keep words, spaces, hyphen, apostrophe
        text = re.sub(r"[^\w\s\-']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_loaded_phrases(self) -> None:
        for cat in list(self.weights.keys()):
            mapping = self.weights[cat]
            new_map: Dict[str, float] = {}
            for phrase, w in mapping.items():
                norm = self._normalize_text(phrase)
                if norm:
                    # if duplicates exist, keep the max (more stable than summing)
                    new_map[norm] = max(new_map.get(norm, 0.0), float(w))
            self.weights[cat] = new_map

    def _compile_phrase_pattern(self, phrase: str) -> Pattern[str]:
        """
        Compile a regex pattern that matches the phrase as a whole word/phrase.
        - For single tokens: use \\bword\\b
        - For multi-word phrases: match word boundaries at ends and allow flexible whitespace.
        """
        phrase = phrase.strip()
        if not phrase:
            return re.compile(r"(?!x)x")  # never matches

        if " " not in phrase:
            # single token with word boundary
            token = re.escape(phrase)
            return re.compile(rf"\b{token}\b")
        else:
            # multi-token phrase: escape, then allow flexible whitespace between tokens
            parts = [re.escape(p) for p in phrase.split()]
            mid = r"\s+".join(parts)
            return re.compile(rf"\b{mid}\b")

    def _has_any(self, norm_text: str, phrases: List[str]) -> bool:
        # simple substring is fine here because phrases are already normalized
        return any(p in norm_text for p in phrases)

    def classify(self, text: str) -> str:
        if not text:
            return "Search"

        norm = self._normalize_text(text)

        # --- Intent gates (highest ROI) ---
        # Transform / edit / summarize -> Local
        if self._has_any(norm, self.TRANSFORM_GATES):
            return "Local"

        # Proof / theorem / derive -> Cloud
        if self._has_any(norm, self.REASONING_GATES):
            return "Cloud"

        # Explicit lookup cues -> Search
        # (You can expand this over time; it's intentionally conservative.)
        if self._has_any(norm, self.LOOKUP_GATES):
            # still allow Cloud to override if it's clearly generative
            # (e.g., "generate a diagram of weather data")
            pass

        # --- Weighted matching ---
        scores: Dict[str, float] = {c: 0.0 for c in self.weights}

        for cat, compiled in self._compiled.items():
            for phrase, weight, pat, _length in compiled:
                if not phrase:
                    continue
                if pat.search(norm):
                    scores[cat] += weight

        # If no points anywhere -> Search (your requested behavior)
        if all(v == 0 for v in scores.values()):
            return "Search"

        # If we saw explicit lookup cues, give Search a small boost
        if self._has_any(norm, self.LOOKUP_GATES):
            scores["Search"] += 25.0

        # Stable tie-break preference (Search > Local > Cloud for ties)
        order = ["Search", "Local", "Cloud"]
        best = max(order, key=lambda c: (scores.get(c, 0.0), -order.index(c)))
        return best

    def explain(self, text: str) -> Dict[str, Any]:
        norm = self._normalize_text(text or "")

        # Gate explanations
        gates_hit = {
            "transform_gate": [p for p in self.TRANSFORM_GATES if p in norm],
            "reasoning_gate": [p for p in self.REASONING_GATES if p in norm],
            "lookup_gate": [p for p in self.LOOKUP_GATES if p in norm],
        }

        # If gates force a decision, still show matches for debugging
        matches: Dict[str, Dict[str, float]] = {c: {} for c in self.weights}
        scores: Dict[str, float] = {c: 0.0 for c in self.weights}

        for cat, compiled in self._compiled.items():
            for phrase, weight, pat, _length in compiled:
                if phrase and pat.search(norm):
                    matches[cat][phrase] = weight
                    scores[cat] += weight

        if all(v == 0 for v in scores.values()):
            winner = "Search"
        else:
            if gates_hit["lookup_gate"]:
                scores["Search"] += 25.0
            order = ["Search", "Local", "Cloud"]
            winner = max(order, key=lambda c: (scores.get(c, 0.0), -order.index(c)))

        # If transform/reasoning gate triggers, show the forced winner too
        forced = None
        if gates_hit["transform_gate"]:
            forced = "Local"
        elif gates_hit["reasoning_gate"]:
            forced = "Cloud"

        return {
            "normalized": norm,
            "gates_hit": gates_hit,
            "matches": matches,
            "scores": scores,
            "winner": forced or winner,
            "forced_by_gate": forced is not None,
        }

