# router.py
"""Router that classifies text into: Search, Local, Cloud using CSV keyword weights.

Files (default names):
- search_weights.csv
- local_weights.csv
- cloud_weights.csv

Each CSV is expected to be: phrase,weight  (no header required)

Behavior:
- Loads weights from CSVs
- If a weight parsed is < 1.0, it is scaled by 10 (so 0.9 -> 9.0)
- Phrases are normalized (lowercased, punctuation removed) for substring matching

Usage:
    from router import Router
    r = Router()
    r.classify("What's the weather in Seattle?")  # -> 'Search'

"""
from pathlib import Path
import csv
import re
from typing import Dict


class Classifier:
    DEFAULT_SEARCH = "search_weights.csv"
    DEFAULT_LOCAL = "local_weights.csv"
    DEFAULT_CLOUD = "cloud_weights.csv"

    def __init__(
        self,
        search_path: str = DEFAULT_SEARCH,
        local_path: str = DEFAULT_LOCAL,
        cloud_path: str = DEFAULT_CLOUD,
    ):
        # load weights as phrase -> float
        self.weights = {
            "Search": self._load_csv_weights(Path(search_path)),
            "Local": self._load_csv_weights(Path(local_path)),
            "Cloud": self._load_csv_weights(Path(cloud_path)),
        }
        # normalize loaded phrases for fast substring matching
        self._normalize_loaded_phrases()

    def _load_csv_weights(self, path: Path) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                # allow rows like: phrase,weight  (ignore extra columns)
                phrase = row[0].strip()
                if not phrase or phrase.startswith("#"):
                    continue
                weight = 1.0
                if len(row) > 1:
                    try:
                        weight = float(row[1])
                    except Exception:
                        weight = 1.0
                # Normalize small fractional weights by scaling factor 10
                # so that 0.9 -> 9.0 (per workspace convention)
                if 0.0 < weight < 1.0:
                    weight = weight * 10.0
                out[phrase] = weight
        return out

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower()
        # remove punctuation except hyphen and apostrophe
        text = re.sub(r"[^\w\s\-']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_loaded_phrases(self) -> None:
        # convert loaded phrase keys to normalized form for matching
        for cat in list(self.weights.keys()):
            mapping = self.weights[cat]
            new_map: Dict[str, float] = {}
            for phrase, w in mapping.items():
                norm = self._normalize_text(phrase)
                if norm:
                    new_map[norm] = new_map.get(norm, 0.0) + float(w)
            self.weights[cat] = new_map

    def classify(self, text: str) -> str:
        """Classify input text into one of: 'Search', 'Local', 'Cloud'.

        - Normalize input text
        - For each normalized phrase present as a substring, add its weight
        - Return category with highest total weight
        - If no matches, default to 'Cloud'
        """
        if not text:
            return "Cloud"
        norm = self._normalize_text(text)
        scores: Dict[str, float] = {c: 0.0 for c in self.weights}
        for cat, mapping in self.weights.items():
            for phrase, weight in mapping.items():
                if phrase and phrase in norm:
                    scores[cat] += weight
        if all(v == 0 for v in scores.values()):
            return "Cloud"
        # deterministic tie-break: prefer Search, then Local, then Cloud
        order = ["Search", "Local", "Cloud"]
        best = max(order, key=lambda c: (scores.get(c, 0.0), -order.index(c)))
        return best

    def explain(self, text: str) -> Dict[str, object]:
        """Return explanation dict:
        - normalized: normalized input text
        - matches: dict of category -> {phrase: weight} for phrases that matched
        - scores: total scores per category
        - winner: chosen category (defaults to 'Cloud' when no matches)
        """
        norm = self._normalize_text(text or "")
        matches: Dict[str, Dict[str, float]] = {c: {} for c in self.weights}
        scores: Dict[str, float] = {c: 0.0 for c in self.weights}
        for cat, mapping in self.weights.items():
            for phrase, weight in mapping.items():
                if phrase and phrase in norm:
                    matches[cat][phrase] = weight
                    scores[cat] += weight
        winner = "Cloud"
        if not all(v == 0 for v in scores.values()):
            order = ["Search", "Local", "Cloud"]
            winner = max(order, key=lambda c: (scores.get(c, 0.0), -order.index(c)))
        return {
            "normalized": norm,
            "matches": matches,
            "scores": scores,
            "winner": winner,
        }


if __name__ == "__main__":
    r = Classifier()
    samples = [
        "What's the weather in San Francisco today?",
        "Rewrite this email to be shorter and more polite.",
        "Design a distributed system for a messaging app, step by step architecture.",
        "Where is the nearest coffee shop close to me?",
    ]
    for s in samples:
        print(s)
        print("->", r.classify(s))
        print()
