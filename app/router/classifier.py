from pathlib import Path
import csv
import re
from typing import Dict, Optional


class Classifier:
    """Classifier that classifies text into: Search, Local, Cloud using CSV keyword weights.

    CSV files are expected at app/db/<name>.csv by default; you can override paths.
    """

    def __init__(
        self,
        search_path: Optional[str] = None,
        local_path: Optional[str] = None,
        cloud_path: Optional[str] = None,
    ):
        base_db = Path(__file__).resolve().parent.parent / "db"
        self.search_path = (
            Path(search_path) if search_path else base_db / "search_weights.csv"
        )
        self.local_path = (
            Path(local_path) if local_path else base_db / "local_weights.csv"
        )
        self.cloud_path = (
            Path(cloud_path) if cloud_path else base_db / "cloud_weights.csv"
        )

        self.weights = {
            "Search": self._load_csv_weights(self.search_path),
            "Local": self._load_csv_weights(self.local_path),
            "Cloud": self._load_csv_weights(self.cloud_path),
        }
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
                phrase = row[0].strip()
                if not phrase or phrase.startswith("#"):
                    continue
                weight = 1.0
                if len(row) > 1:
                    try:
                        weight = float(row[1])
                    except Exception:
                        weight = 1.0
                if 0.0 < weight < 1.0:
                    weight = weight * 10.0
                out[phrase] = weight
        return out

    def _normalize_text(self, text: str) -> str:
        text = (text or "").lower()
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
                    new_map[norm] = new_map.get(norm, 0.0) + float(w)
            self.weights[cat] = new_map

    def classify(self, text: str) -> str:
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
        order = ["Search", "Local", "Cloud"]
        best = max(order, key=lambda c: (scores.get(c, 0.0), -order.index(c)))
        return best

    def explain(self, text: str) -> Dict[str, object]:
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
    c = Classifier()
    print(c.explain("What's the weather in New York and rewrite this email?"))
