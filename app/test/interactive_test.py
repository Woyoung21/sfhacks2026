# interactive_test.py
"""Interactive console for Classifier in app.test with debug prints.
Run (preferred): python -m app.test.interactive_test
Or run directly: python app/test/interactive_test.py
Type a query and press Enter. Empty input or Ctrl+C to exit.
"""
from pathlib import Path
import sys

try:
    from app.router.classifier import Classifier
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.router.classifier import Classifier


def main():
    r = Classifier()
    # Debug: show resolved paths and loaded weight counts
    print("Search CSV:", r.search_path)
    print("Local CSV:", r.local_path)
    print("Cloud CSV:", r.cloud_path)
    print("Loaded weight counts:")
    for cat, mapping in r.weights.items():
        print(f"  {cat}: {len(mapping)} entries")
        # show up to 5 samples
        items = list(mapping.items())[:5]
        for phrase, weight in items:
            print(f"    '{phrase}' -> {weight}")
    print("\nNow interactive. Enter a query (empty to quit).")

    try:
        while True:
            text = input("query> ").strip()
            if not text:
                print("Exiting.")
                break
            result = r.explain(text)
            print(f"Normalized: {result['normalized']}")
            print("Matches per category:")
            for cat, matches in result["matches"].items():
                if matches:
                    print(f"  {cat}:")
                    for phrase, weight in matches.items():
                        print(f"    '{phrase}' -> {weight}")
            print("Scores:")
            for cat, score in result["scores"].items():
                print(f"  {cat}: {score}")
            print(f"Winner: {result['winner']}")
            print()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
