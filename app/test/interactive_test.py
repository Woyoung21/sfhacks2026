# interactive_test.py
"""Interactive console for Classifier in app.test with debug prints.
Run (preferred): python -m app.test.interactive_test
Or run directly: python app/test/interactive_test.py
Type a query and press Enter. Empty input or Ctrl+C to exit.
"""

from pathlib import Path
import sys
from datetime import datetime
from datetime import timezone

try:
    from app.router.classifier import Classifier
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.router.classifier import Classifier

# Import Gemini call (local file)
try:
    from app.tiers.gemini_call import make_call
except ImportError:
    make_call = None


def write_cloud_response(query: str, response: str) -> Path:
    """Write cloud response to a new file next to this script."""
    base_dir = Path(__file__).resolve().parent
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / f"cloud_response_{ts}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("QUERY:\n")
        f.write(query)
        f.write("\n\nRESPONSE:\n")
        f.write(response)

    return out_path


def main():
    r = Classifier()

    # Debug: show resolved paths and loaded weight counts
    print("Search CSV:", r.search_path)
    print("Local CSV:", r.local_path)
    print("Cloud CSV:", r.cloud_path)
    print("Loaded weight counts:")
    for cat, mapping in r.weights.items():
        print(f"  {cat}: {len(mapping)} entries")
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

            winner = result["winner"]
            print(f"Winner: {winner}")

            # 🔥 Cloud path → call Gemini
            if winner == "Cloud":
                if make_call is None:
                    print("⚠️ gemini_call.py not available; skipping cloud call.")
                else:
                    print("☁️ Routing to Gemini (cloud)...")
                    try:
                        response = make_call(text)
                        out_file = write_cloud_response(text, response)
                        print(f"✅ Cloud response written to: {out_file}")
                    except Exception as e:
                        print(f"❌ Error during Gemini call: {e}")

            print()

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()

