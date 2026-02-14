# interactive_test.py
"""Interactive console for Router.
Run: python interactive_test.py
Type a query and press Enter. Empty input or Ctrl+C to exit.
"""
from router import Router


def main():
    r = Router()
    print("Router interactive test. Enter a query (empty to quit).")
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
