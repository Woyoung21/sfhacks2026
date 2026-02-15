"""
Low-cost Tier 3 smoke test (Gemini API).

Run:
  python -m app.test.cloud_smoke_test
"""

from __future__ import annotations

from app.tiers.gemini_call import make_call


PROMPT = "Reply with exactly: OK"


def main() -> int:
    response = make_call(PROMPT)
    print("prompt:", PROMPT)
    print("response_preview:", (response or "")[:120])

    if not response or str(response).lower().startswith("error"):
        print("FAIL: cloud call failed")
        return 1

    print("PASS: cloud call succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
