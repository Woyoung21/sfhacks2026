"""
Backend API smoke test using FastAPI TestClient (no heavy model startup).

Run:
  python -m app.test.backend_api_smoke_test
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def main() -> int:
    app = create_app(testing=True)
    with TestClient(app) as client:
        r = client.get("/healthz")
        if r.status_code != 200 or r.json().get("ok") is not True:
            print("FAIL: /healthz", r.status_code, r.text)
            return 1

        r = client.post("/api/chat", json={"query": "hello", "mode": "eco"})
        if r.status_code != 200:
            print("FAIL: /api/chat", r.status_code, r.text)
            return 1

        chat_payload = r.json()
        if "response" not in chat_payload:
            print("FAIL: /api/chat missing response")
            return 1

        r = client.get("/api/metrics")
        if r.status_code != 200:
            print("FAIL: /api/metrics", r.status_code, r.text)
            return 1

        r = client.post(
            "/api/feedback",
            json={"entry_id": -1, "classification": "Search", "feedback": "up"},
        )
        if r.status_code != 200:
            print("FAIL: /api/feedback", r.status_code, r.text)
            return 1

        r = client.get("/api/admin")
        if r.status_code != 200:
            print("FAIL: /api/admin GET", r.status_code, r.text)
            return 1

        r = client.post("/api/admin", json={"default_mode": "performance"})
        if r.status_code != 200:
            print("FAIL: /api/admin POST", r.status_code, r.text)
            return 1

    print("PASS: backend API smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
