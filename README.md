# SFHACKS2026

Carbon-aware hybrid AI router with 3 tiers:
- Tier 1: Search (Linkup)
- Tier 2: Local LLM (ExecuTorch/Transformers)
- Tier 3: Cloud LLM (Gemini)

## 1. Prerequisites

- Python 3.12+
- Docker Engine (for Actian VectorDB)

## 2. Install dependencies

From project root (`sfhacks2026/`):

```bash
pip install -r requirements.txt
```

## 3. Install Actian Cortex client wheel

```bash
pip install ./lib/actiancortex-0.1.0b1-py3-none-any.whl
```

## 4. Start VectorDB (Docker)

```bash
docker pull williamimoh/actian-vectorai-db:1.0b
docker run -d --name actian-vectordb -p 50051:50051 williamimoh/actian-vectorai-db:1.0b
```

Verify:

```bash
ss -ltn | grep 50051
```

## 5. Configure environment

Create/update `.env` in project root:

```env
# Search
LINKUP_API_KEY=...

# Cloud LLM
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_OUTPUT_TOKENS=250
GEMINI_TEMPERATURE=0.4

# Carbon
WATTTIME_USERNAME=...
WATTTIME_PASSWORD=...
GRID_REGION=CAISO_NORTH

# VectorDB
VECTORDB_HOST=localhost:50051
# Optional gRPC tuning (helps avoid "too_many_pings")
VECTORDB_POOL_SIZE=1
VECTORDB_KEEPALIVE_MS=600000
VECTORDB_KEEPALIVE_TIMEOUT_MS=20000

# Optional runtime toggles
ENABLE_VECTORDB=true
ENABLE_CARBON=true
LOCAL_MAX_TOKENS=250
```

## 6. Run the web app

```bash
uvicorn app.main:app --reload
```

Open:
- Chat: `http://127.0.0.1:8000/`
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Admin: `http://127.0.0.1:8000/admin`
- API docs: `http://127.0.0.1:8000/docs`

## 7. Sanity tests

Backend API smoke test:

```bash
python -m app.test.backend_api_smoke_test
```

Cache repeat integration test:

```bash
python -m app.test.cache_repeat_integration_test
```

Search-route integrated test:

```bash
python -m app.test.search_route_test
```
