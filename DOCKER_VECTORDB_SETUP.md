# Actian VectorDB Docker Setup

This guide sets up the Actian VectorAI DB container (headless) for this project.

## 1. Install Docker Engine (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker --version
```

## 2. Pull the Actian image

```bash
docker pull williamimoh/actian-vectorai-db:1.0b
```

## 3. Run the DB container (headless/background)

```bash
docker run -d \
  --name actian-vectordb \
  -p 50051:50051 \
  williamimoh/actian-vectorai-db:1.0b
```

## 4. Verify DB is listening

```bash
docker ps
ss -ltn | grep 50051
```

Expected: a LISTEN entry on `0.0.0.0:50051` (or equivalent).

## 5. Configure app environment

In `./.env`, set:

```env
VECTORDB_HOST=localhost:50051
```

## 6. Install Actian Python client wheel (project dependency)

From project root:

```bash
pip install ./lib/actiancortex-0.1.0b1-py3-none-any.whl
```

## 7. Smoke test DB integration

```bash
python -m app.test.vectordb_smoke_test
```

Expected output includes:
- `cache_hit: True`
- `stats: {'connected': True, ...}`

## 8. Run full engine with VectorDB enabled

```bash
python - <<'PY'
import asyncio
from app.router.engine import RoutingEngine

async def main():
    e = RoutingEngine()
    await e.setup(enable_vectordb=True, enable_carbon=True)
    r = await e.process_query("what color is the sky", mode="eco")
    print(r.to_dict())
    print("metrics:", e.get_metrics())
    await e.shutdown()

asyncio.run(main())
PY
```

## Useful Docker commands

Stop container:

```bash
docker stop actian-vectordb
```

Start existing container:

```bash
docker start actian-vectordb
```

View logs:

```bash
docker logs -f actian-vectordb
```

Remove container:

```bash
docker rm -f actian-vectordb
```

## Troubleshooting

- `docker: command not found`: Docker Engine is not installed.
- `permission denied` on Docker commands: run `newgrp docker` or log out/in after `usermod -aG docker $USER`.
- `port 50051 already in use`: stop conflicting process/container and re-run.
- gRPC `UNAVAILABLE`: confirm `VECTORDB_HOST` and `ss -ltn | grep 50051`.
