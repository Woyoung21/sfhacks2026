"""
Backend entrypoint for API + routing engine lifecycle.

Run:
  uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.api.feedback import close_store, router as feedback_router
from app.api.metrics import router as metrics_router
from app.metrics.logger import RequestLogger
from app.metrics.tracker import MetricsTracker
from app.router.engine import RouteResult
from app.router.engine import RoutingEngine


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    testing = bool(getattr(app.state, "testing", False))
    if testing:
        engine = _TestEngine()
    else:
        engine = RoutingEngine()
        await engine.setup(
            enable_vectordb=_env_bool("ENABLE_VECTORDB", True),
            enable_carbon=_env_bool("ENABLE_CARBON", True),
        )

    app.state.engine = engine
    app.state.tracker = MetricsTracker()
    app.state.req_logger = RequestLogger(
        log_path="logs/test_requests.jsonl" if testing else "logs/requests.jsonl"
    )
    app.state.admin_config = {
        "default_mode": os.getenv("DEFAULT_MODE", "eco"),
        "vector_hint_min_conf": float(os.getenv("VECTOR_HINT_MIN_CONF", "0.7")),
        "cache_write_sync": _env_bool("CACHE_WRITE_SYNC", False),
    }

    try:
        yield
    finally:
        await close_store()
        if not testing:
            await engine.shutdown()


class _TestEngine:
    async def process_query(self, query: str, mode: str = "eco", session_id: str | None = None) -> RouteResult:
        return RouteResult(
            response=f"test-response:{query}",
            tier_used=1,
            tier_name="Search",
            routing_reason="test-engine",
            mode=mode,
            latency_ms=1.0,
            energy_kwh=0.0,
            model_info="test:search",
        )

    def get_metrics(self) -> dict:
        return {
            "total_requests": 0,
            "tier_counts": {1: 0, 2: 0, 3: 0},
            "cache_hits": 0,
            "escalations": 0,
            "total_energy_kwh": 0.0,
            "frontier_calls_avoided": 0,
            "components": {"test_engine": True},
        }


def create_app(*, testing: bool = False) -> FastAPI:
    app = FastAPI(title="SFHACKS2026 API", version="0.1.0", lifespan=lifespan)
    app.state.testing = testing

    app.include_router(chat_router, prefix="/api")
    app.include_router(metrics_router, prefix="/api")
    app.include_router(feedback_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return app


app = create_app()
