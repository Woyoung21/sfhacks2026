"""
API route for live metrics.
"""

from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> dict:
    tracker = request.app.state.tracker
    engine = request.app.state.engine
    req_logger = request.app.state.req_logger
    return {
        "live": tracker.snapshot(),
        "engine": engine.get_metrics(),
        "recent_requests": req_logger.recent(limit=20),
    }
