"""
Admin endpoints for backend runtime controls and status.
"""

from __future__ import annotations

from typing import Optional, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()

Mode = Literal["eco", "performance"]


class AdminUpdate(BaseModel):
    default_mode: Optional[Mode] = None
    vector_hint_min_conf: Optional[float] = None
    cache_write_sync: Optional[bool] = None


@router.get("/admin")
async def get_admin_state(request: Request) -> dict:
    cfg = dict(getattr(request.app.state, "admin_config", {}))
    engine = request.app.state.engine
    return {
        "config": cfg,
        "engine": engine.get_metrics(),
    }


@router.post("/admin")
async def update_admin_state(payload: AdminUpdate, request: Request) -> dict:
    cfg = dict(getattr(request.app.state, "admin_config", {}))
    updates = payload.model_dump(exclude_none=True)
    if "vector_hint_min_conf" in updates:
        val = float(updates["vector_hint_min_conf"])
        updates["vector_hint_min_conf"] = max(0.0, min(1.0, val))
    cfg.update(updates)
    request.app.state.admin_config = cfg
    return {"ok": True, "config": cfg}
