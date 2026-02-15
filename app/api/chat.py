"""
API route for routed chat requests.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    mode: Literal["eco", "performance"] = "eco"
    session_id: Optional[str] = None


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> dict:
    engine = request.app.state.engine
    tracker = request.app.state.tracker
    req_logger = request.app.state.req_logger

    try:
        result = await engine.process_query(
            payload.query,
            mode=payload.mode,
            session_id=payload.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat processing failed: {e}") from e

    tracker.record(result)
    req_logger.log(query=payload.query, mode=payload.mode, result=result)
    return result.to_dict()
