"""
Tier 1 search adapter for Linkup.

Single-purpose module used by RoutingEngine so Tier 1 behavior is centralized.
"""

from __future__ import annotations

import os

from linkup import LinkupClient


def search_with_linkup(query: str) -> str:
    """
    Run a Linkup search query and return answer text.

    Raises:
        RuntimeError: if LINKUP_API_KEY is missing.
        Exception: any Linkup client/network error.
    """
    api_key = os.getenv("LINKUP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LINKUP_API_KEY not set")

    client = LinkupClient(api_key=api_key)
    search_response = client.search(
        query=query,
        depth="standard",
        output_type="sourcedAnswer",
    )

    return getattr(search_response, "answer", str(search_response)) or ""
