"""
Real engine integration test for miss->cache->hit behavior.

Requires:
  - VectorDB running at VECTORDB_HOST
  - sentence-transformers installed

Run:
  python -m app.test.cache_repeat_integration_test
"""

from __future__ import annotations

import asyncio

from app.router.engine import RoutingEngine


QUERY = "what is the weather in san francisco today"


async def main() -> int:
    engine = RoutingEngine()
    await engine.setup(enable_vectordb=True, enable_carbon=True)
    try:
        first = await engine.process_query(QUERY, mode="eco")
        # Cache writes are async fire-and-forget in current engine design.
        await asyncio.sleep(2)
        second = await engine.process_query(QUERY, mode="eco")

        print("first.cached:", first.cached, "tier:", first.tier_name)
        print("second.cached:", second.cached, "tier:", second.tier_name)
        print("metrics:", engine.get_metrics())

        if not second.cached:
            print("FAIL: expected second query to hit semantic cache")
            return 1

        print("PASS: cache repeat integration test")
        return 0
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
