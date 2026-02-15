"""
Smoke test: verify a query routes to Tier 1 (Search).

Run:
  python -m app.test.search_route_test
"""

from __future__ import annotations

import asyncio
import sys

from app.router.engine import RoutingEngine


QUERY = "what is the weather in san francisco today"


async def main() -> int:
    engine = RoutingEngine()
    # Full integration path: classifier + carbon + vectordb + tier execution.
    await engine.setup(enable_vectordb=True, enable_carbon=True)
    try:
        result = await engine.process_query(QUERY, mode="eco")
        print("query:", QUERY)
        print("tier_used:", result.tier_used, result.tier_name)
        print("original_tier:", result.original_tier)
        print("cached:", result.cached)
        print("grid:", result.carbon_status, f"modifier={result.carbon_modifier:+d}")
        print("latency_ms:", round(result.latency_ms, 1))
        print("energy_kwh:", result.energy_kwh)
        print("model_info:", result.model_info)
        print("routing_reason:", result.routing_reason)
        print("response:")
        print(result.response)

        # In cache-hit path, classifier is skipped in current engine flow.
        if result.cached:
            if result.tier_used != 1:
                print("FAIL: cache hit returned a non-Search tier for this test case")
                return 1
        else:
            # Non-cached path must classify as Search first.
            if result.original_tier != 1:
                print("FAIL: classifier did not route to Search (Tier 1)")
                return 1

        # For integration realism, allow cache-hit or direct Tier 1.
        if not (result.cached or result.tier_used == 1):
            print("FAIL: final path was neither cache-hit nor Tier 1 Search")
            return 1

        print("PASS: integrated Search-path test passed")
        print("metrics:", engine.get_metrics())
        return 0
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
