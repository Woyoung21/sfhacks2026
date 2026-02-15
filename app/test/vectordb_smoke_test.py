"""
VectorDB smoke test for cache + routing workflow.

Run:
  python -m app.test.vectordb_smoke_test
"""

from __future__ import annotations

import asyncio

from app.db.vectordb import VectorStore


async def main() -> int:
    store = VectorStore()
    try:
        await store.setup()
        dim = store._dim
        base_vec = [0.001] * dim
        alt_vec = [0.002] * dim

        await store.cache_response(base_vec, "smoke-cache-response", tier_used=1)
        cache_hit = await store.check_cache(base_vec)

        await store.log_query(base_vec, tier_routed=1, complexity_score=20)
        await store.log_query(alt_vec, tier_routed=2, complexity_score=55)
        prediction = await store.predict_tier(base_vec, top_k=5)
        stats = await store.get_stats()

        print("cache_hit:", bool(cache_hit))
        if cache_hit:
            print("cache_hit_response:", cache_hit.response)
            print("cache_hit_tier:", cache_hit.tier_used)
            print("cache_hit_similarity:", round(cache_hit.similarity, 4))

        print("prediction:", prediction)
        print("stats:", stats)
        return 0
    except Exception as exc:
        print("vectordb_smoke_test_failed:", exc)
        return 1
    finally:
        await store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
