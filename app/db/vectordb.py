"""
db/vectordb.py — Actian VectorAI DB Integration

Two core uses:
    1. Semantic Caching  — avoid redundant LLM calls by matching on meaning
    2. Semantic Routing   — predict the best tier from similar past queries

Privacy model:
    - The `routing_history` collection stores ONLY embeddings + metadata
      (tier, feedback, complexity score). No raw query text. Embeddings
      are mathematically irreversible — they cannot be decoded back to text.
    - The `semantic_cache` collection stores response text (needed to serve
      cached answers) but entries auto-expire after a configurable TTL.
    - Users can clear their session data at any time.

Usage:
    from app.db.vectordb import VectorStore

    store = VectorStore()                         # default localhost:50051
    store = VectorStore(host="localhost:50051")    # explicit host

    await store.setup()                           # create collections on startup
    await store.close()                           # clean shutdown

    # Semantic cache — check before routing
    hit = await store.check_cache(query_vector)
    if hit:
        return hit.response  # no LLM call needed

    # Semantic routing — predict tier from past queries
    predicted_tier = await store.predict_tier(query_vector)

    # After generating a response — log for future routing
    await store.log_query(query_vector, tier=2, feedback=None, complexity=45)

    # After generating a response — cache for future hits
    await store.cache_response(query_vector, response_text, tier=2)

Environment variables:
    VECTORDB_HOST       — VectorAI DB host (default: localhost:50051)
    CACHE_TTL_HOURS     — Hours before cache entries expire (default: 24)
    CACHE_SIMILARITY    — Min cosine similarity to count as cache hit (default: 0.93)
    EMBEDDING_DIM       — Embedding vector dimension (default: 384)
"""

import os
import time
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

from cortex import AsyncCortexClient, DistanceMetric

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class CacheResult:
    """A semantic cache hit — a past response that matches the current query."""
    response: str           # The cached response text
    tier_used: int          # Which tier generated it (1, 2, or 3)
    similarity: float       # Cosine similarity score (0-1)
    cached_at: str          # ISO timestamp of when it was cached


@dataclass
class TierPrediction:
    """Semantic routing prediction from similar past queries."""
    predicted_tier: int     # Most likely tier (1, 2, or 3)
    confidence: float       # How confident (0-1), based on agreement among neighbors
    num_neighbors: int      # How many similar past queries were found
    tier_scores: dict       # Weighted scores per tier {1: 0.4, 2: 1.8, 3: 0.3}


# ─────────────────────────────────────────────
# Collection names
# ─────────────────────────────────────────────

ROUTING_COLLECTION = "routing_history"   # Embeddings only — no raw text
CACHE_COLLECTION   = "semantic_cache"    # Has response text — auto-expires


# ─────────────────────────────────────────────
# VectorStore — Main Interface
# ─────────────────────────────────────────────

class VectorStore:
    """
    Actian VectorAI DB integration for semantic caching and routing.

    Connects to a VectorAI DB instance (via Docker on port 50051)
    and manages two collections:

        routing_history  — stores query embeddings + tier/feedback metadata
                           (no raw text for privacy). Used to predict the
                           best tier for new queries via similarity search.

        semantic_cache   — stores query embeddings + response text with TTL.
                           Used to serve cached responses for semantically
                           similar questions, avoiding LLM calls entirely.
    """

    def __init__(self, host: Optional[str] = None):
        self._host = host or os.getenv("VECTORDB_HOST", "localhost:50051")
        self._dim = int(os.getenv("EMBEDDING_DIM", "384"))
        self._cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
        self._cache_similarity = float(os.getenv("CACHE_SIMILARITY", "0.93"))
        self._pool_size = int(os.getenv("VECTORDB_POOL_SIZE", "1"))
        self._keepalive_time_ms = int(os.getenv("VECTORDB_KEEPALIVE_MS", "600000"))
        self._keepalive_timeout_ms = int(os.getenv("VECTORDB_KEEPALIVE_TIMEOUT_MS", "20000"))
        self._client: Optional[AsyncCortexClient] = None

        # Auto-incrementing IDs (tracked in-memory, reset on restart)
        self._routing_id_counter = 0
        self._cache_id_counter = 0

        logger.info(
            f"VectorStore configured: host={self._host}, dim={self._dim}, "
            f"cache_ttl={self._cache_ttl_hours}h, cache_sim={self._cache_similarity}"
        )

    # ─── Lifecycle ───────────────────────────

    async def setup(self):
        """
        Connect to VectorAI DB and create collections if they don't exist.
        Call this on app startup.
        """
        self._client = AsyncCortexClient(self._host, pool_size=self._pool_size)

        # Actian SDK exposes internal pool config; tune keepalive to avoid
        # server GOAWAY "too_many_pings" under idle/polling-heavy sessions.
        if hasattr(self._client, "_pool_config"):
            self._client._pool_config.keepalive_time_ms = self._keepalive_time_ms
            self._client._pool_config.keepalive_timeout_ms = self._keepalive_timeout_ms

        await self._client.connect()

        # Create routing_history collection (embeddings + metadata only)
        if not await self._client.has_collection(ROUTING_COLLECTION):
            await self._client.create_collection(
                name=ROUTING_COLLECTION,
                dimension=self._dim,
                distance_metric=DistanceMetric.COSINE,
            )
            logger.info(f"Created collection: {ROUTING_COLLECTION}")
        else:
            logger.info(f"Collection exists: {ROUTING_COLLECTION}")

        # Create semantic_cache collection (embeddings + response text)
        if not await self._client.has_collection(CACHE_COLLECTION):
            await self._client.create_collection(
                name=CACHE_COLLECTION,
                dimension=self._dim,
                distance_metric=DistanceMetric.COSINE,
            )
            logger.info(f"Created collection: {CACHE_COLLECTION}")
        else:
            logger.info(f"Collection exists: {CACHE_COLLECTION}")

        # Sync ID counters with existing data
        try:
            self._routing_id_counter = await self._client.count(ROUTING_COLLECTION)
            self._cache_id_counter = await self._client.count(CACHE_COLLECTION)
        except Exception:
            self._routing_id_counter = 0
            self._cache_id_counter = 0

        logger.info(
            f"VectorStore ready: {self._routing_id_counter} routing entries, "
            f"{self._cache_id_counter} cache entries"
        )

    async def close(self):
        """Disconnect from VectorAI DB. Call this on app shutdown."""
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
            logger.info("VectorStore connection closed")

    # ─── Semantic Caching ────────────────────

    async def check_cache(self, query_vector: list[float]) -> Optional[CacheResult]:
        """
        Check if a semantically similar query has been answered before.

        Searches the cache collection for the closest match. If the cosine
        similarity exceeds the threshold (default 0.93), returns the cached
        response. Otherwise returns None.

        This is called BEFORE routing — if there's a cache hit, no LLM
        call is needed at all (zero energy cost).

        Args:
            query_vector: The embedded query vector (dimension must match EMBEDDING_DIM)

        Returns:
            CacheResult if cache hit, None if miss
        """
        if not self._client:
            logger.warning("VectorStore not connected, skipping cache check")
            return None

        try:
            results = await self._client.search(
                CACHE_COLLECTION,
                query=query_vector,
                top_k=1,
                with_payload=True,
            )

            if not results:
                return None

            best = results[0]
            payload = best.payload or {}

            # Check similarity threshold
            if best.score < self._cache_similarity:
                logger.debug(
                    f"Cache near-miss: similarity {best.score:.3f} "
                    f"< threshold {self._cache_similarity}"
                )
                return None

            # Check TTL — is this entry still fresh?
            cached_at = payload.get("cached_at", "")
            if cached_at and self._is_expired(cached_at):
                logger.debug("Cache hit but expired, treating as miss")
                # Optionally delete the expired entry
                try:
                    await self._client.delete(CACHE_COLLECTION, best.id)
                except Exception:
                    pass
                return None

            logger.info(
                f"Cache HIT: similarity {best.score:.3f}, "
                f"tier {payload.get('tier_used', '?')}"
            )

            return CacheResult(
                response=payload.get("response", ""),
                tier_used=payload.get("tier_used", 0),
                similarity=best.score,
                cached_at=cached_at,
            )

        except Exception as e:
            logger.error(f"Cache check failed: {e}")
            return None

    async def cache_response(
        self,
        query_vector: list[float],
        response: str,
        tier_used: int,
    ) -> None:
        """
        Store a response in the semantic cache for future hits.

        The response text IS stored here (needed to serve cached answers).
        Entries auto-expire after CACHE_TTL_HOURS.

        Args:
            query_vector: The embedded query vector
            response: The generated response text to cache
            tier_used: Which tier generated this response (1, 2, or 3)
        """
        if not self._client:
            return

        try:
            cache_id = self._cache_id_counter
            self._cache_id_counter += 1

            await self._client.upsert(
                CACHE_COLLECTION,
                id=cache_id,
                vector=query_vector,
                payload={
                    "response": response,
                    "tier_used": tier_used,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.debug(f"Cached response (id={cache_id}, tier={tier_used})")

        except Exception as e:
            logger.error(f"Cache write failed: {e}")

    # ─── Semantic Routing ────────────────────

    async def predict_tier(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> Optional[TierPrediction]:
        """
        Predict the best tier for a query based on similar past queries.

        Searches the routing history for the K most similar past queries
        and does a weighted vote based on which tier they were routed to
        (weighted by cosine similarity). Only considers queries that
        received positive feedback (thumbs_up) or no feedback yet.

        This is called DURING routing as a second opinion alongside
        the regex-based classifier.

        Args:
            query_vector: The embedded query vector
            top_k: Number of similar past queries to consider

        Returns:
            TierPrediction with predicted tier and confidence, or None
            if not enough history exists yet.
        """
        if not self._client:
            return None

        try:
            results = await self._client.search(
                ROUTING_COLLECTION,
                query=query_vector,
                top_k=top_k,
                with_payload=True,
            )

            if not results:
                return None

            # Filter to only reasonably similar queries (> 0.7 similarity)
            relevant = [r for r in results if r.score > 0.7]

            if len(relevant) < 2:
                # Not enough similar history to make a prediction
                return None

            # Weighted vote: each neighbor votes for its tier,
            # weighted by similarity score
            tier_scores = {1: 0.0, 2: 0.0, 3: 0.0}
            for r in relevant:
                payload = r.payload or {}
                tier = payload.get("tier_routed", 2)
                feedback = payload.get("feedback")

                # Weight positive feedback higher, skip negative
                if feedback == "thumbs_down":
                    continue  # Don't learn from bad routing decisions

                weight = r.score
                if feedback == "thumbs_up":
                    weight *= 1.3  # Boost confirmed-good decisions

                tier_scores[tier] = tier_scores.get(tier, 0.0) + weight

            if sum(tier_scores.values()) == 0:
                return None

            # Pick the tier with the highest weighted score
            predicted = max(tier_scores, key=tier_scores.get)

            # Confidence = how much the winner dominates
            total = sum(tier_scores.values())
            confidence = tier_scores[predicted] / total if total > 0 else 0

            logger.debug(
                f"Tier prediction: tier={predicted}, confidence={confidence:.2f}, "
                f"scores={tier_scores}, neighbors={len(relevant)}"
            )

            return TierPrediction(
                predicted_tier=predicted,
                confidence=confidence,
                num_neighbors=len(relevant),
                tier_scores=tier_scores,
            )

        except Exception as e:
            logger.error(f"Tier prediction failed: {e}")
            return None

    async def log_query(
        self,
        query_vector: list[float],
        tier_routed: int,
        complexity_score: int,
        feedback: Optional[str] = None,
    ) -> int:
        """
        Log a completed query for future semantic routing.

        PRIVACY: This stores ONLY the embedding vector and metadata.
        No raw query text or response text is stored in this collection.
        Embeddings cannot be reversed back to the original text.

        Args:
            query_vector: The embedded query vector
            tier_routed: Which tier handled this query (1, 2, or 3)
            complexity_score: The parser's complexity score (0-100)
            feedback: Optional user feedback ("thumbs_up" / "thumbs_down")

        Returns:
            The ID of the stored entry (for updating feedback later)
        """
        if not self._client:
            return -1

        try:
            entry_id = self._routing_id_counter
            self._routing_id_counter += 1

            await self._client.upsert(
                ROUTING_COLLECTION,
                id=entry_id,
                vector=query_vector,
                payload={
                    "tier_routed": tier_routed,
                    "complexity_score": complexity_score,
                    "feedback": feedback,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.debug(
                f"Logged query (id={entry_id}, tier={tier_routed}, "
                f"score={complexity_score})"
            )

            return entry_id

        except Exception as e:
            logger.error(f"Query log failed: {e}")
            return -1

    async def update_feedback(self, entry_id: int, feedback: str) -> None:
        """
        Update the feedback on a previously logged query.

        Called when the user gives a thumbs up/down on a response.
        This improves future semantic routing predictions — positive
        feedback reinforces the tier choice, negative feedback causes
        future similar queries to skip that tier.

        Args:
            entry_id: The ID returned by log_query()
            feedback: "thumbs_up" or "thumbs_down"
        """
        if not self._client or entry_id < 0:
            return

        try:
            # get() returns tuple[list[float], dict | None]
            vector, payload = await self._client.get(ROUTING_COLLECTION, entry_id)
            if vector is not None:
                # Update payload with feedback
                payload = payload or {}
                payload["feedback"] = feedback
                payload["feedback_at"] = datetime.now(timezone.utc).isoformat()

                await self._client.upsert(
                    ROUTING_COLLECTION,
                    id=entry_id,
                    vector=vector,
                    payload=payload,
                )
                logger.debug(f"Updated feedback for entry {entry_id}: {feedback}")

        except Exception as e:
            logger.error(f"Feedback update failed: {e}")

    # ─── Cache Maintenance ───────────────────

    async def cleanup_expired_cache(self) -> int:
        """
        Remove expired entries from the semantic cache.

        Call this periodically (e.g. every hour) to keep the cache clean.
        Returns the number of entries removed.
        """
        if not self._client:
            return 0

        removed = 0
        try:
            # Scroll through all cache entries and check TTL
            cursor = None
            batch_size = 100

            while True:
                # scroll() returns tuple[list[PointRecord], int | None]
                records, next_cursor = await self._client.scroll(
                    CACHE_COLLECTION,
                    limit=batch_size,
                    cursor=cursor,
                    with_payload=True,
                )

                if not records:
                    break

                for entry in records:
                    payload = entry.payload or {}
                    cached_at = payload.get("cached_at", "")
                    if cached_at and self._is_expired(cached_at):
                        try:
                            await self._client.delete(CACHE_COLLECTION, entry.id)
                            removed += 1
                        except Exception:
                            pass

                if next_cursor is None:
                    break
                cursor = next_cursor

            if removed > 0:
                logger.info(f"Cache cleanup: removed {removed} expired entries")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

        return removed

    async def clear_session_data(self, session_id: str = None) -> None:
        """
        Clear all cached data. Used for user privacy — "clear my data" button.

        If session_id tracking is implemented in the future, this can
        be scoped to a specific session. For now, it clears all cache entries.
        """
        if not self._client:
            return

        try:
            # Recreate the cache collection (fastest way to clear all data)
            await self._client.recreate_collection(
                name=CACHE_COLLECTION,
                dimension=self._dim,
                distance_metric=DistanceMetric.COSINE,
            )
            self._cache_id_counter = 0
            logger.info("Semantic cache cleared (user privacy request)")

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    # ─── Stats ───────────────────────────────

    async def get_stats(self) -> dict:
        """
        Get current VectorAI DB stats for the metrics dashboard.
        """
        if not self._client:
            return {"connected": False}

        try:
            routing_count = await self._client.count(ROUTING_COLLECTION)
            cache_count = await self._client.count(CACHE_COLLECTION)

            return {
                "connected": True,
                "host": self._host,
                "routing_entries": routing_count,
                "cache_entries": cache_count,
                "embedding_dim": self._dim,
                "cache_ttl_hours": self._cache_ttl_hours,
                "cache_similarity_threshold": self._cache_similarity,
            }

        except Exception as e:
            logger.error(f"Stats fetch failed: {e}")
            return {"connected": False, "error": str(e)}

    # ─── Helpers ─────────────────────────────

    def _is_expired(self, iso_timestamp: str) -> bool:
        """Check if a cached entry has exceeded its TTL."""
        try:
            cached_time = datetime.fromisoformat(iso_timestamp)
            age_hours = (
                datetime.now(timezone.utc) - cached_time
            ).total_seconds() / 3600
            return age_hours > self._cache_ttl_hours
        except (ValueError, TypeError):
            return True  # If we can't parse the timestamp, treat as expired

    def __repr__(self) -> str:
        return (
            f"VectorStore(host={self._host}, "
            f"routing={self._routing_id_counter}, "
            f"cache={self._cache_id_counter})"
        )
