"""
router/engine.py — Central Orchestrator for Hybrid AI Routing

The single entry point for the entire system. Any interface (web UI, API, CLI)
calls engine.process_query() and gets back a complete response with routing
metadata.

Flow:
    1. Check semantic cache (VectorDB) → instant answer if hit
    2. Classify query complexity → Search / Local / Cloud
    3. Fetch carbon grid reading → modifier shifts tier boundary
    4. Apply Eco/Performance mode adjustment
    5. Resolve final tier pick
    6. Execute chosen tier
    7. Auto-escalate if response is insufficient
    8. Cache response + log routing decision

Usage:
    from app.router.engine import RoutingEngine

    engine = RoutingEngine()
    await engine.setup()

    result = await engine.process_query("What is quantum computing?", mode="eco")
    print(result.response)       # The answer
    print(result.tier_name)      # "Local"
    print(result.routing_reason) # "Classified as Local | grid=clean | mode=eco"

    await engine.shutdown()
"""

from __future__ import annotations

import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Energy estimates per tier (kWh per request)
ENERGY_SEARCH_KWH   = 0.0001   # ~0.1 Wh — simple API call
ENERGY_LOCAL_KWH     = 0.001    # ~1 Wh — on-device inference
ENERGY_FRONTIER_KWH  = 0.008    # ~8 Wh — cloud frontier model

# Tier labels
TIER_MAP = {
    "Search": 1,
    "Local":  2,
    "Cloud":  3,
}
TIER_NAMES = {1: "Search", 2: "Local", 3: "Cloud"}
TIER_ENERGY = {1: ENERGY_SEARCH_KWH, 2: ENERGY_LOCAL_KWH, 3: ENERGY_FRONTIER_KWH}

# Auto-escalation: minimum response length to consider "sufficient"
MIN_RESPONSE_LEN = 10

Mode = Literal["eco", "performance"]


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class RouteResult:
    """Complete result from a routed query."""
    response: str                          # The answer text
    tier_used: int = 0                     # 1, 2, or 3
    tier_name: str = ""                    # "Search" / "Local" / "Cloud"
    routing_reason: str = ""               # Human-readable explanation
    was_escalated: bool = False            # Did it auto-escalate?
    original_tier: int = 0                 # Tier before escalation
    carbon_status: str = "unknown"         # "clean" / "medium" / "dirty"
    carbon_modifier: int = 0              # Grid modifier applied
    mode: str = "eco"                      # "eco" / "performance"
    latency_ms: float = 0.0               # Total request time
    energy_kwh: float = 0.0               # Estimated energy for this request
    cached: bool = False                   # Served from semantic cache?
    model_info: str = ""                   # Backend/model details
    timestamp: str = ""                    # ISO timestamp

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "tier_used": self.tier_used,
            "tier_name": self.tier_name,
            "routing_reason": self.routing_reason,
            "was_escalated": self.was_escalated,
            "original_tier": self.original_tier,
            "carbon_status": self.carbon_status,
            "carbon_modifier": self.carbon_modifier,
            "mode": self.mode,
            "latency_ms": round(self.latency_ms, 1),
            "energy_kwh": round(self.energy_kwh, 6),
            "cached": self.cached,
            "model_info": self.model_info,
            "timestamp": self.timestamp,
        }


@dataclass
class EngineMetrics:
    """Cumulative metrics across all requests."""
    total_requests: int = 0
    tier_counts: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    cache_hits: int = 0
    escalations: int = 0
    total_energy_kwh: float = 0.0
    frontier_calls_avoided: int = 0      # Times we could have gone T3 but didn't

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "tier_counts": dict(self.tier_counts),
            "cache_hits": self.cache_hits,
            "escalations": self.escalations,
            "total_energy_kwh": round(self.total_energy_kwh, 6),
            "frontier_calls_avoided": self.frontier_calls_avoided,
        }


# ─────────────────────────────────────────────
# Routing Engine
# ─────────────────────────────────────────────

class RoutingEngine:
    """
    Central orchestrator that connects:
        - Classifier (query → Search/Local/Cloud)
        - Carbon Grid (real-time CA grid intensity)
        - VectorDB (semantic cache + routing history)
        - Tier 1: Search (Linkup API)
        - Tier 2: Local LLM (ExecuTorch/Qwen3)
        - Tier 3: Frontier LLM (Gemini 2.5 Flash)

    All dependencies are optional — the engine degrades gracefully
    if VectorDB, Carbon Grid, or specific tiers aren't available.
    """

    def __init__(self):
        # Components (initialized in setup())
        self._classifier = None
        self._carbon_grid = None
        self._vector_store = None
        self._local_llm = None

        # State
        self._ready = False
        self._metrics = EngineMetrics()

    # ─── Lifecycle ─────────────────────────────

    async def setup(self, enable_vectordb: bool = True, enable_carbon: bool = True):
        """
        Initialize all components. Each is optional — failures are logged
        but don't prevent the engine from starting.
        """
        logger.info("RoutingEngine: starting setup...")

        # 1. Classifier (required — lightweight, always works)
        try:
            from app.router.classifier import Classifier
            self._classifier = Classifier()
            logger.info("  Classifier: loaded (%d search, %d local, %d cloud keywords)",
                        len(self._classifier.weights.get("Search", {})),
                        len(self._classifier.weights.get("Local", {})),
                        len(self._classifier.weights.get("Cloud", {})))
        except Exception as e:
            logger.error("  Classifier: FAILED — %s", e)

        # 2. Carbon Grid (optional)
        if enable_carbon:
            try:
                from app.carbon.grid import CarbonGrid
                self._carbon_grid = CarbonGrid()
                await self._carbon_grid.start()
                reading = await self._carbon_grid.get_current()
                logger.info("  Carbon Grid: active (status=%s, modifier=%+d)",
                            reading.status.value, reading.modifier)
            except Exception as e:
                logger.warning("  Carbon Grid: unavailable — %s (will route without carbon data)", e)
                self._carbon_grid = None

        # 3. VectorDB / Semantic Cache (optional)
        if enable_vectordb:
            try:
                from app.db.vectordb import VectorStore
                self._vector_store = VectorStore()
                await self._vector_store.setup()
                logger.info("  VectorDB: connected (semantic cache + routing)")
            except Exception as e:
                logger.warning("  VectorDB: unavailable — %s (will route without cache)", e)
                self._vector_store = None

        # 4. Local LLM — Tier 2 (optional, heavy)
        try:
            from app.tiers.local_llm import LocalLLM
            self._local_llm = LocalLLM()
            await self._local_llm.setup()
            logger.info("  Local LLM: ready (backend=%s, model=%s)",
                        self._local_llm.backend, self._local_llm.model_id)
        except Exception as e:
            logger.warning("  Local LLM: unavailable — %s (Tier 2 will escalate to Tier 3)", e)
            self._local_llm = None

        self._ready = True
        logger.info("RoutingEngine: setup complete")

    async def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("RoutingEngine: shutting down...")

        if self._carbon_grid:
            try:
                await self._carbon_grid.stop()
            except Exception:
                pass

        if self._vector_store:
            try:
                await self._vector_store.close()
            except Exception:
                pass

        if self._local_llm:
            try:
                await self._local_llm.close()
            except Exception:
                pass

        self._ready = False
        logger.info("RoutingEngine: shutdown complete")

    # ─── Main Entry Point ──────────────────────

    async def process_query(
        self,
        query: str,
        mode: Mode = "eco",
        session_id: Optional[str] = None,
    ) -> RouteResult:
        """
        Process a user query through the full routing pipeline.

        Args:
            query:      The user's question/prompt
            mode:       "eco" (favor lower tiers) or "performance" (allow cloud)
            session_id: Optional session ID for per-user tracking

        Returns:
            RouteResult with response, tier info, routing reason, and metrics
        """
        if not self._ready:
            raise RuntimeError("RoutingEngine not initialized — call setup() first")

        start = time.perf_counter()
        reasons = []    # Build up the routing explanation

        # ── Step 1: Semantic Cache Check ───────────
        # (skipped if VectorDB not available)
        cache_hit = await self._check_cache(query)
        if cache_hit:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics.total_requests += 1
            self._metrics.cache_hits += 1
            return RouteResult(
                response=cache_hit["response"],
                tier_used=cache_hit["tier"],
                tier_name=TIER_NAMES.get(cache_hit["tier"], "Cached"),
                routing_reason="Semantic cache hit (similar query answered before)",
                cached=True,
                latency_ms=elapsed_ms,
                energy_kwh=0.0,     # No computation needed
                mode=mode,
                carbon_status=await self._get_carbon_status(),
                timestamp=_utc_now(),
            )

        # ── Step 2: Classify Query ─────────────────
        classified = self._classify(query)
        initial_tier = TIER_MAP.get(classified, 2)
        reasons.append(f"Classified as {classified}")

        # ── Step 3: Carbon Grid Modifier ───────────
        carbon_status, carbon_modifier = await self._get_carbon_info()
        reasons.append(f"grid={carbon_status}" + (f" ({carbon_modifier:+d})" if carbon_modifier else ""))

        # ── Step 4: Mode Adjustment ────────────────
        final_tier = self._apply_mode(initial_tier, mode, carbon_modifier, carbon_status)
        reasons.append(f"mode={mode}")

        if final_tier != initial_tier:
            reasons.append(f"adjusted {TIER_NAMES[initial_tier]}->{TIER_NAMES[final_tier]}")
            if initial_tier == 3 and final_tier < 3:
                self._metrics.frontier_calls_avoided += 1

        # ── Step 5: Execute Tier ───────────────────
        response, model_info = await self._execute_tier(final_tier, query)

        # ── Step 6: Auto-Escalate if Needed ────────
        was_escalated = False
        original_tier = final_tier

        if self._should_escalate(response, final_tier):
            was_escalated = True
            next_tier = min(final_tier + 1, 3)
            reasons.append(f"escalated {TIER_NAMES[final_tier]}->{TIER_NAMES[next_tier]}")
            self._metrics.escalations += 1

            response, model_info = await self._execute_tier(next_tier, query)
            final_tier = next_tier

        # ── Step 7: Cache + Log ────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        energy = TIER_ENERGY.get(final_tier, ENERGY_LOCAL_KWH)

        # Update metrics
        self._metrics.total_requests += 1
        self._metrics.tier_counts[final_tier] = self._metrics.tier_counts.get(final_tier, 0) + 1
        self._metrics.total_energy_kwh += energy

        # Cache response in VectorDB (fire-and-forget, don't block)
        asyncio.create_task(self._cache_response(query, response, final_tier))

        result = RouteResult(
            response=response,
            tier_used=final_tier,
            tier_name=TIER_NAMES.get(final_tier, "Unknown"),
            routing_reason=" | ".join(reasons),
            was_escalated=was_escalated,
            original_tier=original_tier,
            carbon_status=carbon_status,
            carbon_modifier=carbon_modifier,
            mode=mode,
            latency_ms=elapsed_ms,
            energy_kwh=energy,
            cached=False,
            model_info=model_info,
            timestamp=_utc_now(),
        )

        logger.info("Query routed: tier=%d (%s) | %.0fms | %s",
                     final_tier, TIER_NAMES[final_tier], elapsed_ms,
                     result.routing_reason)

        return result

    # ─── Step Implementations ──────────────────

    def _classify(self, query: str) -> str:
        """Run the classifier. Falls back to 'Local' if classifier unavailable."""
        if not self._classifier:
            logger.warning("No classifier available, defaulting to Local")
            return "Local"

        try:
            return self._classifier.classify(query)
        except Exception as e:
            logger.error("Classifier error: %s, defaulting to Local", e)
            return "Local"

    def _get_classification_detail(self, query: str) -> dict:
        """Get detailed classification with scores (for logging/debug)."""
        if not self._classifier:
            return {}
        try:
            return self._classifier.explain(query)
        except Exception:
            return {}

    async def _get_carbon_info(self) -> tuple[str, int]:
        """
        Get current carbon status and modifier.
        Returns ("unknown", 0) if grid is unavailable.
        """
        if not self._carbon_grid:
            return "unknown", 0

        try:
            reading = await self._carbon_grid.get_current()
            return reading.status.value, reading.modifier
        except Exception as e:
            logger.warning("Carbon grid error: %s", e)
            return "unknown", 0

    async def _get_carbon_status(self) -> str:
        """Quick carbon status string for cache hits."""
        status, _ = await self._get_carbon_info()
        return status

    def _apply_mode(
        self,
        tier: int,
        mode: Mode,
        carbon_modifier: int,
        carbon_status: str,
    ) -> int:
        """
        Adjust the tier based on mode and carbon.

        Eco mode:
            - Cloud → downgrade to Local (unless grid is clean AND classifier is very confident)
            - Search stays Search
            - Local stays Local

        Performance mode:
            - Trust classifier as-is
            - Only dirty grid pushes Cloud → Local

        Carbon modifier (from grid.py):
            - Applied as a "penalty" that makes it harder to stay at Tier 3
            - clean=+0, medium=+8, dirty=+15
        """
        if mode == "eco":
            # Eco: aggressively avoid Tier 3
            if tier == 3:
                if carbon_status == "clean":
                    # Clean grid in eco: still downgrade to Local
                    # (eco means save energy regardless of grid)
                    return 2
                else:
                    # Medium/dirty grid in eco: definitely downgrade
                    return 2
            # Eco doesn't affect Tier 1 or 2
            return tier

        else:  # performance
            # Performance: allow Cloud, but carbon can still block it
            if tier == 3 and carbon_status == "dirty":
                # Even in perf mode, dirty grid pushes to Local
                return 2
            return tier

    async def _check_cache(self, query: str) -> Optional[dict]:
        """
        Check VectorDB semantic cache for a similar past response.
        Returns {"response": str, "tier": int} or None.
        """
        if not self._vector_store:
            return None

        try:
            # Generate embedding from query text
            embedding = await self._get_embedding(query)
            if not embedding:
                return None

            hit = await self._vector_store.check_cache(embedding)
            if hit:
                logger.info("Semantic cache hit (similarity=%.3f)", hit.similarity)
                return {"response": hit.response, "tier": hit.tier_used}
        except Exception as e:
            logger.debug("Cache check failed: %s", e)

        return None

    async def _cache_response(self, query: str, response: str, tier: int):
        """Cache the response in VectorDB for future semantic matches."""
        if not self._vector_store:
            return

        try:
            embedding = await self._get_embedding(query)
            if embedding:
                await self._vector_store.cache_response(embedding, response, tier)
                await self._vector_store.log_query(embedding, tier=tier)
        except Exception as e:
            logger.debug("Cache store failed: %s", e)

    async def _get_embedding(self, text: str) -> Optional[list]:
        """
        Generate an embedding vector for the given text.

        Uses sentence-transformers if available, otherwise returns None
        (which disables semantic caching gracefully).
        """
        try:
            from sentence_transformers import SentenceTransformer
            # Lazy-load the model (cached after first call)
            if not hasattr(self, "_embed_model"):
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            embedding = self._embed_model.encode(text).tolist()
            return embedding
        except ImportError:
            logger.debug("sentence-transformers not installed, skipping embeddings")
            return None
        except Exception as e:
            logger.debug("Embedding generation failed: %s", e)
            return None

    async def _execute_tier(self, tier: int, query: str) -> tuple[str, str]:
        """
        Execute the chosen tier and return (response_text, model_info).
        Falls back gracefully if a tier is unavailable.
        """
        if tier == 1:
            return await self._execute_search(query)
        elif tier == 2:
            return await self._execute_local(query)
        elif tier == 3:
            return await self._execute_cloud(query)
        else:
            return "Unknown tier", ""

    async def _execute_search(self, query: str) -> tuple[str, str]:
        """Tier 1: Web search via Linkup API."""
        try:
            from linkup import LinkupClient
            import os

            api_key = os.getenv("LINKUP_API_KEY", "")
            if not api_key:
                logger.warning("LINKUP_API_KEY not set, search unavailable")
                return "", "search:no-api-key"

            client = LinkupClient(api_key=api_key)

            # Run sync client in executor to not block async loop
            loop = asyncio.get_event_loop()
            search_response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    depth="standard",
                    output_type="sourcedAnswer",
                ),
            )

            answer = getattr(search_response, "answer", str(search_response))
            return answer or "", "search:linkup"

        except Exception as e:
            logger.error("Search tier error: %s", e)
            return "", f"search:error:{e}"

    async def _execute_local(self, query: str) -> tuple[str, str]:
        """Tier 2: Local LLM via ExecuTorch/transformers."""
        if not self._local_llm:
            logger.warning("Local LLM unavailable, returning empty for escalation")
            return "", "local:unavailable"

        try:
            result = await self._local_llm.generate(query)
            model_info = f"local:{self._local_llm.backend}:{self._local_llm.model_id}"
            return result.text, model_info
        except Exception as e:
            logger.error("Local LLM error: %s", e)
            return "", f"local:error:{e}"

    async def _execute_cloud(self, query: str) -> tuple[str, str]:
        """Tier 3: Frontier LLM via Gemini 2.5 Flash."""
        try:
            from app.tiers.gemini_call import make_call

            # make_call is synchronous — run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, make_call, query)

            if response and not response.startswith("Error"):
                return response, "cloud:gemini-2.5-flash"
            else:
                return response or "", f"cloud:error:{response}"

        except Exception as e:
            logger.error("Cloud tier error: %s", e)
            return "", f"cloud:error:{e}"

    def _should_escalate(self, response: str, current_tier: int) -> bool:
        """
        Decide if we should auto-escalate to the next tier.

        Triggers:
            - Response is empty or too short
            - Response is an error message
            - Current tier is not already the highest (3)
        """
        if current_tier >= 3:
            return False

        if not response or len(response.strip()) < MIN_RESPONSE_LEN:
            return True

        # Check if response looks like an error
        error_signals = ["error", "unavailable", "failed", "not found"]
        response_lower = response.lower()[:100]
        if any(sig in response_lower for sig in error_signals):
            return True

        return False

    # ─── Metrics & Status ──────────────────────

    def get_metrics(self) -> dict:
        """Return cumulative engine metrics."""
        return {
            **self._metrics.to_dict(),
            "components": {
                "classifier": self._classifier is not None,
                "carbon_grid": self._carbon_grid is not None,
                "vector_store": self._vector_store is not None,
                "local_llm": self._local_llm is not None,
            },
        }

    @property
    def is_ready(self) -> bool:
        return self._ready

    def __repr__(self) -> str:
        components = []
        if self._classifier:
            components.append("classifier")
        if self._carbon_grid:
            components.append("carbon")
        if self._vector_store:
            components.append("vectordb")
        if self._local_llm:
            components.append(f"local_llm({self._local_llm.backend})")
        return f"RoutingEngine(ready={self._ready}, components=[{', '.join(components)}])"


# ─── Helpers ──────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
