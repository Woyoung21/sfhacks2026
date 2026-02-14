"""
carbon/grid.py — Real-time California Grid Carbon Intensity

Polls the WattTime API (v3) for CAISO marginal emissions data,
caches the result, and exposes a clean interface for the routing engine.

Usage:
    from app.carbon.grid import CarbonGrid

    grid = CarbonGrid()          # uses WATTTIME creds from env
    grid = CarbonGrid(mock=True) # simulation mode for dev

    reading = await grid.get_current()
    # reading.intensity   -> float (gCO2/kWh)
    # reading.status      -> GridStatus.CLEAN / MEDIUM / DIRTY
    # reading.modifier    -> int (threshold shift for routing engine)
    # reading.timestamp   -> datetime
    # reading.source      -> "watttime" / "electricitymaps" / "mock"

Environment variables:
    WATTTIME_USERNAME   — WattTime API username
    WATTTIME_PASSWORD   — WattTime API password
    GRID_REGION         — WattTime region (default: CAISO_NORTH)
    GRID_POLL_INTERVAL  — Seconds between polls (default: 300 = 5 min)
    CARBON_MOCK         — Set to "true" to force mock mode
"""

import os
import time
import random
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

class GridStatus(str, Enum):
    """Classification of grid carbon intensity."""
    CLEAN  = "clean"    # Low emissions — renewables dominant
    MEDIUM = "medium"   # Moderate emissions — mixed generation
    DIRTY  = "dirty"    # High emissions — fossil-heavy generation


@dataclass
class GridReading:
    """A single carbon intensity reading with routing metadata."""
    intensity: float          # gCO2/kWh (or WattTime percent 0-100)
    status: GridStatus        # clean / medium / dirty
    modifier: int             # Routing threshold shift for engine
    timestamp: datetime       # When this reading was captured
    source: str               # "watttime" | "electricitymaps" | "mock"
    region: str               # e.g. "CAISO_NORTH"
    raw_data: dict = field(default_factory=dict)  # Full API response for logging


# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────

# WattTime signal-index returns a percentile (0-100):
#   0  = grid is at its cleanest
#   100 = grid is at its dirtiest
#
# We classify into 3 buckets and assign a routing modifier
# that shifts the Tier 2→3 boundary in the routing engine.

CLEAN_THRESHOLD  = 33   # 0-33  → clean
DIRTY_THRESHOLD  = 66   # 67-100 → dirty
                        # 34-66 → medium

# Routing modifier: added to the Tier 2→3 complexity boundary
# Positive = harder to reach Tier 3 (prefer local)
# Zero     = no change
MODIFIER_CLEAN  = 0     # Clean grid: no penalty, route normally
MODIFIER_MEDIUM = 8     # Medium grid: nudge toward local (+8 to boundary)
MODIFIER_DIRTY  = 15    # Dirty grid: strongly prefer local (+15 to boundary)


def classify_intensity(percent: float) -> tuple[GridStatus, int]:
    """
    Classify a WattTime percentile (0-100) into a GridStatus
    and return the corresponding routing modifier.

    Args:
        percent: WattTime signal index (0=cleanest, 100=dirtiest)

    Returns:
        (GridStatus, modifier)
    """
    if percent <= CLEAN_THRESHOLD:
        return GridStatus.CLEAN, MODIFIER_CLEAN
    elif percent <= DIRTY_THRESHOLD:
        return GridStatus.MEDIUM, MODIFIER_MEDIUM
    else:
        return GridStatus.DIRTY, MODIFIER_DIRTY


def classify_gco2(gco2_per_kwh: float) -> tuple[GridStatus, int]:
    """
    Classify a raw gCO2/kWh value (e.g. from electricityMaps) into
    a GridStatus and return the routing modifier.

    Typical CA ranges:
        <150 gCO2/kWh  → clean (lots of solar/wind)
        150-350         → medium
        >350            → dirty (gas peakers, imports)

    Args:
        gco2_per_kwh: Carbon intensity in grams CO2 per kWh

    Returns:
        (GridStatus, modifier)
    """
    if gco2_per_kwh < 150:
        return GridStatus.CLEAN, MODIFIER_CLEAN
    elif gco2_per_kwh < 350:
        return GridStatus.MEDIUM, MODIFIER_MEDIUM
    else:
        return GridStatus.DIRTY, MODIFIER_DIRTY


# ─────────────────────────────────────────────
# CarbonGrid — Main Interface
# ─────────────────────────────────────────────

class CarbonGrid:
    """
    Real-time California grid carbon intensity tracker.

    Polls the WattTime API on an interval, caches the latest reading,
    and provides the routing engine with a modifier to adjust tier thresholds.

    Supports three modes:
        1. WattTime API (primary)    — requires WATTTIME_USERNAME/PASSWORD
        2. electricityMaps (fallback) — requires EMAPS_API_KEY
        3. Mock / simulation          — no API key needed, for development
    """

    def __init__(self, mock: bool = False):
        # Config from environment
        self._mock = mock or os.getenv("CARBON_MOCK", "").lower() == "true"
        self._region = os.getenv("GRID_REGION", "CAISO_NORTH")
        self._poll_interval = int(os.getenv("GRID_POLL_INTERVAL", "300"))

        # WattTime credentials
        self._wt_username = os.getenv("WATTTIME_USERNAME", "")
        self._wt_password = os.getenv("WATTTIME_PASSWORD", "")
        self._wt_token: Optional[str] = None
        self._wt_token_expiry: float = 0  # epoch timestamp

        # electricityMaps fallback
        self._emaps_key = os.getenv("EMAPS_API_KEY", "")

        # Cache
        self._cached_reading: Optional[GridReading] = None
        self._cache_time: float = 0

        # Background poller task handle
        self._poll_task: Optional[asyncio.Task] = None

        # HTTP client (reusable)
        self._client: Optional[httpx.AsyncClient] = None

        if self._mock:
            logger.info("CarbonGrid initialized in MOCK mode")
        elif self._wt_username:
            logger.info(f"CarbonGrid initialized with WattTime (region={self._region})")
        elif self._emaps_key:
            logger.info(f"CarbonGrid initialized with electricityMaps (region=US-CAL-CISO)")
        else:
            logger.warning(
                "CarbonGrid: No API credentials found. Set WATTTIME_USERNAME/PASSWORD, "
                "EMAPS_API_KEY, or CARBON_MOCK=true. Falling back to mock mode."
            )
            self._mock = True

    # ─── Lifecycle ───────────────────────────

    async def start(self):
        """Start the background polling loop. Call this on app startup."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)

        # Fetch immediately on start
        await self._poll_once()

        # Start background loop
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(f"CarbonGrid polling started (every {self._poll_interval}s)")

    async def stop(self):
        """Stop the background polling loop. Call this on app shutdown."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("CarbonGrid polling stopped")

    async def _poll_loop(self):
        """Background loop that refreshes the carbon reading on an interval."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._poll_once()
            except Exception as e:
                logger.error(f"CarbonGrid poll error: {e}")

    async def _poll_once(self):
        """Fetch a fresh reading from the best available source."""
        if self._mock:
            self._cached_reading = self._mock_reading()
        elif self._wt_username:
            self._cached_reading = await self._fetch_watttime()
        elif self._emaps_key:
            self._cached_reading = await self._fetch_electricitymaps()
        else:
            self._cached_reading = self._mock_reading()

        self._cache_time = time.time()
        logger.info(
            f"Carbon update: {self._cached_reading.intensity:.0f} "
            f"({self._cached_reading.status.value}) "
            f"modifier={self._cached_reading.modifier:+d} "
            f"[{self._cached_reading.source}]"
        )

    # ─── Public API ──────────────────────────

    async def get_current(self) -> GridReading:
        """
        Get the current grid carbon reading.

        Returns the cached value if fresh, otherwise fetches a new one.
        This is the main method the routing engine calls.
        """
        # If no reading yet, fetch one
        if self._cached_reading is None:
            await self._poll_once()

        # If cache is stale (2× poll interval), force refresh
        if time.time() - self._cache_time > self._poll_interval * 2:
            try:
                await self._poll_once()
            except Exception as e:
                logger.warning(f"Stale cache refresh failed: {e}, using old reading")

        return self._cached_reading

    def get_cached(self) -> Optional[GridReading]:
        """
        Get the cached reading synchronously (no API call).
        Returns None if no reading has been fetched yet.
        Useful for non-async contexts or quick checks.
        """
        return self._cached_reading

    # ─── WattTime API v3 ─────────────────────

    async def _watttime_login(self) -> str:
        """
        Authenticate with WattTime and return a JWT token.
        Tokens are cached and reused until they expire (~30 min).
        """
        # Return cached token if still valid (refresh 5 min before expiry)
        if self._wt_token and time.time() < self._wt_token_expiry - 300:
            return self._wt_token

        resp = await self._client.get(
            "https://api.watttime.org/login",
            auth=(self._wt_username, self._wt_password),
        )
        resp.raise_for_status()
        data = resp.json()
        self._wt_token = data["token"]
        # WattTime tokens last ~30 minutes
        self._wt_token_expiry = time.time() + 1800
        logger.debug("WattTime token refreshed")
        return self._wt_token

    async def _fetch_watttime(self) -> GridReading:
        """
        Fetch real-time signal index from WattTime API v3.

        Endpoint: GET /v3/signal-index
        Returns a percentile 0-100 (0=cleanest, 100=dirtiest).
        Docs: https://docs.watttime.org/
        """
        try:
            token = await self._watttime_login()
            resp = await self._client.get(
                "https://api.watttime.org/v3/signal-index",
                params={"region": self._region, "signal_type": "co2_moer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract the signal index (percent 0-100)
            # v3 response: {"data": [{"point_time": "...", "value": 42.5, ...}]}
            signal_data = data.get("data", [{}])
            if signal_data:
                percent = float(signal_data[0].get("value", 50))
            else:
                percent = 50.0  # Default to medium if response is unexpected

            status, modifier = classify_intensity(percent)

            return GridReading(
                intensity=percent,
                status=status,
                modifier=modifier,
                timestamp=datetime.now(timezone.utc),
                source="watttime",
                region=self._region,
                raw_data=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"WattTime API error {e.response.status_code}: {e.response.text}")
            # Fall back to electricityMaps if available
            if self._emaps_key:
                logger.info("Falling back to electricityMaps")
                return await self._fetch_electricitymaps()
            raise
        except Exception as e:
            logger.error(f"WattTime fetch failed: {e}")
            if self._emaps_key:
                return await self._fetch_electricitymaps()
            # Last resort: return mock reading
            logger.warning("All APIs failed, returning mock reading")
            return self._mock_reading()

    # ─── electricityMaps API (fallback) ───────

    async def _fetch_electricitymaps(self) -> GridReading:
        """
        Fetch real-time carbon intensity from electricityMaps API.

        Endpoint: GET /v3/carbon-intensity/latest
        Returns gCO2eq/kWh for a given zone.
        Docs: https://static.electricitymaps.com/api/docs/index.html
        """
        try:
            resp = await self._client.get(
                "https://api.electricitymap.org/v3/carbon-intensity/latest",
                params={"zone": "US-CAL-CISO"},
                headers={"auth-token": self._emaps_key},
            )
            resp.raise_for_status()
            data = resp.json()

            gco2 = float(data.get("carbonIntensity", 200))
            status, modifier = classify_gco2(gco2)

            return GridReading(
                intensity=gco2,
                status=status,
                modifier=modifier,
                timestamp=datetime.now(timezone.utc),
                source="electricitymaps",
                region="US-CAL-CISO",
                raw_data=data,
            )

        except Exception as e:
            logger.error(f"electricityMaps fetch failed: {e}")
            logger.warning("Falling back to mock reading")
            return self._mock_reading()

    # ─── Mock / Simulation ────────────────────

    def _mock_reading(self) -> GridReading:
        """
        Generate a simulated carbon reading for development.

        Simulates a realistic California daily pattern:
        - Night (10pm-6am):  Medium-high (gas baseload)
        - Morning (6am-10am): Dropping (solar coming online)
        - Midday (10am-4pm):  Clean (peak solar)
        - Evening (4pm-9pm):  Rising (solar fading, gas ramping)

        This models the famous California "duck curve".
        """
        hour = datetime.now().hour

        if 10 <= hour < 16:
            # Midday: peak solar → clean
            base = random.uniform(5, 25)
        elif 6 <= hour < 10:
            # Morning: solar ramping up
            base = random.uniform(20, 45)
        elif 16 <= hour < 21:
            # Evening: solar fading, demand high → dirty
            base = random.uniform(55, 85)
        else:
            # Night: gas baseload
            base = random.uniform(40, 65)

        # Add some noise
        percent = max(0, min(100, base + random.uniform(-5, 5)))
        status, modifier = classify_intensity(percent)

        return GridReading(
            intensity=round(percent, 1),
            status=status,
            modifier=modifier,
            timestamp=datetime.now(timezone.utc),
            source="mock",
            region=self._region,
            raw_data={"simulated": True, "hour": hour, "note": "CA duck curve simulation"},
        )

    # ─── Helpers ──────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the current reading for API responses."""
        reading = self._cached_reading
        if reading is None:
            return {
                "intensity": None,
                "status": "unknown",
                "modifier": 0,
                "timestamp": None,
                "source": "none",
                "region": self._region,
            }
        return {
            "intensity": reading.intensity,
            "status": reading.status.value,
            "modifier": reading.modifier,
            "timestamp": reading.timestamp.isoformat(),
            "source": reading.source,
            "region": reading.region,
        }

    def __repr__(self) -> str:
        if self._cached_reading:
            r = self._cached_reading
            return (
                f"CarbonGrid({r.status.value}, "
                f"intensity={r.intensity:.0f}, "
                f"modifier={r.modifier:+d}, "
                f"source={r.source})"
            )
        return "CarbonGrid(no reading yet)"
