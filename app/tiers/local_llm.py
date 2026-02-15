"""
tiers/local_llm.py — Tier 2: On-Device AI Inference via ExecuTorch

Uses Meta's ExecuTorch framework (PyTorch on-device inference) with a
quantized Qwen3-1.7B model from the hackathon sponsor's HuggingFace.

Primary:  optimum-executorch  → ExecuTorchModelForCausalLM  (.pte runtime)
Fallback: transformers        → AutoModelForCausalLM        (pure PyTorch)

Both use the identical HuggingFace generate() API so the rest of the
routing system doesn't care which backend is active.

Usage:
    from app.tiers.local_llm import LocalLLM

    llm = LocalLLM()
    await llm.setup()

    result = await llm.generate("Explain quantum computing simply")
    # result.text       -> str
    # result.tokens     -> int  (tokens generated)
    # result.latency_ms -> float
    # result.backend    -> "executorch" | "transformers"

    async for token in llm.stream("Write a haiku about AI"):
        print(token, end="", flush=True)

    await llm.close()

Environment variables:
    LOCAL_MODEL_ID      — HuggingFace model ID (default: see EXECUTORCH_MODEL)
    LOCAL_MAX_TOKENS    — Default max new tokens (default: 250)
    LOCAL_TEMPERATURE   — Default temperature (default: 0.7)
    LOCAL_DEVICE        — Force device: cpu / cuda / mps (default: auto)
    (No mock mode — always uses real ExecuTorch or transformers backend)
"""

import os
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────

# Primary: quantized Qwen3-1.7B from Meta ExecuTorch hackathon sponsor
# Uses 8-bit dynamic activations + 4-bit weights (8da4w) for efficiency
EXECUTORCH_MODEL = "larryliu0820/Qwen3-1.7B-INT8-INT4-ExecuTorch-XNNPACK"

# Fallback: same architecture, standard PyTorch weights
TRANSFORMERS_MODEL = "Qwen/Qwen3-1.7B"

# Energy proxy constants (estimated kWh per request)
ENERGY_LOCAL_KWH = 0.0025   # ~2.5 Wh for local inference
ENERGY_FRONTIER_KWH = 0.008 # ~8 Wh for a frontier API call (for comparison)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class GenerateResult:
    """Result from a single generation call."""
    text: str                           # Generated response text
    tokens: int = 0                     # Number of tokens generated
    latency_ms: float = 0.0            # Wall-clock time in milliseconds
    backend: str = "unknown"            # "executorch" | "transformers"
    model_id: str = ""                  # Which model was used
    energy_kwh: float = ENERGY_LOCAL_KWH  # Estimated energy consumption
    prompt_tokens: int = 0              # Tokens in the input prompt

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "backend": self.backend,
            "model_id": self.model_id,
            "energy_kwh": self.energy_kwh,
        }


@dataclass
class LLMMetrics:
    """Cumulative metrics for the local LLM."""
    total_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_energy_kwh: float = 0.0
    avg_tokens_per_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_energy_kwh": round(self.total_energy_kwh, 6),
            "avg_tokens_per_sec": round(self.avg_tokens_per_sec, 1),
        }


# ─────────────────────────────────────────────
# Local LLM Class
# ─────────────────────────────────────────────

class LocalLLM:
    """
    Tier 2 Local LLM — On-device inference via ExecuTorch + Qwen3.

    Automatically detects whether optimum-executorch is available:
    - If yes: loads pre-exported .pte model via ExecuTorchModelForCausalLM
    - If no:  falls back to standard transformers AutoModelForCausalLM
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
    ):
        # Model selection
        self._executorch_model_id = model_id or os.getenv(
            "LOCAL_MODEL_ID", EXECUTORCH_MODEL
        )
        self._transformers_model_id = TRANSFORMERS_MODEL

        # Generation defaults
        self._max_tokens = max_tokens or int(os.getenv("LOCAL_MAX_TOKENS", "250"))
        self._temperature = temperature or float(os.getenv("LOCAL_TEMPERATURE", "0.7"))

        # System prompt — keeps Tier 2 responses concise and energy-efficient
        self._system_prompt = (
            "You are a helpful assistant. Give clear, concise answers. "
            "Avoid unnecessary formatting like headers, horizontal rules, or long lists. "
            "Get straight to the point."
        )
        self._device = device or os.getenv("LOCAL_DEVICE", "auto")

        # Internal state
        self._model = None
        self._tokenizer = None
        self._backend: str = "unknown"
        self._active_model_id: str = ""
        self._ready = False
        self._metrics = LLMMetrics()

        # Thread pool for running sync inference off the event loop
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")

    # ─── Lifecycle ────────────────────────────

    async def setup(self) -> None:
        """
        Initialize the model. Tries backends in order:
        1. ExecuTorch (optimum-executorch) — sponsor tech, .pte runtime
        2. Transformers (HuggingFace) — pure PyTorch fallback
        """
        # Try ExecuTorch first
        if await self._try_executorch():
            return

        # Fall back to transformers
        if await self._try_transformers():
            return

        # No backend available — this is a fatal error
        raise RuntimeError(
            "LocalLLM: no backend available. Install torch + transformers "
            "or optimum-executorch."
        )

    async def _try_executorch(self) -> bool:
        """Attempt to load model via optimum-executorch."""
        try:
            logger.info(
                "LocalLLM: attempting ExecuTorch backend with %s",
                self._executorch_model_id,
            )

            def _load():
                from optimum.executorch import ExecuTorchModelForCausalLM
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self._executorch_model_id
                )
                model = ExecuTorchModelForCausalLM.from_pretrained(
                    self._executorch_model_id
                )
                return model, tokenizer

            loop = asyncio.get_event_loop()
            self._model, self._tokenizer = await loop.run_in_executor(
                self._executor, _load
            )
            self._backend = "executorch"
            self._active_model_id = self._executorch_model_id
            self._ready = True
            logger.info(
                "LocalLLM: ExecuTorch backend ready — %s",
                self._executorch_model_id,
            )
            return True

        except Exception as exc:
            logger.warning("LocalLLM: ExecuTorch not available — %s", exc)
            return False

    async def _try_transformers(self) -> bool:
        """Attempt to load model via standard HuggingFace transformers."""
        try:
            logger.info(
                "LocalLLM: attempting transformers backend with %s",
                self._transformers_model_id,
            )

            def _load():
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Resolve device
                if self._device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                else:
                    device = self._device

                tokenizer = AutoTokenizer.from_pretrained(
                    self._transformers_model_id,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self._transformers_model_id,
                    dtype=torch.float32,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True,
                )
                if device == "cpu":
                    model = model.to(device)

                model.eval()
                return model, tokenizer

            loop = asyncio.get_event_loop()
            self._model, self._tokenizer = await loop.run_in_executor(
                self._executor, _load
            )
            self._backend = "transformers"
            self._active_model_id = self._transformers_model_id
            self._ready = True
            logger.info(
                "LocalLLM: transformers backend ready — %s",
                self._transformers_model_id,
            )
            return True

        except Exception as exc:
            logger.warning("LocalLLM: transformers not available — %s", exc)
            return False

    async def close(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._ready = False
        self._executor.shutdown(wait=False)
        logger.info("LocalLLM: closed")

    # ─── Generation ───────────────────────────

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerateResult:
        """
        Generate a complete response for the given prompt.

        Args:
            prompt: The user's query text
            max_tokens: Max new tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)

        Returns:
            GenerateResult with text, metrics, and backend info
        """
        if not self._ready:
            raise RuntimeError("LocalLLM not initialized — call setup() first")

        max_tok = max_tokens or self._max_tokens
        temp = temperature if temperature is not None else self._temperature

        start = time.perf_counter()

        def _run():
            # Format as chat with system prompt for concise Tier 2 responses
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Try chat template first, fall back to raw prompt
            try:
                text_input = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                text_input = prompt

            # ── ExecuTorch backend ──────────────────────
            # API: generate(prompt_tokens: List[int], echo=False,
            #               max_seq_len=None) -> List[int]
            # With echo=False (default), returns ONLY new tokens (no prompt).
            if self._backend == "executorch":
                token_ids = self._tokenizer.encode(text_input)
                prompt_len = len(token_ids)
                new_token_ids = self._model.generate(
                    prompt_tokens=token_ids,
                    max_seq_len=prompt_len + max_tok,
                )
                text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
                return text.strip(), len(new_token_ids), prompt_len

            # ── Transformers backend ────────────────────
            inputs = self._tokenizer(text_input, return_tensors="pt")

            import torch
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]

            gen_kwargs = {
                "max_new_tokens": max_tok,
                "do_sample": temp > 0,
                "temperature": temp if temp > 0 else None,
                "top_p": 0.9 if temp > 0 else None,
            }
            # Remove None values
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            outputs = self._model.generate(**inputs, **gen_kwargs)

            # Decode only the new tokens (skip the prompt)
            new_tokens = outputs[0][prompt_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text.strip(), len(new_tokens), prompt_len

        loop = asyncio.get_event_loop()
        text, num_tokens, prompt_tokens = await loop.run_in_executor(
            self._executor, _run
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        was_limited = num_tokens >= max_tok
        finalized_text = self._finalize_text(text, was_limited)

        result = GenerateResult(
            text=finalized_text,
            tokens=num_tokens,
            prompt_tokens=prompt_tokens,
            latency_ms=elapsed_ms,
            backend=self._backend,
            model_id=self._active_model_id,
            energy_kwh=ENERGY_LOCAL_KWH,
        )

        # Update cumulative metrics
        self._update_metrics(result)

        return result

    def _finalize_text(self, text: str, was_limited: bool) -> str:
        """
        Keep responses readable when generation reaches token cap.
        If capped and likely unfinished, trim to the last complete sentence.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        if not was_limited:
            return cleaned

        end_punct = ".!?"
        if cleaned[-1] in end_punct:
            return cleaned

        last_end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
        if last_end > 80:
            return cleaned[: last_end + 1].strip()

        # If no solid sentence boundary, at least avoid dangling punctuation.
        return cleaned.rstrip(",;:-").strip()

    async def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens one at a time for responsive UX.

        Yields individual tokens/words as they're generated.
        Falls back to chunked full-generation if streaming not supported.
        """
        if not self._ready:
            raise RuntimeError("LocalLLM not initialized — call setup() first")

        max_tok = max_tokens or self._max_tokens
        temp = temperature if temperature is not None else self._temperature

        # ── ExecuTorch: no TextIteratorStreamer, use chunked fallback ──
        if self._backend == "executorch":
            result = await self.generate(prompt, max_tok, temp)
            words = result.text.split()
            for i, word in enumerate(words):
                yield word + ("" if i == len(words) - 1 else " ")
                await asyncio.sleep(0.01)
            return

        # ── Transformers: native streaming with TextIteratorStreamer ──
        try:
            from transformers import TextIteratorStreamer
            import threading

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                text_input = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                text_input = prompt

            inputs = self._tokenizer(text_input, return_tensors="pt")

            import torch
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_tok,
                "do_sample": temp > 0,
                "streamer": streamer,
            }
            if temp > 0:
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = 0.9

            thread = threading.Thread(
                target=self._model.generate, kwargs=gen_kwargs
            )
            thread.start()

            for token_text in streamer:
                if token_text:
                    yield token_text

            thread.join()

        except (ImportError, Exception) as exc:
            # Fallback: generate full response, yield in chunks
            logger.debug("Streaming not available (%s), using chunked fallback", exc)
            result = await self.generate(prompt, max_tok, temp)
            words = result.text.split()
            for i, word in enumerate(words):
                yield word + ("" if i == len(words) - 1 else " ")
                await asyncio.sleep(0.01)

    # ─── Metrics ──────────────────────────────

    def _update_metrics(self, result: GenerateResult) -> None:
        """Update cumulative metrics after a generation."""
        self._metrics.total_requests += 1
        self._metrics.total_tokens += result.tokens
        self._metrics.total_latency_ms += result.latency_ms
        self._metrics.total_energy_kwh += result.energy_kwh

        if self._metrics.total_latency_ms > 0:
            self._metrics.avg_tokens_per_sec = (
                self._metrics.total_tokens
                / (self._metrics.total_latency_ms / 1000)
            )

    def get_metrics(self) -> dict:
        """Return current cumulative metrics."""
        return {
            **self._metrics.to_dict(),
            "backend": self._backend,
            "model_id": self._active_model_id,
            "ready": self._ready,
        }

    # ─── Status ───────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_id(self) -> str:
        return self._active_model_id

    def __repr__(self) -> str:
        return (
            f"LocalLLM(backend={self._backend!r}, "
            f"model={self._active_model_id!r}, "
            f"ready={self._ready})"
        )
