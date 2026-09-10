"""Ollama client.

Wraps the HTTP API rather than the ``ollama`` Python package so the dependency
surface stays small and streaming, timeouts and model fallback are all explicit.
Swapping models is a config change - ``llm.model`` in ``config.yaml`` or
``LLM_MODEL`` in the environment - with no code change anywhere.

Running the model locally is the point, not a convenience: for a German client
under GDPR, "the documents never leave the machine" is often the difference
between a project that ships and one that dies in a data protection review.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from ragbuilder.config import Config, get_config

LOG = logging.getLogger("ragbuilder.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.latency_ms / 1000)


class OllamaClient:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.host = self.config.llm.host.rstrip("/")
        self.session = requests.Session()
        self._resolved_model: str | None = None

    # ------------------------------------------------------------- discovery
    def available_models(self) -> list[str]:
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001 - callers treat this as "unknown"
            LOG.warning("could not list Ollama models at %s: %s", self.host, exc)
            return []

    def healthy(self) -> bool:
        try:
            return self.session.get(f"{self.host}/api/tags", timeout=5).ok
        except Exception:  # noqa: BLE001
            return False

    def resolve_model(self, requested: str | None = None) -> str:
        """Pick a model that is actually installed.

        Falls back to ``llm.fallback_model`` and then to whatever is present, so
        a fresh clone without ``ollama pull mistral`` still answers instead of
        erroring - and says loudly in the logs which model it used.
        """
        wanted = requested or self.config.llm.model
        if self._resolved_model and requested is None:
            return self._resolved_model

        installed = self.available_models()
        if not installed:
            raise LLMError(
                f"no models available from Ollama at {self.host}. "
                f"Start it with `ollama serve` and run `ollama pull {wanted}`."
            )

        def matches(name: str) -> str | None:
            base = name.split(":")[0]
            for candidate in installed:
                if candidate == name or candidate.split(":")[0] == base:
                    return candidate
            return None

        chosen = matches(wanted) or matches(self.config.llm.fallback_model)
        if chosen is None:
            chosen = installed[0]
            LOG.warning(
                "neither %r nor %r is installed; falling back to %r. "
                "Run `ollama pull %s` for the intended behaviour.",
                wanted,
                self.config.llm.fallback_model,
                chosen,
                wanted,
            )
        elif chosen != wanted:
            LOG.info("model %r resolved to installed model %r", wanted, chosen)

        if requested is None:
            self._resolved_model = chosen
        return chosen

    # ------------------------------------------------------------ generation
    def _options(self, **overrides: Any) -> dict[str, Any]:
        options = {
            "temperature": self.config.llm.temperature,
            "top_p": self.config.llm.top_p,
            "num_ctx": self.config.llm.num_ctx,
            "num_predict": self.config.llm.max_tokens,
        }
        options.update({k: v for k, v in overrides.items() if v is not None})
        return options

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Blocking completion."""
        resolved = self.resolve_model(model)
        payload: dict[str, Any] = {
            "model": resolved,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature=temperature, num_predict=max_tokens),
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        started = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.config.llm.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            raise LLMError(
                f"Ollama timed out after {self.config.llm.timeout_seconds}s. "
                "A 7B model on CPU is slow - raise llm.timeout_seconds or use a smaller model."
            ) from exc
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        completion = Completion(
            text=data.get("response", "").strip(),
            model=resolved,
            latency_ms=latency_ms,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            truncated=data.get("done_reason") == "length",
        )
        LOG.info(
            "generated %d token(s) with %s in %.0f ms (%.1f tok/s)",
            completion.completion_tokens,
            resolved,
            latency_ms,
            completion.tokens_per_second,
        )
        return completion

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield tokens as they are produced.

        Streaming is not cosmetic here: on CPU the first token can take several
        seconds and the full answer 30+, so a UI that waits for the whole
        response feels broken.
        """
        resolved = self.resolve_model(model)
        payload: dict[str, Any] = {
            "model": resolved,
            "prompt": prompt,
            "stream": True,
            "options": self._options(temperature=temperature, num_predict=max_tokens),
        }
        if system:
            payload["system"] = system

        try:
            with self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=True,
                timeout=self.config.llm.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = event.get("response", "")
                    if token:
                        yield token
                    if event.get("done"):
                        break
        except requests.RequestException as exc:
            raise LLMError(f"Ollama streaming failed: {exc}") from exc

    def generate_json(self, prompt: str, system: str | None = None, model: str | None = None) -> Any:
        """Completion parsed as JSON, tolerating the prose LLMs wrap it in."""
        completion = self.generate(prompt, system=system, model=model, json_mode=True)
        text = completion.text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
            end = max(text.rfind("}"), text.rfind("]"))
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            LOG.warning("model did not return valid JSON: %s", text[:200])
            return None


_CLIENT: OllamaClient | None = None


def get_client(config: Config | None = None) -> OllamaClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OllamaClient(config)
    return _CLIENT
