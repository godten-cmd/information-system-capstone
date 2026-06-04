"""Anthropic provider wrapper for the LLM-based retrieval planner."""

from __future__ import annotations

import os
import time

from enterprise_rag.planning.providers.base import BasePlannerProvider, ProviderResponse

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider(BasePlannerProvider):
    """Wraps the Anthropic Messages API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Install 'anthropic' to use AnthropicProvider: pip install anthropic"
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")

        self._model = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=resolved_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.monotonic() - t0

        content = response.content[0].text if response.content else ""
        usage = response.usage
        return ProviderResponse(
            content=content,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            model=response.model,
            latency_s=latency,
        )
