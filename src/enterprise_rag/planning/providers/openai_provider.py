"""OpenAI provider wrapper for the LLM-based retrieval planner."""

from __future__ import annotations

import logging
import os
import time

from enterprise_rag.planning.providers.base import BasePlannerProvider, ProviderResponse

_DEFAULT_MODEL = "gpt-4o-mini"

# GPT-5 and future reasoning models need a much larger max_completion_tokens because
# the budget is shared between internal reasoning tokens and visible output tokens.
# Empirically: GPT-5 uses ~400 reasoning tokens per planner query; 2000 gives safe headroom.
_REASONING_MODELS_MIN_BUDGET = 2000

logger = logging.getLogger(__name__)


def _is_reasoning_model(model: str) -> bool:
    """Return True for models that consume reasoning tokens from the completion budget."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIProvider(BasePlannerProvider):
    """Wraps the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install 'openai' to use OpenAIProvider: pip install openai") from e

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")

        self._model = model or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
        self._client = OpenAI(api_key=resolved_key)

    @property
    def provider_name(self) -> str:
        return "openai"

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
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if _is_reasoning_model(self._model):
            # Reasoning models consume max_completion_tokens for both internal reasoning
            # and visible output. With a small budget (e.g. 200) ALL tokens are used for
            # reasoning and message.content is empty. Use a larger budget so visible output
            # tokens remain after reasoning completes.
            budget = max(max_tokens + _REASONING_MODELS_MIN_BUDGET, _REASONING_MODELS_MIN_BUDGET)
            payload["max_completion_tokens"] = budget
            logger.debug("GPT-5 reasoning model: using max_completion_tokens=%d", budget)
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature

        response = self._client.chat.completions.create(**payload)
        latency = time.monotonic() - t0

        content = response.choices[0].message.content or ""

        if not content and _is_reasoning_model(self._model):
            usage = response.usage
            reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0)
            logger.warning(
                "GPT-5 returned empty content: completion_tokens=%s reasoning_tokens=%s; "
                "budget may still be too small",
                usage.completion_tokens if usage else "?",
                reasoning,
            )

        usage = response.usage
        return ProviderResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            latency_s=latency,
        )
