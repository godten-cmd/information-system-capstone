"""Factory for creating planner providers from config or environment."""

from __future__ import annotations

import os

from enterprise_rag.planning.providers.base import BasePlannerProvider


def create_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    prompt_version: str = "structured",
) -> BasePlannerProvider:
    """Create a provider instance.

    Provider selection order when `provider` is None:
    1. OPENAI_API_KEY set → OpenAIProvider
    2. ANTHROPIC_API_KEY set → AnthropicProvider
    3. Otherwise → MockProvider

    Args:
        provider: "openai", "anthropic", or "mock". None = auto-detect.
        model: Override model name (uses env var or provider default if None).
        api_key: Override API key (uses env var if None).
        prompt_version: Passed to MockProvider for accuracy simulation.
    """
    resolved = provider or _auto_detect()

    if resolved == "openai":
        from enterprise_rag.planning.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model, api_key=api_key)

    if resolved == "anthropic":
        from enterprise_rag.planning.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model, api_key=api_key)

    from enterprise_rag.planning.providers.mock_provider import MockProvider
    return MockProvider(prompt_version=prompt_version)


def _auto_detect() -> str:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "mock"
