"""Abstract base class for LLM provider wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    """Normalised response returned by every provider."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_s: float


class BasePlannerProvider(ABC):
    """Unified interface for OpenAI, Anthropic, and Mock providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Call the underlying LLM and return a normalised response."""
