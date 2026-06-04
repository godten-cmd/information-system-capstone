"""Seeded random-number generator wrapper for deterministic document generation.

All generators receive a ``SeededRNG`` instance. Each category forks an
independent child RNG via ``fork(salt)`` so adding or removing documents in
one category does not shift the random state of any other category.
"""

from __future__ import annotations

import random
from typing import Any, Sequence, TypeVar

T = TypeVar("T")


class SeededRNG:
    """Thin wrapper around :class:`random.Random` with a ``fork`` method."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    # ── Delegation to standard random methods ────────────────────────────────

    def choice(self, seq: Sequence[T]) -> T:
        return self._rng.choice(seq)

    def choices(self, seq: Sequence[T], k: int = 1) -> list[T]:
        return self._rng.choices(seq, k=k)

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        return self._rng.sample(seq, k)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def random(self) -> float:
        return self._rng.random()

    def shuffle(self, lst: list[Any]) -> None:
        self._rng.shuffle(lst)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def weighted_choice(self, items: Sequence[T], weights: Sequence[float]) -> T:
        return self._rng.choices(items, weights=list(weights), k=1)[0]

    # ── Fork ──────────────────────────────────────────────────────────────────

    def fork(self, salt: str) -> "SeededRNG":
        """Return an independent child RNG whose seed is derived from the
        current RNG state XOR'd with the hash of *salt*.

        Forking guarantees that the random stream of one category generator
        is independent of how many documents another category generates.
        """
        state_sample = self._rng.randint(0, 2**31 - 1)
        child_seed = state_sample ^ (hash(salt) & 0x7FFF_FFFF)
        return SeededRNG(child_seed)
