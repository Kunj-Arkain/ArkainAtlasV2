"""
engine.determinism — Deterministic RNG Seeding (Gap A)
========================================================
All randomness in the engine is seeded from a single source:
  run_id → SHA-256 → int seed → seed all RNG modules

LIVE mode: seed from run_id (reproducible if same run_id)
FROZEN mode: seed from recorded run_id (exact replay)

Every module that uses random MUST call get_rng() instead of
using random.random() directly.
"""

from __future__ import annotations
import hashlib
import random as _random_module
from typing import Optional


# Global engine RNG — all modules use this instead of random.*
_engine_rng: Optional[_random_module.Random] = None
_engine_seed: Optional[int] = None


def seed_engine(run_id: str = None, seed: int = None):
    """Seed all engine randomness. Call once at orchestrator start.

    Priority: explicit seed > run_id-derived > time-based
    """
    global _engine_rng, _engine_seed

    if seed is not None:
        _engine_seed = seed
    elif run_id:
        # Deterministic seed from run_id
        hash_bytes = hashlib.sha256(run_id.encode()).digest()
        _engine_seed = int.from_bytes(hash_bytes[:8], 'big')
    else:
        _engine_seed = None  # Will use system entropy

    _engine_rng = _random_module.Random(_engine_seed)

    # Also seed the global random module for any code that
    # hasn't been migrated to get_rng() yet
    if _engine_seed is not None:
        _random_module.seed(_engine_seed)


def get_rng() -> _random_module.Random:
    """Get the engine's seeded RNG instance.

    All engine code should use this instead of random.random().
    Example: rng = get_rng(); x = rng.random()
    """
    global _engine_rng
    if _engine_rng is None:
        seed_engine()  # Auto-initialize if not seeded
    return _engine_rng


def get_seed() -> Optional[int]:
    """Get the current seed (for recording in replay logs)."""
    return _engine_seed
