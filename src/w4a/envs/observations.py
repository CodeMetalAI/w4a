"""
Observations module

Globals-only observation builder and encoder.

This module currently encodes only the global features requested:
- time_remaining_norm
- kills_for_norm, kills_against_norm
- awacs_alive_flag
- capture_timer_norm
- island_contested_flag
- capture_possible_flag

Placeholders are used where simulation-backed values are not yet wired.
"""

from typing import Any

import numpy as np
from gymnasium import spaces


def build_observation_space(config) -> spaces.Box:
    """Build the observation space for globals-only vector.

    Returns a Box of shape (7,) with values in [0, 1].
    """
    # 7 globals as listed in the module docstring
    low = np.zeros((7,), dtype=np.float32)
    high = np.ones((7,), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def compute_observation(env: Any) -> np.ndarray:
    """Compute the globals-only observation vector."""
    # time remaining normalized
    max_steps = max(1, env.config.max_episode_steps)
    steps_left = max(0, max_steps - env.current_step)
    time_remaining_norm = float(steps_left) / float(max_steps)

    # kill tallies normalized by max_entities
    denom = float(max(1, env.config.max_entities))
    kills_for = float(env.friendly_kills)
    kills_against = float(env.enemy_kills)
    kills_for_norm = np.clip(kills_for / denom, 0.0, 1.0)
    kills_against_norm = np.clip(kills_against / denom, 0.0, 1.0)

    # AWACS alive flag
    awacs_alive_flag = 1.0 if _awacs_alive(env) else 0.0

    # Capture timer normalized by required capture time
    required_capture_time = env.config.capture_required_seconds
    capture_timer_remaining = float(env.capture_timer_remaining)
    if required_capture_time <= 0.0:
        capture_timer_norm = 0.0
    else:
        capture_timer_norm = float(np.clip(capture_timer_remaining / required_capture_time, 0.0, 1.0))

    # Island contested and capture possible flags
    island_contested_flag = 1.0 if env.island_contested else 0.0
    capture_possible_flag = 1.0 if env.capture_possible else 0.0

    obs = np.array([
        time_remaining_norm,
        kills_for_norm,
        kills_against_norm,
        awacs_alive_flag,
        capture_timer_norm,
        island_contested_flag,
        capture_possible_flag,
    ], dtype=np.float32)

    return obs


def _awacs_alive(env: Any) -> bool:
    """Return True if an AWACS-equivalent friendly radar asset is alive.

    Scan for any friendly, alive entity with has_radar=True.
    TODO: Refine with explicit AWACS role/type when available.
    """
    our_faction = env.config.our_faction
    for entity in env.entities.values():
        if (entity.is_alive
                and entity.has_radar
                and entity.faction.value == our_faction):
            return True
    return False

    
