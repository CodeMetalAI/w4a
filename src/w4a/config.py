"""
W4A Configuration

User-configurable settings for the environment.
These can be changed without affecting the core simulation.
"""

from dataclasses import dataclass
from typing import Optional

from .constants import TRIDENT_ISLAND_MAP_SIZE


@dataclass
class Config:
    """User-configurable settings for W4A environment"""
    
    # Training parameters
    max_episode_steps: int = 1000 # TODO: This is a placeholder
    # TODO: Add time steps to simulation (10sec is fine)
    
    # Early termination for training efficiency (optional)
    early_termination_enabled: bool = False
    early_win_threshold: float = 0.8  # End episode early if 80% enemies destroyed
    early_loss_threshold: float = 0.2  # End episode early if 80% own forces destroyed
    
    # Environment setup (uses constants but can be overridden)
    map_size: tuple[int, int] = TRIDENT_ISLAND_MAP_SIZE
    
    # Runtime settings
    render_mode: str = "rgb_array"
    debug: bool = False
    seed: Optional[int] = None
    
    # RL-specific settings
    reward_scale: float = 1.0
    normalize_observations: bool = True
    enable_curriculum: bool = False