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
    grid_resolution_km: int = 50  # Discretized grid resolution in km
    
    # Action space parameters
    max_entities: int = 50  # Maximum entities per side # TODO: Is this reasonable?
    max_weapon_types: int = 8  # Maximum weapon types per entity # TODO: Is this reasonable?
    max_patrol_axis_km: int = 30  # Maximum CAP route long axis length, discretized in 1km increments # TODO: Is this reasonable?
    angle_resolution_degrees: int = 10  # Angle discretization
    
    # Runtime settings
    render_mode: str = "rgb_array"
    debug: bool = False
    seed: Optional[int] = None
    
    # RL-specific settings
    reward_scale: float = 1.0
    normalize_observations: bool = True
    enable_curriculum: bool = False