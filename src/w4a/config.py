"""
W4A Configuration

User-configurable settings for the environment.
These can be changed without affecting the core simulation.
"""

from dataclasses import dataclass
from typing import Optional

from .constants import *

from pathlib import Path

@dataclass
class Config:
    """User-configurable settings for W4A environment"""
    
    # Training parameters  
    max_game_time: float = 100000.0  # @Sanjna: we cap this now in the simulation
    capture_required_seconds: float = CAPTURE_REQUIRED_SECONDS

    # Early termination for training efficiency (optional)
    early_termination_enabled: bool = False
    early_win_threshold: float = 0.8  # End episode early if 80% enemies destroyed
    early_loss_threshold: float = 0.2  # End episode early if 80% own forces destroyed
    early_termination_capture_seconds: float = 600.0  # Capture time in seconds

    # Environment setup (uses constants but can be overridden)
    map_size_km: tuple[int, int] = TRIDENT_ISLAND_MAP_SIZE
    grid_resolution_km: int = 50  # Discretized grid resolution in km
    
    # Action space parameters
    max_entities: int = 100  # Maximum entities per faction in observation/action space 
    max_target_groups: int = 50  # Maximum target groups in scenario
    max_weapons: int = 5  # Maximum weapons any entity can have across all domains (typically 2)
    max_weapon_combinations: int = 2**5 - 1  # 31 combinations for action space
    
    # CAP route parameters
    min_patrol_axis_km: int = 100  # Minimum CAP route long axis length
    max_patrol_axis_km: int = 1000  # Maximum CAP route long axis length  
    patrol_axis_increment_km: int = 25  # CAP route discretization increment
    angle_resolution_degrees: int = 10  # Angle discretization
    
    # Runtime settings
    render_mode: str = "rgb_array"
    debug: bool = False
    seed: Optional[int] = None # We override this during adjudication
    
    # RL-specific settings
    reward_scale: float = 1.0
    normalize_observations: bool = True
    enable_curriculum: bool = False

    scenario_path = Path(__file__).parent / "scenarios"

    # We override these two during adjudication
    legacy_force_laydown_path: str = scenario_path / "force_laydown" / "W4A_ForceLaydown_Legacy.json"
    dynasty_force_laydown_path: str = scenario_path / "force_laydown" / "W4A_ForceLaydown_Dynasty.json"
    
    @property
    def max_episode_steps(self) -> int:
        """Derive max episode steps from max game time (10 seconds per step)"""
        return int(self.max_game_time / 10)