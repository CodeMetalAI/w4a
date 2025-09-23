"""
W4A - Wargaming for All

A simple, customizable RL environment for wargaming simulation.
"""

__version__ = "0.1.0"

# Core exports
from .constants import *
from .config import Config

# TODO: Uncomment when ready
# from .trident_island_env import TridentIslandEnv
# from .wrapper import EnvWrapper  
# from .replay import ReplayRecorder
# from .evaluation import RandomAgent, evaluate
# from .force_design_env import ForceDesignEnv

__all__ = [
    "__version__",
    "Config",
]
