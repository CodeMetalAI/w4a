"""
W4A - Wargaming for All

A simple, customizable RL environment for wargaming simulation.
"""

__version__ = "0.1.0"

# Core exports
from .constants import *
from .config import Config

from .envs.trident_island_env import TridentIslandEnv
from .envs.force_design_env import ForceDesignEnv
from .envs.actions import *
from .wrappers.wrapper import EnvWrapper  
from .training.replay import ReplayRecorder
from .training.evaluation import RandomAgent, evaluate

# TODO: Decide public api
__all__ = [
    "__version__",
    "Config",
    "TridentIslandEnv",
    "ForceDesignEnv", 
    "EnvWrapper",
    "ReplayRecorder",
    "RandomAgent",
    "evaluate",
]
