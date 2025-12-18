"""
W4A - Wargaming for All

A simple, customizable RL environment for wargaming simulation.
"""

__version__ = "0.1.0"

# Core exports
from .constants import *
from .config import Config
from .envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from .agents import CompetitionAgent, SimpleAgent
from .replay import ReplayRecorder, visualize_replay, record_multiagent_episode
from .training.evaluation import evaluate, print_evaluation_results

from .entities import w4a_entities

__all__ = [
    "__version__",
    "Config",
    "TridentIslandMultiAgentEnv",
    "CompetitionAgent",
    "SimpleAgent",
    "ReplayRecorder",
    "visualize_replay",
    "record_multiagent_episode",
    "evaluate",
    "print_evaluation_results",
    "w4a_entities",
]
