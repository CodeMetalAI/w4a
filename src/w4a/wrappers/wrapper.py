"""
RL wrappers:

- RLAgent: minimal simulation Agent (passive for training)
- RLEnvWrapper: user-facing wrapper with overridable obs/action/reward methods and adversary toggle
"""

import gymnasium as gym
from typing import Optional

from SimulationInterface import Agent as SimAgent
from SimulationInterface import Faction

# Simple built-in adversary
from .simple_agent import SimpleAgent


class RLAgent(SimAgent):
    """
    Minimal simulation Agent for RL integration.

    - Inherits SimulationInterface.Agent
    - Sets faction from Config (or explicit)
    - Passive during training (no PlayerEvents)
    """

    def __init__(self, config, faction: Optional[Faction] = None):
        super().__init__()
        self.config = config
        if faction is None:
            self.faction = Faction.DYNASTY if getattr(config, "our_faction", 0) == 1 else Faction.LEGACY
        else:
            self.faction = faction

    def pre_simulation_tick(self, simulation_data):
        # Passive during training; in eval/competition this can compile PlayerEvents from policy
        # actions to coordinate multiple agents inside the simulation loop.
        simulation_data.player_events = []


class RLEnvWrapper(gym.Wrapper):
    """
    RL Environment Wrapper

    - Lets users customize observation, action, reward
    - Installs real simulation Agents before reset
      • Student side: provided `agent` or a passive RLAgent
      • Opponent side: SimpleAgent if `enable_adversary`, else a passive RLAgent
    - Optional: override force composition/spawn JSONs

    Usage:
        from w4a.envs.trident_island_env import TridentIslandEnv

        env = TridentIslandEnv()
        wrapped = RLEnvWrapper(
            env,
            agent=None,                # or RLAgent(env.config)
            enable_adversary=True      # toggle SimpleAgent opponent
        )
        # Train with any RL library (e.g., Ray RLlib, SB3 PPO, etc.)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        agent: Optional[SimAgent] = None,
        enable_adversary: bool = True,
        legacy_entity_list_path: Optional[str] = None,
        dynasty_entity_list_path: Optional[str] = None,
        legacy_spawn_data_path: Optional[str] = None,
        dynasty_spawn_data_path: Optional[str] = None,
    ):
        super().__init__(env)

        self.agent = agent
        self.enable_adversary = enable_adversary

        # Optional force configuration JSONs
        self.legacy_entity_list_path = legacy_entity_list_path
        self.dynasty_entity_list_path = dynasty_entity_list_path
        self.legacy_spawn_data_path = legacy_spawn_data_path
        self.dynasty_spawn_data_path = dynasty_spawn_data_path

        if all([
            legacy_entity_list_path,
            dynasty_entity_list_path,
            legacy_spawn_data_path,
            dynasty_spawn_data_path,
        ]):
            self.env.set_force_config_paths(
                legacy_entity_list_path,
                dynasty_entity_list_path,
                legacy_spawn_data_path,
                dynasty_spawn_data_path,
            )

    def _setup_agents(self):

        is_legacy = getattr(self.env.config, "our_faction", 0) == 1
        user_faction = Faction.LEGACY if is_legacy else Faction.DYNASTY
        opponent_faction = Faction.DYNASTY if not is_legacy else Faction.LEGACY

        # User agent: provided or passive RLAgent
        user_agent = self.agent or RLAgent(self.env.config, faction=user_faction)

        # Opponent agent: SimpleAgent or passive RLAgent (to keep sim happy)
        if self.enable_adversary:
            opponent_agent = SimpleAgent(opponent_faction)
        else:
            opponent_agent = RLAgent(self.env.config, faction=opponent_faction)

        if is_legacy:
            self.env.legacy_agent = user_agent
            self.env.dynasty_agent = opponent_agent
        else:
            self.env.legacy_agent = opponent_agent
            self.env.dynasty_agent = user_agent

    def reset(self, **kwargs):
        # Setup simulation Agents before creating/initializing the sim
        self._setup_agents()

        obs, info = self.env.reset(**kwargs)
        obs = self.obs_fn(obs)
        return obs, info

    def step(self, action):
        action = self.action_fn(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.obs_fn(obs)
        reward = self.reward_fn(obs, action, reward, info)

        return obs, reward, terminated, truncated, info

    def obs_fn(self, obs):
        return obs

    def action_fn(self, action):
        return action

    def reward_fn(self, obs, action, reward, info):
        return reward
