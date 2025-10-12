"""
Public API for competition agents.

Users inherit from CompetitionAgent to build their AI for the competition.
This class provides a minimal, clean interface while hiding simulation complexity.
"""

from typing import Dict, List, Any
from SimulationInterface import Faction
from ._simulation_agent import _SimulationAgentImpl


class CompetitionAgent:
    """
    Base class for competition agents.
    
    Users inherit this class and override methods to implement their AI strategy.
    
    Basic Usage Example:
        ```python
        from w4a.agents import CompetitionAgent, SimpleAgent
        from w4a.envs import TridentIslandMultiAgentEnv
        from SimulationInterface import Faction
        
        # Simple training loop
        env = TridentIslandMultiAgentEnv()
        agent = SimpleAgent(Faction.LEGACY, env.config)
        opponent = SimpleAgent(Faction.DYNASTY, env.config)
        env.set_agents(agent, opponent)
        
        obs, info = env.reset()
        for step in range(1000):
            actions = {
                "legacy": agent.select_action(obs["legacy"]),
                "dynasty": opponent.select_action(obs["dynasty"])
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            if terminated["legacy"]:
                obs, info = env.reset()
        ```
    """
    
    def __init__(self, faction: Faction, config):
        """
        Initialize agent for the given faction.
        
        Args:
            faction: Faction.LEGACY or Faction.DYNASTY
            config: Environment configuration (read-only)
        """
        # Internal simulation agent (HIDDEN from users)
        self._sim_agent = _SimulationAgentImpl(faction, config)
        self._env = None
        
    @property
    def faction(self) -> Faction:
        """The faction this agent controls (read-only)."""
        return self._sim_agent.faction
        
    @property
    def config(self):
        """Environment configuration (read-only)."""
        return self._sim_agent.config
    
    @property
    def action_space(self):
        """Action space for this agent (read-only)."""
        if self._env is None:
            raise RuntimeError("Agent not registered with environment. Call env.set_agents() first.")
        return self._env.action_spaces[self.faction.name.lower()]
    
    @property
    def observation_space(self):
        """Observation space for this agent (read-only)."""
        if self._env is None:
            raise RuntimeError("Agent not registered with environment. Call env.set_agents() first.")
        return self._env.observation_spaces[self.faction.name.lower()]
    
    # === Methods users can override ===
    
    def get_observation(self):
        """
        Get the current observation for this agent.
        
        Default implementation uses environment's observation builder
        which encodes this agent's visible state (entities, target_groups, flags).
        
        The observation is normalized to [0, 1] and includes:
        - Global mission state (time, kills, capture progress)
        - This agent's controllable entities (filtered by fog-of-war)
        - Detected enemy target groups (filtered by fog-of-war)
        
        Users can override to customize observation format.
        See OBSERVATION_SPACE.md for detailed documentation.
        
        Returns:
            Observation (numpy array with shape from observation_space)
        """
        if self._env is None:
            raise RuntimeError("Agent not registered with environment. Call env.set_agents() first.")
        
        # Use environment's observation builder
        from ..envs import observations
        return observations.compute_observation(self._env, self)
    
    def select_action(self, observation):
        """
        Select an action based on observation.
        
        Default implementation returns a random valid action.
        Users should override this with their RL policy or heuristic.
        
        Args:
            observation: Current observation (from get_observation)
            
        Returns:
            Action dict (see ACTION_SPACE.md for detailed documentation)
        """
        # Default: return random action from action space
        return self.action_space.sample()
    
    def get_entities(self) -> List:
        """
        Get list of entities controlled by this agent.
        
        Returns:
            List of ControllableEntity objects that are alive.
        """
        return [entity for entity in self._sim_agent.controllable_entities.values() 
                if entity.is_alive]
    
    def get_target_groups(self) -> List:
        """
        Get list of detected enemy target groups.
        
        Returns:
            List of TargetGroup objects visible to this agent.
        """
        return list(self._sim_agent.target_groups.values())
    
    def calculate_reward(self, env) -> float:
        """
        Calculate per-step reward for this agent.
        
        Default: Returns 0.0 (sparse terminal rewards only).
        Override to implement custom reward shaping.
        
        Args:
            env: Environment instance
            
        Returns:
            Per-step reward (float)
        """
        return 0.0
    
    # === Internal hooks (called by environment) ===
    
    def _get_sim_agent(self):
        """Return internal simulation agent (for environment use only)."""
        return self._sim_agent
    
    def _set_env(self, env):
        """
        Link agent to environment (called by environment).
        
        Args:
            env: Environment instance
        """
        self._env = env

