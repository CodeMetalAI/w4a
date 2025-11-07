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
        - Detected target groups representing enemy forces (filtered by fog-of-war)
        
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
    
    def get_alive_entities(self) -> List:
        """
        Get alive entities you can command.
        
        Returns:
            List of ControllableEntity objects where is_alive=True
        """
        return [entity for entity in self._sim_agent.controllable_entities.values() 
                if entity.is_alive]
    
    def get_all_entities(self) -> List:
        """
        Get all entities including recently dead ones (appears in observations with health_ok=0.0).
        
        Returns:
            List of all ControllableEntity objects (alive and dead)
        """
        return list(self._sim_agent.controllable_entities.values())
    
    def get_target_groups(self) -> List:
        """
        Get list of detected target groups.
        
        Target groups represent enemy forces visible to this agent. Each target group
        is tagged with this agent's faction ID (e.g., Legacy sees groups with faction=LEGACY).
        
        Returns:
            List of TargetGroup objects visible to this agent.
        """
        return list(self._sim_agent.target_groups.values())
    
    
    def get_entity_by_id(self, entity_id: int):
        """
        Get entity by its ID.
        
        Args:
            entity_id: Entity ID int
            
        Returns:
            ControllableEntity if found, None otherwise
        """
        return self._sim_agent.controllable_entities.get(entity_id)
    
    def get_target_group_by_id(self, group_id: int):
        """
        Get target group by its ID.

        Args:
            group_id: Target group ID int
            
        Returns:
            TargetGroup if found, None otherwise
        """
        return self._sim_agent.target_groups.get(group_id)
        
    def get_capture_capable_entities(self) -> List:
        """
        Get alive entities capable of capturing flags.
        
        Returns:
            List of entities with can_capture=True
        """
        return [e for e in self.get_alive_entities() if e.can_capture]
    
    def get_refuelable_entities(self) -> List:
        """
        Get alive entities capable of receiving fuel.
        
        Returns:
            List of entities with can_refuel=True
        """
        return [e for e in self.get_alive_entities() if e.can_refuel]
    
    def is_entity_capturing(self, entity_id: int) -> bool:
        """
        Check if entity is currently capturing a flag.
        
        Args:
            entity_id: Entity ID from observation or action space
            
        Returns:
            True if entity is actively capturing, False otherwise
        """
        return entity_id in self._sim_agent.active_capturing_entities
    
    def is_entity_refueling(self, entity_id: int) -> bool:
        """
        Check if entity is involved in refueling (giving or receiving).
        
        Args:
            entity_id: Entity ID from observation or action space
            
        Returns:
            True if entity is refueling or being refueled, False otherwise
        """
        return (entity_id in self._sim_agent.active_refuel_receivers or 
                entity_id in self._sim_agent.active_refuel_providers)
    
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

