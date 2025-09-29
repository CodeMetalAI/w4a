"""
Agent Wrapper

A simple wrapper system for creating custom RL agents that can customize 
rewards, observations, and actions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from ..envs.trident_island_env import TridentIslandEnv
from ..config import Config
from .agent import TridentIslandAgent


class BaseAgent(TridentIslandAgent):
    """
    Abstract base class for custom agents.
    
    Inherits from TridentIslandAgent to provide simulation integration
    while allowing customization of rewards and observations.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize base agent"""
        self.config = config or Config()
        
        # Convert config faction to Faction enum
        from SimulationInterface import Faction
        faction = Faction.DYNASTY if self.config.our_faction == 1 else Faction.LEGACY
        
        super().__init__(faction)
        
    @abstractmethod
    def compute_reward(self, obs: np.ndarray, action: Dict, reward: float, 
                      info: Dict, prev_obs: Optional[np.ndarray] = None) -> float:
        """
        Customize reward function.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Base reward from environment  
            info: Info dict from environment step
            prev_obs: Previous observation (optional)
            
        Returns:
            Customized reward value
        """
        pass
    
    def process_observation(self, obs: np.ndarray, info: Dict) -> np.ndarray:
        """
        Process/transform observations.
        
        Args:
            obs: Raw observation from environment
            info: Info dict with valid action masks and game state
            
        Returns:
            Processed observation
        """
        # Default: return observation as-is
        return obs
    
    def get_observation_space(self, base_obs_space: spaces.Space) -> spaces.Space:
        """
        Customize observation space.
        
        Args:
            base_obs_space: Base observation space from environment
            
        Returns:
            Customized observation space
        """
        # Default: use base observation space
        return base_obs_space
    
    def get_action_space(self, base_action_space: spaces.Space) -> spaces.Space:
        """
        Customize action space.
        
        Args:
            base_action_space: Base action space from environment
            
        Returns:
            Customized action space
        """
        # Default: use base action space
        return base_action_space


class AgentWrapper(gym.Wrapper):
    """
    Simple wrapper for custom agents.
    
    Integrates custom reward/observation logic with the simulation while
    ensuring proper partial observability through TridentIslandAgent.
    
    Usage:
        class MyAgent(BaseAgent):
            def compute_reward(self, obs, action, reward, info, prev_obs=None):
                return reward * 2  # Custom reward logic
        
        env = TridentIslandEnv()
        agent = MyAgent()  # Faction comes from config automatically
        wrapped_env = AgentWrapper(env, agent)
    """
    
    def __init__(self, env: TridentIslandEnv, custom_agent: BaseAgent):
        """
        Initialize agent wrapper.
        
        Args:
            env: TridentIslandEnv to wrap
            custom_agent: Custom agent implementation (inherits from TridentIslandAgent)
        """
        super().__init__(env)
        self.custom_agent = custom_agent
        
        # Override action and observation spaces with custom agent customizations
        self.action_space = custom_agent.get_action_space(env.action_space)
        self.observation_space = custom_agent.get_observation_space(env.observation_space)
        
        # Track previous observation for reward computation
        self.prev_obs = None
    
    def reset(self, **kwargs):
        """Reset environment with custom agent customizations"""
        obs, info = self.env.reset(**kwargs)
        
        # Add custom agent to simulation for proper partial observability
        if self.env.simulation:
            self.env.simulation.add_agent(self.custom_agent)
        
        obs = self.custom_agent.process_observation(obs, info)
        self.prev_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        return obs, info
    
    def step(self, action):
        """Execute step with custom agent customizations"""
        # Execute step in base environment (action should already be in correct format)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Process observation through custom agent  
        obs = self.custom_agent.process_observation(obs, info)
        
        # Compute custom reward
        custom_reward = self.custom_agent.compute_reward(
            obs, action, reward, info, self.prev_obs
        )
        
        # Update previous observation
        self.prev_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        
        return obs, custom_reward, terminated, truncated, info


# =============================================================================
# Example Usage
# =============================================================================

class ExampleAgent(BaseAgent):
    """Example custom agent showing reward shaping and observation processing."""
    
    def compute_reward(self, obs, action, reward, info, prev_obs=None):
        """Custom reward function with multiple components."""
        custom_reward = reward
        
        # Bonus for successful engagements
        if info.get('last_events_count', 0) > 0:
            custom_reward += 10.0
        
        # Bonus for maintaining units
        entity_count = info.get('controllable_entities_count', 0)
        custom_reward += entity_count * 0.1
        
        # Penalty for no-ops when actions are available
        valid_actions = info.get('valid_masks', {}).get('action_types', set())
        if action.get('action_type', 0) == 0 and len(valid_actions) > 1:
            custom_reward -= 1.0
        
        return custom_reward
    
    def process_observation(self, obs, info):
        """Optional: Transform observations."""
        # Example: You could normalize, filter, or extract features here
        return obs


def create_wrapped_environment(agent_class, config=None, **agent_kwargs):
    """
    Convenience function showing the standard setup pattern.
    
    Usage:
        # Factory setup
        config = Config()
        env = create_wrapped_environment(ExampleAgent, config=config)
        
        # Manual setup 
        config = Config()
        env = TridentIslandEnv(config=config)
        agent = ExampleAgent(config=config)
        wrapped_env = AgentWrapper(env, agent)
        
        # Training with any RL framework:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", wrapped_env, verbose=1)
        model.learn(total_timesteps=10000)
    """
    if config is None:
        config = Config()
    
    env = TridentIslandEnv(config=config)
    agent = agent_class(config=config, **agent_kwargs)
    
    return AgentWrapper(env, agent)
