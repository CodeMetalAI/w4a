"""
Environment Wrapper

A wrapper for custom rewards, actions, and observations.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Callable, Dict, Any


class EnvWrapper(gym.Wrapper):
    """
    Wrapper for customizing reward, action, and observation.
    
    Usage:
        env = TridentIslandEnv()
        wrapped = EnvWrapper(
            env,
            reward_fn=lambda obs, action, reward, info: reward * 2,  # Custom reward
            obs_fn=lambda obs: obs[:10],  # Custom observation
            action_fn=lambda action: action + 1  # Custom action
        )
    """
    
    def __init__(
        self,
        env: gym.Env,
        reward_fn: Optional[Callable] = None,
        obs_fn: Optional[Callable] = None,
        action_fn: Optional[Callable] = None
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            reward_fn: Function to modify rewards: (obs, action, reward, info) -> new_reward
            obs_fn: Function to modify observations: (obs) -> new_obs
            action_fn: Function to modify actions: (action) -> new_action
        """
        super().__init__(env)
        self.reward_fn = reward_fn
        self.obs_fn = obs_fn
        self.action_fn = action_fn
    
    def reset(self, **kwargs):
        """Reset with optional observation transform"""
        obs, info = self.env.reset(**kwargs)
        
        if self.obs_fn:
            obs = self.obs_fn(obs)
            
        return obs, info
    
    def step(self, action):
        """Step with custom transforms"""
        # Transform action if needed
        if self.action_fn:
            action = self.action_fn(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Transform observation if needed
        if self.obs_fn:
            obs = self.obs_fn(obs)
        
        # Transform reward if needed
        if self.reward_fn:
            reward = self.reward_fn(obs, action, reward, info)
        
        return obs, reward, terminated, truncated, info
