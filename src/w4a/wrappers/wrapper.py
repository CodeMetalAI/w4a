"""
Environment Wrapper

A wrapper for custom rewards, actions, and observations.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Callable, Dict, Any
from pathlib import Path


class EnvWrapper(gym.Wrapper):
    """
    Wrapper for customizing reward, action, and observation, and loading force configurations.
    
    Usage:
        env = TridentIslandEnv()
        wrapped = EnvWrapper(
            env,
            force_config_path="path/to/force_config.json",       # Force composition JSON
            spawn_areas_path="path/to/spawn_areas.json",         # Spawn areas JSON (optional)
            reward_fn=lambda obs, action, reward, info: reward * 2,  # Custom reward
            obs_fn=lambda obs: obs[:10],  # Custom observation
            action_fn=lambda action: action + 1  # Custom action
        )
    """
    
    def __init__(
        self,
        env: gym.Env,
        force_config_path: Optional[str] = None,
        spawn_areas_path: Optional[str] = None,
        reward_fn: Optional[Callable] = None,
        obs_fn: Optional[Callable] = None,
        action_fn: Optional[Callable] = None
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            force_config_path: Path to JSON file specifying force composition
            spawn_areas_path: Path to JSON file specifying spawn areas 
            reward_fn: Function to modify rewards: (obs, action, reward, info) -> new_reward
            obs_fn: Function to modify observations: (obs) -> new_obs
            action_fn: Function to modify actions: (action) -> new_action
        """
        super().__init__(env)

        # Load configurations if provided
        if force_config_path:
            self.env.load_force_config(force_config_path, spawn_areas_path)
        elif spawn_areas_path:
            # Load just spawn areas if only that is provided
            self.env.load_spawn_areas(spawn_areas_path)
            
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
