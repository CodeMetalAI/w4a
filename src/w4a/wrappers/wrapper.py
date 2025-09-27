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
    Wrapper for customizing reward, action, observation, and optionally overriding force configurations.
    
    Usage:
        env = TridentIslandEnv()
        wrapped = EnvWrapper(
            env,
            reward_fn=lambda obs, action, reward, info: reward * 2,  # Custom reward
            obs_fn=lambda obs: obs[:10],  # Custom observation
            action_fn=lambda action: action + 1  # Custom action
        )
        
    Can also specify custom force configurations:
        env = TridentIslandEnv()
        wrapped = EnvWrapper(
            env,
            legacy_entity_list_path="my_custom/LegacyEntityList.json",
            dynasty_entity_list_path="my_custom/DynastyEntityList.json", 
            legacy_spawn_data_path="my_custom/LegacyEntitySpawnData.json",
            dynasty_spawn_data_path="my_custom/DynastyEntitySpawnData.json",
            reward_fn=lambda obs, action, reward, info: reward * 2
        )
    """
    
    def __init__(
        self,
        env: gym.Env,
        legacy_entity_list_path: Optional[str] = None,
        dynasty_entity_list_path: Optional[str] = None,
        legacy_spawn_data_path: Optional[str] = None,
        dynasty_spawn_data_path: Optional[str] = None,
        reward_fn: Optional[Callable] = None,
        obs_fn: Optional[Callable] = None,
        action_fn: Optional[Callable] = None
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            legacy_entity_list_path: Path to Legacy faction entity composition JSON
            dynasty_entity_list_path: Path to Dynasty faction entity composition JSON
            legacy_spawn_data_path: Path to Legacy faction spawn areas JSON 
            dynasty_spawn_data_path: Path to Dynasty faction spawn areas JSON
            reward_fn: Function to modify rewards: (obs, action, reward, info) -> new_reward
            obs_fn: Function to modify observations: (obs) -> new_obs
            action_fn: Function to modify actions: (action) -> new_action
        """
        super().__init__(env)
        
        # Store JSON file paths for force configuration
        self.legacy_entity_list_path = legacy_entity_list_path
        self.dynasty_entity_list_path = dynasty_entity_list_path
        self.legacy_spawn_data_path = legacy_spawn_data_path
        self.dynasty_spawn_data_path = dynasty_spawn_data_path
            
        self.reward_fn = reward_fn
        self.obs_fn = obs_fn
        self.action_fn = action_fn
        
        # Pass the JSON paths to the environment
        if all([legacy_entity_list_path, dynasty_entity_list_path, 
                legacy_spawn_data_path, dynasty_spawn_data_path]):
            self.env.set_force_config_paths(
                legacy_entity_list_path, dynasty_entity_list_path,
                legacy_spawn_data_path, dynasty_spawn_data_path
            )
    
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
