"""
Simple Evaluation

Random agent and basic evaluation function.
"""

import numpy as np
import gymnasium as gym
from typing import Optional


class RandomAgent:
    """Random baseline agent"""
    
    def __init__(self, seed: Optional[int] = None):
        self.name = "Random"
        self.rng = np.random.RandomState(seed)
    
    def act(self, observation: np.ndarray, action_space: gym.Space) -> int:
        """Select random action"""
        return action_space.sample()
    
    def reset(self):
        """Reset agent state"""
        pass


def evaluate(agent, env, episodes: int = 10):
    """
    Simple evaluation function.
    
    Args:
        agent: Agent with act() method or callable
        env: Environment to test on
        episodes: Number of episodes to run
        
    Returns:
        dict with basic stats
    """
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            if hasattr(agent, 'act'):
                action = agent.act(obs, env.action_space)
            else:
                action = agent(obs)  # Assume callable
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_steps": np.mean(total_steps),
        "episodes": episodes
    }