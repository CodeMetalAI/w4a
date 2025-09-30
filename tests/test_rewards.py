"""
Reward Tests

Tests for reward calculation and customization.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import RLEnvWrapper


class TestRewardCustomization:
    """Test reward customization through wrappers"""
    
    def test_wrapper_reward_modification(self):
        """Test reward modification via overriding wrapper.reward_fn"""
        class ConstantRewardWrapper(RLEnvWrapper):
            def reward_fn(self, obs, action, reward, info):
                return 10.0

        wrapped_env = ConstantRewardWrapper(TridentIslandEnv())

        wrapped_env.reset(seed=42)

        action = wrapped_env.action_space.sample()
        _, wrapped_reward, _, _, _ = wrapped_env.step(action)

        assert isinstance(wrapped_reward, (int, float))
        assert abs(wrapped_reward - 10.0) < 1e-9

        wrapped_env.close()


class TestRewardSignals:
    """Test reward signals for different game events"""
    
    def test_reward_changes_with_actions(self):
        """Test that different actions can produce different rewards"""
        env = RLEnvWrapper(TridentIslandEnv())
        
        rewards_by_action_type = {}
        
        for episode in range(3):
            obs, info = env.reset()
            
            for step in range(10):
                action = env.action_space.sample()
                action_type = action["action_type"]
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if action_type not in rewards_by_action_type:
                    rewards_by_action_type[action_type] = []
                rewards_by_action_type[action_type].append(reward)
                
                if terminated or truncated:
                    break
        
        # Should have collected rewards for different action types
        assert len(rewards_by_action_type) > 1, "No variety in action types tested"
        
        # Print reward statistics by action type
        for action_type, rewards in rewards_by_action_type.items():
            if rewards:
                mean_reward = np.mean(rewards)
                print(f"Action type {action_type}: mean reward = {mean_reward:.3f} ({len(rewards)} samples)")
        
        env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running reward tests...")
    
    test_custom = TestRewardCustomization()
    test_custom.test_wrapper_reward_modification()
    print("Reward customization tests passed")
    
    test_signals = TestRewardSignals()
    test_signals.test_reward_changes_with_actions()
    print("Reward signals tests passed")
    
    print("All reward tests passed")