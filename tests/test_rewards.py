"""
Reward Tests

Tests for reward calculation and customization.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import EnvWrapper


class TestRewardCustomization:
    """Test reward customization through wrappers"""
    
    def test_wrapper_reward_modification(self):
        """Test reward modification through EnvWrapper"""
        base_env = TridentIslandEnv()
        
        # Create wrapper with custom reward function
        def add_constant_reward(obs, action, reward, info):
            return reward + 5.0
        
        wrapped_env = EnvWrapper(base_env, reward_fn=add_constant_reward)
        
        # Compare rewards
        base_env.reset(seed=42)
        wrapped_env.reset(seed=42)
        
        for step in range(3):
            action = base_env.action_space.sample()
            
            # Step both environments with same action
            _, base_reward, base_term, base_trunc, _ = base_env.step(action)
            _, wrapped_reward, wrapped_term, wrapped_trunc, _ = wrapped_env.step(action)
            
            # Wrapped reward should be base reward + 5
            expected_wrapped = base_reward + 5.0
            assert abs(wrapped_reward - expected_wrapped) < 1e-10, \
                f"Wrapper reward incorrect: {wrapped_reward} != {expected_wrapped}"
            
            if base_term or base_trunc:
                break
        
        base_env.close()
        wrapped_env.close()


class TestRewardSignals:
    """Test reward signals for different game events"""
    
    def test_reward_changes_with_actions(self):
        """Test that different actions can produce different rewards"""
        env = TridentIslandEnv()
        
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