"""
Space Tests

Tests for observation space, action space, and action masking.
"""

import pytest
import numpy as np
from gymnasium import spaces
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv


class TestObservationSpace:
    """Test observation space properties and consistency"""
    
    def test_observation_consistency(self):
        """Test observations match observation space"""
        env = TridentIslandEnv()
        
        obs, info = env.reset()
        
        # Observation should match space shape
        assert obs.shape == env.observation_space.shape
        
        # Observation should be within bounds
        assert env.observation_space.contains(obs)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        
        # Observation should be float32
        assert obs.dtype == np.float32
        
        # Test multiple steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert obs.shape == env.observation_space.shape
            assert env.observation_space.contains(obs)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
            assert obs.dtype == np.float32
            
            if terminated or truncated:
                break
        
        env.close()
    
    def test_observation_features_bounded(self):
        """Test that observation features are properly normalized"""
        env = TridentIslandEnv()
        
        # Run multiple episodes to test different states
        for episode in range(3):
            obs, info = env.reset()
            
            # All features should be in [0, 1]
            assert np.all(obs >= 0.0), f"Episode {episode}: Found values < 0: {obs[obs < 0.0]}"
            assert np.all(obs <= 1.0), f"Episode {episode}: Found values > 1: {obs[obs > 1.0]}"
            
            # No NaN or infinite values
            assert np.all(np.isfinite(obs)), f"Episode {episode}: Found non-finite values"
            
            # Test during episode
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                assert np.all(obs >= 0.0), f"Episode {episode}, Step {step}: Found values < 0"
                assert np.all(obs <= 1.0), f"Episode {episode}, Step {step}: Found values > 1"
                assert np.all(np.isfinite(obs)), f"Episode {episode}, Step {step}: Found non-finite values"
                
                if terminated or truncated:
                    break
        
        env.close()


class TestActionSpace:
    """Test action space properties and consistency"""
    
    def test_action_space_structure(self):
        """Test action space has correct hierarchical structure"""
        env = TridentIslandEnv()
        
        # Should be Dict space
        assert isinstance(env.action_space, spaces.Dict)
        
        # Should have expected keys
        expected_keys = {
            "action_type", "entity_id", "move_center_grid", "move_short_axis_km",
            "move_long_axis_km", "move_axis_angle", "target_group_id", 
            "weapon_selection", "weapon_usage", "weapon_engagement",
            "stealth_enabled", "sensing_position_grid"
        }
        assert set(env.action_space.spaces.keys()) == expected_keys
        
        # All sub-spaces should be Discrete
        for key, space in env.action_space.spaces.items():
            assert isinstance(space, spaces.Discrete), f"Action {key} is not Discrete: {type(space)}"
            assert space.n > 0, f"Action {key} has invalid size: {space.n}"
        
        env.close()
    
    def test_action_space_bounds(self):
        """Test action space discrete bounds are reasonable"""
        env = TridentIslandEnv()
        
        # Test specific bounds
        assert env.action_space["action_type"].n == 8  # 0-7 action types
        assert env.action_space["entity_id"].n == env.config.max_entities
        assert env.action_space["stealth_enabled"].n == 2  # 0=off, 1=on
        
        # Grid positions should match calculated grid size
        expected_grid_positions = env.max_grid_positions
        assert env.action_space["move_center_grid"].n == expected_grid_positions
        assert env.action_space["sensing_position_grid"].n == expected_grid_positions + 1
        
        env.close()
    
    def test_action_sampling(self):
        """Test action sampling produces valid actions"""
        env = TridentIslandEnv()
        
        for _ in range(10):
            action = env.action_space.sample()
            
            # Should be a dict
            assert isinstance(action, dict)
            
            # Should contain all required keys
            for key in env.action_space.spaces.keys():
                assert key in action, f"Missing key: {key}"
            
            # Should be valid according to space
            assert env.action_space.contains(action)
            
            # Values should be integers in correct ranges
            for key, value in action.items():
                assert isinstance(value, (int, np.integer)), f"Action {key} not integer: {type(value)}"
                assert 0 <= value < env.action_space[key].n, f"Action {key} out of bounds: {value}"
        
        env.close()


class TestActionMasking:
    """Test action masking functionality"""
    
    def test_masks_consistency_during_episode(self):
        """Test action masks remain consistent during episode"""
        env = TridentIslandEnv()
        obs, info = env.reset()
        
        for step in range(5):
            # Get current masks
            masks = info["valid_masks"]
            
            # Masks should be non-empty (at least no-op should be available)
            assert len(masks["action_types"]) > 0, f"No valid action types at step {step}"
            
            # No-op (action_type=0) should always be available
            assert 0 in masks["action_types"], f"No-op not available at step {step}"
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running space tests...")
    
    test_obs = TestObservationSpace()
    test_obs.test_observation_consistency()
    test_obs.test_observation_features_bounded()
    print("Observation space tests passed")
    
    test_action = TestActionSpace()
    test_action.test_action_space_structure()
    test_action.test_action_space_bounds()
    test_action.test_action_sampling()
    print("Action space tests passed")
    
    test_masks = TestActionMasking()
    test_masks.test_masks_consistency_during_episode()
    print("Action masking tests passed")
    
    print("All space tests passed")