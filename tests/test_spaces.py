"""
Space Tests

Tests for observation space, action space, and action masking.
"""

import pytest
import numpy as np
from gymnasium import spaces
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import RLEnvWrapper


class TestObservationSpace:
    """Test observation space properties and consistency"""
    
    def test_observation_consistency(self):
        """Test observations match observation space"""
        env = RLEnvWrapper(TridentIslandEnv())
        
        obs, info = env.reset()
        # Info structure should be present
        assert isinstance(info, dict)
        assert "valid_masks" in info
        vm = info["valid_masks"]
        # Expect the exact keys used by the environment
        required_mask_keys = {"action_types", "controllable_entities", "visible_targets", "entity_target_matrix"}
        assert set(vm.keys()) == required_mask_keys
        
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
            # Sampled actions should be valid for the space
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, info = env.step(action)
            # Maintain info structure during steps
            assert "valid_masks" in info
            
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
        env = RLEnvWrapper(TridentIslandEnv())
        
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
        env = RLEnvWrapper(TridentIslandEnv())
        
        # Should be Dict space
        assert isinstance(env.action_space, spaces.Dict)
        
        # Should have expected keys
        expected_keys = {
            "action_type", "entity_id", "move_center_grid", "move_short_axis_km",
            "move_long_axis_km", "move_axis_angle", "target_group_id", 
            "weapon_selection", "weapon_usage", "weapon_engagement",
            "stealth_enabled", "sensing_position_grid", "refuel_target_id"
        }
        assert set(env.action_space.spaces.keys()) == expected_keys
        
        # All sub-spaces should be Discrete
        for key, space in env.action_space.spaces.items():
            assert isinstance(space, spaces.Discrete), f"Action {key} is not Discrete: {type(space)}"
            assert space.n > 0, f"Action {key} has invalid size: {space.n}"
        
        env.close()
    
    def test_action_space_bounds(self):
        """Test action space discrete bounds are reasonable"""
        env = RLEnvWrapper(TridentIslandEnv())
        
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
        env = RLEnvWrapper(TridentIslandEnv())
        
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

    def test_step_sampled_action_contains(self):
        """During steps, sampled actions must satisfy action_space.contains."""
        env = RLEnvWrapper(TridentIslandEnv())
        env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            _ = env.step(action)
        env.close()

    def test_action_edge_min_max_step(self):
        """Construct min/max edge actions and ensure they are valid and step without error."""
        env = RLEnvWrapper(TridentIslandEnv())
        env.reset()

        # Build min action (all zeros) and max action (n-1 for each key)
        min_action = {}
        max_action = {}
        for key, space in env.action_space.spaces.items():
            assert isinstance(space, spaces.Discrete)
            min_action[key] = 0
            max_action[key] = space.n - 1

        # Validate and step
        assert env.action_space.contains(min_action)
        _ = env.step(min_action)

        assert env.action_space.contains(max_action)
        _ = env.step(max_action)

        env.close()

    def test_action_negative_contains(self):
        """Invalid actions should fail action_space.contains."""
        env = RLEnvWrapper(TridentIslandEnv())
        env.reset()

        # Start from a valid sampled action
        valid = env.action_space.sample()
        assert env.action_space.contains(valid)

        # 1) Missing key
        missing_key_action = dict(valid)
        any_key = next(iter(env.action_space.spaces.keys()))
        missing_key_action.pop(any_key, None)
        assert not env.action_space.contains(missing_key_action)

        # 2) Wrong type
        wrong_type_action = dict(valid)
        wrong_type_action[any_key] = "not-an-int"
        assert not env.action_space.contains(wrong_type_action)

        # 3) Out of range
        out_of_range_action = dict(valid)
        out_of_range_action[any_key] = env.action_space[any_key].n  # n is outside [0, n-1]
        assert not env.action_space.contains(out_of_range_action)

        # 4) Extra key
        extra_key_action = dict(valid)
        extra_key_action["__extra__"] = 0
        assert not env.action_space.contains(extra_key_action)

        env.close()


class TestActionMasking:
    """Test action masking functionality"""
    
    def test_masks_consistency_during_episode(self):
        """Test action masks remain consistent during episode"""
        env = RLEnvWrapper(TridentIslandEnv())
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