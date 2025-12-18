"""
Space Tests

Tests for observation space, action space, and action masking in multiagent environment.
"""

import pytest
import numpy as np
from gymnasium import spaces
from w4a import Config
from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction


class TestObservationSpace:
    """Test observation space properties and consistency for multiagent environment"""
    
    def test_observation_spaces_defined(self):
        """Test observation spaces are properly defined for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        # Check spaces exist
        obs_spaces = env.observation_spaces
        assert "legacy" in obs_spaces
        assert "dynasty" in obs_spaces
        
        # Check spaces are proper Gymnasium spaces
        assert isinstance(obs_spaces["legacy"], spaces.Space)
        assert isinstance(obs_spaces["dynasty"], spaces.Space)
        
        # Both agents should have same observation space structure
        assert obs_spaces["legacy"].shape == obs_spaces["dynasty"].shape
        
        env.close()
    
    def test_observation_consistency(self):
        """Test observations match observation space for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Check both agents get observations
        assert "legacy" in observations
        assert "dynasty" in observations
        
        # Check observations match space
        obs_legacy = observations["legacy"]
        obs_dynasty = observations["dynasty"]
        
        assert env.observation_spaces["legacy"].contains(obs_legacy)
        assert env.observation_spaces["dynasty"].contains(obs_dynasty)
        
        # Check info structure
        assert "valid_masks" in infos["legacy"]
        assert "valid_masks" in infos["dynasty"]
        
        required_mask_keys = {"action_types", "controllable_entities", "visible_targets", "entity_target_matrix", "spawn_components"}
        assert set(infos["legacy"]["valid_masks"].keys()) == required_mask_keys
        assert set(infos["dynasty"]["valid_masks"].keys()) == required_mask_keys
        
        env.close()
    
    def test_observation_size_and_structure(self):
        """Test that observation has correct size based on config parameters"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # 11 global + (max_entities * 52 friendly) + (max_target_groups * 12 enemy)
        expected_size = 11 + (config.max_entities * 52) + (config.max_target_groups * 12)
        
        for agent_name, obs in observations.items():
            assert obs.shape == (expected_size,), f"{agent_name}: Expected shape ({expected_size},), got {obs.shape}"
            assert obs.dtype == np.float32, f"{agent_name}: obs not float32"
            
            # Observations should be finite and within [0, 1] bounds
            assert np.all(np.isfinite(obs)), f"{agent_name}: Observations should be finite"
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"{agent_name}: Observations should be in [0, 1]"
        
        env.close()
    
    def test_observation_features_bounded(self):
        """Test that observation features are properly normalized for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Test both agents' observations
        for agent_name, obs in observations.items():
            assert isinstance(obs, np.ndarray), f"{agent_name}: obs not ndarray"
            assert obs.dtype == np.float32, f"{agent_name}: obs not float32"
            assert np.all(obs >= 0.0), f"{agent_name}: Found values < 0"
            assert np.all(obs <= 1.0), f"{agent_name}: Found values > 1"
            assert np.all(np.isfinite(obs)), f"{agent_name}: Found non-finite values"
        
        # Test during steps
        for step in range(5):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            
            observations, rewards, terms, truncs, infos = env.step(actions)
            
            for agent_name, obs in observations.items():
                assert np.all(obs >= 0.0), f"{agent_name}, step {step}: Found values < 0"
                assert np.all(obs <= 1.0), f"{agent_name}, step {step}: Found values > 1"
                assert np.all(np.isfinite(obs)), f"{agent_name}, step {step}: Found non-finite values"
            
            if terms["legacy"] or truncs["legacy"]:
                break
        
        env.close()


class TestActionSpace:
    """Test action space properties and consistency for multiagent environment"""
    
    def test_action_space_structure(self):
        """Test action space has correct hierarchical structure for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        # Check action spaces exist
        action_spaces = env.action_spaces
        assert "legacy" in action_spaces
        assert "dynasty" in action_spaces
        
        # Both should be Dict spaces
        assert isinstance(action_spaces["legacy"], spaces.Dict)
        assert isinstance(action_spaces["dynasty"], spaces.Dict)
        
        # Should have expected keys
        expected_keys = {
            "action_type", "entity_id", "move_center_grid", "move_short_axis_km",
            "move_long_axis_km", "move_axis_angle", "target_group_id", 
            "weapon_selection", "weapon_usage", "weapon_engagement",
            "stealth_enabled", "sensing_position_grid", "refuel_target_id",
            "entity_to_protect_id", "jam_target_grid", "spawn_component_idx"
        }
        
        assert set(action_spaces["legacy"].spaces.keys()) == expected_keys
        assert set(action_spaces["dynasty"].spaces.keys()) == expected_keys
        
        # All sub-spaces should be Discrete
        for agent_name in ["legacy", "dynasty"]:
            for key, space in action_spaces[agent_name].spaces.items():
                assert isinstance(space, spaces.Discrete), f"{agent_name} action {key} is not Discrete"
                assert space.n > 0, f"{agent_name} action {key} has invalid size: {space.n}"
        
        env.close()
    
    def test_action_space_bounds(self):
        """Test action space discrete bounds are reasonable for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        # Test bounds for both agents (should be identical)
        for agent_name in ["legacy", "dynasty"]:
            action_space = env.action_spaces[agent_name]
            
            assert action_space["action_type"].n == 10
            assert action_space["entity_id"].n == config.max_entities
            assert action_space["stealth_enabled"].n == 2
            
            # Grid positions should match calculated grid size
            expected_grid_positions = env.max_grid_positions
            assert action_space["move_center_grid"].n == expected_grid_positions
            assert action_space["sensing_position_grid"].n == expected_grid_positions + 1
        
        env.close()
    
    def test_action_sampling(self):
        """Test action sampling produces valid actions for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        for agent_name in ["legacy", "dynasty"]:
            action_space = env.action_spaces[agent_name]
            
            for _ in range(10):
                action = action_space.sample()
                
                assert isinstance(action, dict)
                
                # Should contain all required keys
                for key in action_space.spaces.keys():
                    assert key in action, f"{agent_name} missing key: {key}"
                
                # Should be valid according to space
                assert action_space.contains(action)
                
                # Values should be integers in correct ranges
                for key, value in action.items():
                    assert isinstance(value, (int, np.integer)), f"{agent_name} action {key} not integer"
                    assert 0 <= value < action_space[key].n, f"{agent_name} action {key} out of bounds"
        
        env.close()

    def test_action_validity_checks(self):
        """Test action space contains() method works correctly"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        action_space = env.action_spaces["legacy"]
        
        # Valid action
        valid = action_space.sample()
        assert action_space.contains(valid)
        
        # Missing key
        missing_key_action = dict(valid)
        any_key = next(iter(action_space.spaces.keys()))
        missing_key_action.pop(any_key, None)
        assert not action_space.contains(missing_key_action)
        
        # Wrong type
        wrong_type_action = dict(valid)
        wrong_type_action[any_key] = "not-an-int"
        assert not action_space.contains(wrong_type_action)
        
        # Out of range
        out_of_range_action = dict(valid)
        out_of_range_action[any_key] = action_space[any_key].n
        assert not action_space.contains(out_of_range_action)
        
        # Extra key
        extra_key_action = dict(valid)
        extra_key_action["__extra__"] = 0
        assert not action_space.contains(extra_key_action)
        
        env.close()


class TestActionMasking:
    """Test action masking functionality for multiagent environment"""
    
    def test_masks_consistency_during_episode(self):
        """Test action masks remain consistent during episode for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        for step in range(5):
            # Check masks for both agents
            for agent_name in ["legacy", "dynasty"]:
                masks = infos[agent_name]["valid_masks"]
                
                # Masks should be non-empty (at least no-op should be available)
                assert len(masks["action_types"]) > 0, f"{agent_name}: No valid action types at step {step}"
                
                # No-op (action_type=0) should always be available
                assert 0 in masks["action_types"], f"{agent_name}: No-op not available at step {step}"
            
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            
            observations, rewards, terms, truncs, infos = env.step(actions)
            
            if terms["legacy"] or truncs["legacy"]:
                break
        
        env.close()
    
    def test_action_masks_per_agent(self):
        """Test that action masks are specific to each agent"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Each agent should have their own masks
        masks_legacy = infos["legacy"]["valid_masks"]
        masks_dynasty = infos["dynasty"]["valid_masks"]
        
        # Verify mask structure
        assert isinstance(masks_legacy["action_types"], set)
        assert isinstance(masks_dynasty["action_types"], set)
        assert isinstance(masks_legacy["controllable_entities"], set)
        assert isinstance(masks_dynasty["controllable_entities"], set)
        
        # Both agents should have controllable entities
        assert len(masks_legacy["controllable_entities"]) > 0
        assert len(masks_dynasty["controllable_entities"]) > 0
        
        # Entity IDs may be the same range since each agent tracks their own entities
        # What matters is they each have valid entity masks
        assert all(isinstance(eid, int) for eid in masks_legacy["controllable_entities"])
        assert all(isinstance(eid, int) for eid in masks_dynasty["controllable_entities"])
        
        env.close()


if __name__ == "__main__":
    test_obs = TestObservationSpace()
    test_obs.test_observation_spaces_defined()
    test_obs.test_observation_consistency()
    test_obs.test_observation_size_and_structure()
    test_obs.test_observation_features_bounded()
    
    test_action = TestActionSpace()
    test_action.test_action_space_structure()
    test_action.test_action_space_bounds()
    test_action.test_action_sampling()
    test_action.test_action_validity_checks()
    
    test_masks = TestActionMasking()
    test_masks.test_masks_consistency_during_episode()
    test_masks.test_action_masks_per_agent()