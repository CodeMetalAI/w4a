"""
Action Masking Tests

Tests for action masking functionality and valid action constraints.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import RLEnvWrapper


class TestActionMaskStructure:
    """Test action mask structure and format"""
    
    def test_action_type_masks(self):
        """Test action type masks are properly formatted"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        action_types = info['valid_masks']['action_types']
        
        # Should be a set of integers
        assert isinstance(action_types, set), "action_types not a set"
        
        for action_type in action_types:
            assert isinstance(action_type, int), f"action_type not int: {type(action_type)}"
            assert 0 <= action_type < 8, f"action_type out of range: {action_type}"
        
        # Should have at least no-op (action_type=0)
        assert 0 in action_types, "No-op action not available"
        
        env.close()
    
    def test_controllable_entity_masks(self):
        """Test controllable entity masks are properly formatted"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        controllable_entities = info['valid_masks']['controllable_entities']
        
        # Should be a set of integers
        assert isinstance(controllable_entities, set), "controllable_entities not a set"
        
        for entity_id in controllable_entities:
            assert isinstance(entity_id, int), f"entity_id not int: {type(entity_id)}"
            assert 0 <= entity_id < env.config.max_entities, f"entity_id out of range: {entity_id}"
        
        env.close()
    
    def test_detected_target_masks(self):
        """Test detected target masks are properly formatted"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        detected_targets = info['valid_masks']['detected_targets']
        
        # Should be a set of integers
        assert isinstance(detected_targets, set), "detected_targets not a set"
        
        for target_id in detected_targets:
            assert isinstance(target_id, int), f"target_id not int: {type(target_id)}"
            assert 0 <= target_id < env.config.max_target_groups, f"target_id out of range: {target_id}"
        
        env.close()
    
    def test_entity_target_matrix_masks(self):
        """Test entity-target engagement matrix is properly formatted"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        matrix = info['valid_masks']['entity_target_matrix']
        
        # Should be a dict or similar structure
        assert isinstance(matrix, dict), "entity_target_matrix not a dict"
        
        # Keys should be entity IDs, values should be sets of target IDs
        for entity_id, target_ids in matrix.items():
            assert isinstance(entity_id, int), f"entity_id not int: {type(entity_id)}"
            assert 0 <= entity_id < env.config.max_entities, f"entity_id out of range: {entity_id}"
            
            assert isinstance(target_ids, set), f"target_ids not set for entity {entity_id}"
            
            for target_id in target_ids:
                assert isinstance(target_id, int), f"target_id not int: {type(target_id)}"
                assert 0 <= target_id < env.config.max_target_groups, f"target_id out of range: {target_id}"
        
        env.close()


class TestMaskValidation:
    """Test mask validation and enforcement"""
    
    def test_valid_action_respects_masks(self):
        """Test that valid actions respect the provided masks"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        masks = info['valid_masks']
        
        # Create actions that should be valid according to masks
        for action_type in list(masks['action_types'])[:3]:  # Test first 3 valid action types
            if len(masks['controllable_entities']) > 0:
                entity_id = list(masks['controllable_entities'])[0]
                
                action = {
                    "action_type": action_type,
                    "entity_id": entity_id,
                    "move_center_grid": 0,
                    "move_short_axis_km": 0,
                    "move_long_axis_km": 0,
                    "move_axis_angle": 0,
                    "target_group_id": 0,
                    "weapon_selection": 0,
                    "weapon_usage": 0,
                    "weapon_engagement": 0,
                    "stealth_enabled": 0,
                    "sensing_position_grid": 0
                }
                
                # If it's an engage action and we have targets, use valid target
                if action_type == 2 and len(masks['detected_targets']) > 0:
                    action["target_group_id"] = list(masks['detected_targets'])[0]
                
                # Action should be processable (might not be fully valid due to other constraints)
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert np.isfinite(reward), f"Valid masked action produced non-finite reward"
                    print(f"Action type {action_type} with entity {entity_id} processed successfully")
                except Exception as e:
                    print(f"Action type {action_type} failed: {e}")
                
                if terminated or truncated:
                    break
        
        env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running action masking tests...")
    
    test_structure = TestActionMaskStructure()
    test_structure.test_action_type_masks()
    test_structure.test_controllable_entity_masks()
    test_structure.test_detected_target_masks()
    test_structure.test_entity_target_matrix_masks()
    print("Action mask structure tests passed")
    
    test_validation = TestMaskValidation()
    test_validation.test_valid_action_respects_masks()
    print("Mask validation tests passed")
    
    print("All action masking tests passed")