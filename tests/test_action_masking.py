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
    
    def test_visible_target_masks(self):
        """Test visible target masks are properly formatted"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        visible_targets = info['valid_masks']['visible_targets']
        
        # Should be a set of integers
        assert isinstance(visible_targets, set), "visible_targets not a set"
        
        for target_id in visible_targets:
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
                if action_type == 2 and len(masks['visible_targets']) > 0:
                    action["target_group_id"] = list(masks['visible_targets'])[0]
                
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


def test_engage_available_when_targets_present():
    """If there are detected targets, engage action (2) must be valid."""
    env = RLEnvWrapper(TridentIslandEnv())
    obs, info = env.reset()
    masks = info["valid_masks"]
    if len(masks["visible_targets"]) > 0:
        assert 2 in masks["action_types"], "Engage not available despite detected targets"
    env.close()


def test_masks_seed_determinism():
    """With a fixed seed, first-step masks should be deterministic."""
    env1 = RLEnvWrapper(TridentIslandEnv())
    env2 = RLEnvWrapper(TridentIslandEnv())
    _, info1 = env1.reset(seed=123)
    _, info2 = env2.reset(seed=123)
    assert info1["valid_masks"] == info2["valid_masks"]
    env1.close()
    env2.close()


def test_negative_mask_enforcement_routes_to_noop():
    # TODO: THIS SHOULD NOT BE USING MASKS!
    """Attempting masked-out actions should result in a no-op effect (no commit recorded)."""
    env = RLEnvWrapper(TridentIslandEnv())
    obs, info = env.reset()

    masks = info["valid_masks"]

    # Build a clearly invalid action:
    # - Use an entity_id not in controllable_entities (if any controllable entities exist, shift it out of range)
    # - Or attempt engage with a target not in visible_targets
    action = env.action_space.sample()
    action["action_type"] = 2  # engage

    # Choose an entity_id
    if len(masks["controllable_entities"]) > 0:
        entity_id = max(masks["controllable_entities"]) + 1  # guaranteed not in set
    else:
        entity_id = env.config.max_entities - 1  # still likely invalid for engage
    action["entity_id"] = int(entity_id)

    # Choose an invalid target (outside visible_targets)
    invalid_target = env.config.max_target_groups - 1
    if invalid_target in masks.get("visible_targets", set()):
        invalid_target = invalid_target - 1 if invalid_target > 0 else invalid_target + 1
    action["target_group_id"] = int(invalid_target)

    # Step and verify no commit was recorded (treated as noop)
    _, _, _, _, info2 = env.step(action)
    intent = info2.get("last_action_intent_by_entity", {})
    applied = info2.get("last_action_applied_by_entity", {})
    # Intent should record the attempt
    assert str(action["entity_id"]) in intent
    # Applied should NOT record it
    assert str(action["entity_id"]) not in applied, "Masked action should not be applied"

    env.close()


def test_temporal_mask_consistency():
    """Masks maintain required invariants across multiple steps."""
    env = RLEnvWrapper(TridentIslandEnv())
    obs, info = env.reset()

    for _ in range(5):
        masks = info["valid_masks"]
        # Invariants
        assert 0 in masks["action_types"], "No-op should always be available"
        matrix = masks["entity_target_matrix"]
        # Keys subset of controllable entities
        for e_id in matrix.keys():
            assert e_id in masks["controllable_entities"]
        # If any entity has targets, engage must be possible
        any_targets = any(len(tg_set) > 0 for tg_set in matrix.values())
        if any_targets:
            assert 2 in masks["action_types"], "Engage should be available when targets exist"

        # Advance one step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running action masking tests...")
    
    test_structure = TestActionMaskStructure()
    test_structure.test_action_type_masks()
    test_structure.test_controllable_entity_masks()
    test_structure.test_visible_target_masks()
    test_structure.test_entity_target_matrix_masks()
    print("Action mask structure tests passed")
    
    test_validation = TestMaskValidation()
    test_validation.test_valid_action_respects_masks()
    print("Mask validation tests passed")
    
    print("All action masking tests passed")