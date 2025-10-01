"""
Environment Mechanics Tests

Tests to verify that environment state (obs, info, masks, entities) 
is properly updated when actions are applied.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import RLEnvWrapper


class TestActionTracking:
    """Test action tracking system (intended vs applied actions)"""
    
    def test_valid_action_tracking(self):
        """Test that valid actions are both intended and applied"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find a controllable entity
        controllable_entities = info["valid_masks"]["controllable_entities"]
        assert len(controllable_entities) > 0, "No controllable entities found"
        entity_id = list(controllable_entities)[0]
        
        # Create valid move action
        action = {
            "action_type": 1, "entity_id": entity_id, "move_center_grid": 5,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify action intent recorded correctly
        intent = info["last_action_intent_by_entity"]
        assert str(entity_id) in intent
        assert intent[str(entity_id)]["action_type"] == 1
        assert "time" in intent[str(entity_id)]
        
        # Verify action was applied
        applied = info["last_action_applied_by_entity"]
        assert str(entity_id) in applied
        
        env.close()
    
    def test_invalid_entity_opposite_faction(self):
        """Test action with entity from opposite faction"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find an entity not in controllable mask (opposite faction)
        controllable_entities = info["valid_masks"]["controllable_entities"]
        all_entity_ids = set(env.env.entities.keys())
        opposite_faction_entities = all_entity_ids - controllable_entities
        
        assert len(opposite_faction_entities) > 0, "No opposite faction entities found"
        invalid_entity_id = list(opposite_faction_entities)[0]
        
        # Create action with opposite faction entity
        action = {
            "action_type": 1, "entity_id": invalid_entity_id, "move_center_grid": 5,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify action intent recorded correctly
        intent = info["last_action_intent_by_entity"]
        assert str(invalid_entity_id) in intent
        assert intent[str(invalid_entity_id)]["action_type"] == 1
        assert "time" in intent[str(invalid_entity_id)]
        
        # Verify action was NOT applied
        applied = info["last_action_applied_by_entity"]
        assert str(invalid_entity_id) not in applied
        
        env.close()
    
    def test_engage_with_invalid_target(self):
        """Test engage action with target not in visible_targets"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find controllable entity
        controllable_entities = info["valid_masks"]["controllable_entities"]
        assert len(controllable_entities) > 0, "No controllable entities found"
        entity_id = list(controllable_entities)[0]
        
        # Find invalid target (not in visible_targets)
        visible_targets = info["valid_masks"]["visible_targets"]
        invalid_target_id = env.config.max_target_groups - 1
        while invalid_target_id in visible_targets and invalid_target_id >= 0:
            invalid_target_id -= 1
        
        # Create engage action with invalid target
        action = {
            "action_type": 2, "entity_id": entity_id, "move_center_grid": 0,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": invalid_target_id, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify action intent recorded correctly
        intent = info["last_action_intent_by_entity"]
        assert str(entity_id) in intent
        assert intent[str(entity_id)]["action_type"] == 2
        
        # Verify action was NOT applied (routes to no-op)
        applied = info["last_action_applied_by_entity"]
        assert str(entity_id) not in applied
        
        env.close()
    
    def test_invalid_action_type(self):
        """Test action type not in action_types mask"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find controllable entity
        controllable_entities = info["valid_masks"]["controllable_entities"]
        assert len(controllable_entities) > 0, "No controllable entities found"
        entity_id = list(controllable_entities)[0]
        
        # Find invalid action type
        valid_action_types = info["valid_masks"]["action_types"]
        invalid_action_type = 7
        while invalid_action_type in valid_action_types and invalid_action_type >= 0:
            invalid_action_type -= 1
        
        # Create action with invalid action type
        action = {
            "action_type": invalid_action_type, "entity_id": entity_id, "move_center_grid": 0,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify action intent recorded correctly
        intent = info["last_action_intent_by_entity"]
        assert str(entity_id) in intent
        assert intent[str(entity_id)]["action_type"] == invalid_action_type
        
        # Verify action was NOT applied (routes to no-op)
        applied = info["last_action_applied_by_entity"]
        assert str(entity_id) not in applied
        
        env.close()


class TestKillTracking:
    """Test friendly and enemy kill tracking"""
    
    def test_friendly_entity_death_tracking(self):
        """Test that friendly entity deaths are tracked correctly"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find a friendly entity
        our_faction = env.config.our_faction
        friendly_entity = None
        for entity in env.env.entities.values():
            if entity.faction.value == our_faction and entity.is_alive:
                friendly_entity = entity
                break
        
        assert friendly_entity is not None, "No living friendly entity found"
        
        # Record initial kill count
        initial_friendly_kills = len(env.env.friendly_kills)
        initial_info_kills = info["mission"]["friendly_kills"]
        
        # Manually kill the entity
        # TODO: Is it safe to manipulate is_alive directly?
        friendly_entity.is_alive = False
        
        # Step to trigger mission metrics update
        action = {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify entity appears in friendly_kills
        assert friendly_entity in env.env.friendly_kills
        
        # Verify info dict reflects the kill
        assert info["mission"]["friendly_kills"] == initial_info_kills + 1
        
        env.close()
    
    def test_enemy_entity_death_tracking(self):
        """Test that enemy entity deaths are tracked correctly"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find an enemy entity
        our_faction = env.config.our_faction
        enemy_entity = None
        for entity in env.env.entities.values():
            if entity.faction.value != our_faction and entity.is_alive:
                enemy_entity = entity
                break
        
        assert enemy_entity is not None, "No living enemy entity found"
        
        # Record initial kill count
        initial_enemy_kills = len(env.env.enemy_kills)
        initial_info_kills = info["mission"]["enemy_kills"]
        
        # Manually kill the entity
        # TODO: Is it safe to manipulate is_alive directly?
        enemy_entity.is_alive = False
        
        # Step to trigger mission metrics update
        action = {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify entity appears in enemy_kills
        assert enemy_entity in env.env.enemy_kills
        
        # Verify info dict reflects the kill
        assert info["mission"]["enemy_kills"] == initial_info_kills + 1
        
        env.close()


class TestTerminationConditions:
    """Test termination conditions"""
    
    def test_capture_based_termination(self):
        """Test termination based on capture completion"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Stub capture completion conditions
        env.env.capture_timer_progress = env.config.capture_required_seconds
        env.env.capture_possible = True
        env.env.island_contested = False
        
        # Step to trigger termination check
        action = {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify termination occurred
        assert terminated == True, "Capture completion should trigger termination"
        
        env.close()
    
    def test_kill_ratio_win_termination(self):
        """Test termination based on kill ratio win condition"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Stub kill ratio win by manipulating kill sets
        # Add many enemy kills, few friendly kills
        enemy_entities = [e for e in env.env.entities.values() if e.faction.value != env.config.our_faction]
        for i, entity in enumerate(enemy_entities[:5]):  # Kill first 5 enemies
            env.env.enemy_kills.add(entity)
        
        # Step to trigger termination check
        action = {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if high kill ratio triggers termination
        kill_ratio = info["mission"]["kill_ratio"]
        if kill_ratio >= env.config.kill_ratio_threshold:
            assert terminated == True, "High kill ratio should trigger termination"
        
        env.close()
    
    def test_kill_ratio_loss_termination(self):
        """Test termination based on kill ratio loss condition"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Stub kill ratio loss conditions
        env.env.capture_possible = False  # No capture path
        
        # Add many friendly kills, few enemy kills
        friendly_entities = [e for e in env.env.entities.values() if e.faction.value == env.config.our_faction]
        for i, entity in enumerate(friendly_entities[:3]):  # Kill first 3 friendlies
            env.env.friendly_kills.add(entity)
        
        # Step to trigger termination check
        action = {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if poor kill ratio + no capture path triggers termination
        kill_ratio = info["mission"]["kill_ratio"]
        inverse_threshold = 1.0 / max(1e-6, env.config.kill_ratio_threshold)
        if kill_ratio <= inverse_threshold:
            assert terminated == True, "Poor kill ratio with no capture should trigger termination"
        
        env.close()


class TestActionExecution:
    """Test execution of all action types and info dict updates"""
    
    def test_no_op_action(self):
        """Test no-op action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Record initial state
        initial_time = info["time_elapsed"]
        initial_step = info["step"]
        
        # Execute no-op
        action = {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify time progression
        assert info["time_elapsed"] > initial_time
        assert info["step"] == initial_step + 1
        
        # Verify action tracking
        intent = info["last_action_intent_by_entity"]
        assert "0" in intent
        assert intent["0"]["action_type"] == 0
        
        env.close()
    
    def test_move_action(self):
        """Test move action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find controllable entity
        controllable_entities = info["valid_masks"]["controllable_entities"]
        assert len(controllable_entities) > 0, "No controllable entities found"
        entity_id = list(controllable_entities)[0]
        
        # Execute move action
        action = {"action_type": 1, "entity_id": entity_id, "move_center_grid": 10,
                 "move_short_axis_km": 1, "move_long_axis_km": 2, "move_axis_angle": 1,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify action tracking
        intent = info["last_action_intent_by_entity"]
        assert str(entity_id) in intent
        assert intent[str(entity_id)]["action_type"] == 1
        
        applied = info["last_action_applied_by_entity"]
        assert str(entity_id) in applied
        
        # Verify entity remains controllable
        assert entity_id in info["valid_masks"]["controllable_entities"]
        
        # TODO: Check entity position changes in obs space
        
        env.close()
    
    def test_engage_action(self):
        """Test engage action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find controllable entity and visible target
        controllable_entities = info["valid_masks"]["controllable_entities"]
        visible_targets = info["valid_masks"]["visible_targets"]
        
        if len(controllable_entities) > 0 and len(visible_targets) > 0:
            entity_id = list(controllable_entities)[0]
            target_id = list(visible_targets)[0]
            
            # Execute engage action
            action = {"action_type": 2, "entity_id": entity_id, "move_center_grid": 0,
                     "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                     "target_group_id": target_id, "weapon_selection": 0, "weapon_usage": 0,
                     "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                     "refuel_target_id": 0}
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify action tracking
            intent = info["last_action_intent_by_entity"]
            assert str(entity_id) in intent
            assert intent[str(entity_id)]["action_type"] == 2
            
            applied = info["last_action_applied_by_entity"]
            assert str(entity_id) in applied
            
            # TODO: Check entity engagement status in obs space
            # TODO: Monitor mission metrics for potential kill updates
        
        env.close()
    
    def test_stealth_action(self):
        """Test stealth action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find entity with stealth capability
        controllable_entities = info["valid_masks"]["controllable_entities"]
        stealth_capable_entity = None
        
        for entity_id in controllable_entities:
            entity = env.env.entities[entity_id]
            # Look for stealth capability indicators
            if entity.Entity.Identity in ["F-22", "F-35C", "J-20", "J-35", "B-21"]:  # Stealth aircraft
                stealth_capable_entity = entity_id
                break
        
        if stealth_capable_entity is not None:
            # Execute stealth action
            action = {"action_type": 3, "entity_id": stealth_capable_entity, "move_center_grid": 0,
                     "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                     "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                     "weapon_engagement": 0, "stealth_enabled": 1, "sensing_position_grid": 0,
                     "refuel_target_id": 0}
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify action tracking
            intent = info["last_action_intent_by_entity"]
            assert str(stealth_capable_entity) in intent
            assert intent[str(stealth_capable_entity)]["action_type"] == 3
            
            applied = info["last_action_applied_by_entity"]
            assert str(stealth_capable_entity) in applied
            
            # TODO: Check entity stealth status in obs space
            # TODO: Check radar-related mask changes
        
        env.close()
    
    def test_sensing_action(self):
        """Test sensing action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find entity with radar capability
        controllable_entities = info["valid_masks"]["controllable_entities"]
        radar_capable_entity = None
        
        for entity_id in controllable_entities:
            entity = env.env.entities[entity_id]
            # Look for radar capability indicators
            if entity.Entity.Identity in ["E-2D", "E-7", "KJ-500", "KJ-600", "F-22", "F-35C"]:  # AWACS/Fighter radars
                radar_capable_entity = entity_id
                break
        
        if radar_capable_entity is not None:
            # Execute sensing action
            action = {"action_type": 4, "entity_id": radar_capable_entity, "move_center_grid": 0,
                     "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                     "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                     "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 15,
                     "refuel_target_id": 0}
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify action tracking
            intent = info["last_action_intent_by_entity"]
            assert str(radar_capable_entity) in intent
            assert intent[str(radar_capable_entity)]["action_type"] == 4
            
            applied = info["last_action_applied_by_entity"]
            assert str(radar_capable_entity) in applied
            
            # TODO: Check entity radar focus in obs space
            # TODO: Check if visible_targets mask changes
            # TODO: Check entity_target_matrix for sensing updates
        
        env.close()
    
    def test_capture_action(self):
        """Test capture action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find controllable entity (capture typically requires transport aircraft)
        controllable_entities = info["valid_masks"]["controllable_entities"]
        capture_capable_entity = None
        
        for entity_id in controllable_entities:
            entity = env.env.entities[entity_id]
            # Look for transport/capture capability
            if entity.Entity.Identity in ["C-130", "Y-9"]:  # Transport aircraft
                capture_capable_entity = entity_id
                break
        
        if capture_capable_entity is not None:
            # Execute capture action
            action = {"action_type": 5, "entity_id": capture_capable_entity, "move_center_grid": 0,
                     "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                     "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                     "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                     "refuel_target_id": 0}
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify action tracking
            intent = info["last_action_intent_by_entity"]
            assert str(capture_capable_entity) in intent
            assert intent[str(capture_capable_entity)]["action_type"] == 5
            
            # Check capture-related info updates
            mission_info = info["mission"]
            assert "capture_progress" in mission_info
            assert "capture_possible" in mission_info
            assert "island_contested" in mission_info
        
        env.close()
    
    def test_rtb_action(self):
        """Test RTB action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find controllable entity
        controllable_entities = info["valid_masks"]["controllable_entities"]
        assert len(controllable_entities) > 0, "No controllable entities found"
        entity_id = list(controllable_entities)[0]
        
        # Execute RTB action
        action = {"action_type": 6, "entity_id": entity_id, "move_center_grid": 0,
                 "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                 "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                 "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                 "refuel_target_id": 0}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify action tracking
        intent = info["last_action_intent_by_entity"]
        assert str(entity_id) in intent
        assert intent[str(entity_id)]["action_type"] == 6
        
        # Verify entity remains controllable
        assert entity_id in info["valid_masks"]["controllable_entities"]
        
        # TODO: Check entity status changes in obs space
        
        env.close()
    
    def test_refuel_action(self):
        """Test refuel action execution and info updates"""
        env = RLEnvWrapper(TridentIslandEnv())
        obs, info = env.reset()
        
        # Find refuel receiver and provider
        refuel_info = info["refuel"]
        receivers = refuel_info["receivers"]
        providers = refuel_info["providers"]
        
        if len(receivers) > 0 and len(providers) > 0:
            receiver_id = receivers[0]
            provider_id = providers[0]
            
            # Execute refuel action
            action = {"action_type": 7, "entity_id": receiver_id, "move_center_grid": 0,
                     "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                     "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                     "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                     "refuel_target_id": provider_id}
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify action tracking
            intent = info["last_action_intent_by_entity"]
            assert str(receiver_id) in intent
            assert intent[str(receiver_id)]["action_type"] == 7
            
            # Verify refuel info structure remains
            assert "refuel" in info
            assert "receivers" in info["refuel"]
            assert "providers" in info["refuel"]
            
            # TODO: Check entity fuel status in obs space
        
        env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running environment mechanics tests...")
    
    test_tracking = TestActionTracking()
    test_tracking.test_valid_action_tracking()
    test_tracking.test_invalid_entity_opposite_faction()
    test_tracking.test_engage_with_invalid_target()
    test_tracking.test_invalid_action_type()
    print("Action tracking tests passed")
    
    test_kills = TestKillTracking()
    test_kills.test_friendly_entity_death_tracking()
    test_kills.test_enemy_entity_death_tracking()
    print("Kill tracking tests passed")
    
    test_termination = TestTerminationConditions()
    test_termination.test_capture_based_termination()
    test_termination.test_kill_ratio_win_termination()
    test_termination.test_kill_ratio_loss_termination()
    print("Termination condition tests passed")
    
    test_actions = TestActionExecution()
    test_actions.test_no_op_action()
    test_actions.test_move_action()
    test_actions.test_engage_action()
    test_actions.test_stealth_action()
    test_actions.test_sensing_action()
    test_actions.test_capture_action()
    test_actions.test_rtb_action()
    test_actions.test_refuel_action()
    print("Action execution tests passed")
    
    print("All environment mechanics tests passed")
