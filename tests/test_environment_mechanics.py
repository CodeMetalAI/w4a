"""
Environment Mechanics Tests

Tests to verify that environment state (obs, info, masks, entities) 
is properly updated when actions are applied in the multiagent environment.
"""

import pytest
import numpy as np

from w4a import Config
from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.envs.mission_metrics import update_capture_progress
from w4a.agents import CompetitionAgent, SimpleAgent

from SimulationInterface import Faction


class TestActionTracking:
    """Test action tracking system (intended vs applied actions)"""
    
    def test_valid_action_tracking_legacy(self):
        """Test that valid actions are both intended and applied for legacy agent"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Find a controllable entity for legacy
        controllable_entities = infos["legacy"]["valid_masks"]["controllable_entities"]
        assert len(controllable_entities) > 0, "No controllable entities found"
        entity_id = list(controllable_entities)[0]
        
        # Create valid move action
        action_legacy = {
            "action_type": 1, "entity_id": entity_id, "move_center_grid": 5,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action intent recorded correctly
        intent = infos["legacy"]["last_action_intent_by_entity"]
        assert len(intent) > 0, "No action intent recorded"
        
        # Find our entity in the intent dict
        found_intent = False
        for key, value in intent.items():
            if value["entity_id"] == entity_id:
                assert value["action_type"] == 1
                assert "time" in value
                found_intent = True
                break
        
        assert found_intent, f"Intent for entity {entity_id} not found"
        
        env.close()
    
    def test_invalid_entity_not_applied(self):
        """Test action with invalid entity is not applied"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Use an invalid entity ID (out of range)
        invalid_entity_id = config.max_entities + 100
        
        # Create action with invalid entity
        action_legacy = {
            "action_type": 1, "entity_id": invalid_entity_id, "move_center_grid": 5,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded in intent
        intent = infos["legacy"]["last_action_intent_by_entity"]
        assert len(intent) > 0, "Intent should be recorded"
        
        # Verify action was NOT applied (no entry in applied dict for this entity)
        applied = infos["legacy"]["last_action_applied_by_entity"]
        # Applied should not have this invalid entity
        for key, value in applied.items():
            assert value.get("entity_id") != invalid_entity_id, "Invalid entity should not be applied"
        
        env.close()


class TestKillTracking:
    """Test friendly and enemy kill tracking"""
    
    def test_kill_tracking_updates_info(self):
        """Test that kill counts are tracked in info dict"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Record initial kill counts
        initial_legacy_casualties = infos["legacy"]["mission"]["my_casualties"]
        initial_dynasty_casualties = infos["dynasty"]["mission"]["my_casualties"]
        
        # Step environment a few times
        for _ in range(5):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Verify kill tracking fields exist
            assert "my_casualties" in infos["legacy"]["mission"]
            assert "enemy_casualties" in infos["legacy"]["mission"]
            assert "kill_ratio" in infos["legacy"]["mission"]
            
            assert "my_casualties" in infos["dynasty"]["mission"]
            assert "enemy_casualties" in infos["dynasty"]["mission"]
            assert "kill_ratio" in infos["dynasty"]["mission"]
            
            # Verify symmetry: legacy's enemy casualties = dynasty's my casualties
            assert infos["legacy"]["mission"]["enemy_casualties"] == infos["dynasty"]["mission"]["my_casualties"]
            assert infos["dynasty"]["mission"]["enemy_casualties"] == infos["legacy"]["mission"]["my_casualties"]
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()


class TestEntityDeath:
    """Test that masks and info properly handle dead entities"""
    
    def test_controllable_entities_only_includes_alive(self):
        """Test that only alive entities appear in controllable_entities mask"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Verify all controllable entities are alive
        controllable_entities = infos["legacy"]["valid_masks"]["controllable_entities"]
        for entity_id in controllable_entities:
            entity = agent_legacy._sim_agent.controllable_entities[entity_id]
            assert entity.is_alive, f"Entity {entity_id} in controllable mask should be alive"
        
        env.close()
    
    def test_entity_counts_match_actual_entities(self):
        """Test that entity count fields in info dict match actual alive entities"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Count actual alive entities for legacy
        legacy_alive_count = sum(1 for e in agent_legacy._sim_agent.controllable_entities.values() if e.is_alive)
        dynasty_alive_count = sum(1 for e in agent_dynasty._sim_agent.controllable_entities.values() if e.is_alive)
        
        # Verify info dict matches
        assert infos["legacy"]["my_entities_count"] == legacy_alive_count, "Legacy entity count should match alive entities"
        assert infos["dynasty"]["my_entities_count"] == dynasty_alive_count, "Dynasty entity count should match alive entities"
        
        # total_entities refers to each agent's own entity count
        assert infos["legacy"]["total_entities"] == legacy_alive_count
        assert infos["dynasty"]["total_entities"] == dynasty_alive_count
        
        env.close()
    
    def test_refuel_sets_only_include_alive(self):
        """Test that refuel receivers/providers only include alive entities"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Verify all refuel entities are alive
        receivers = infos["legacy"]["refuel"]["receivers"]
        providers = infos["legacy"]["refuel"]["providers"]
        
        for entity_id in receivers:
            entity = agent_legacy._sim_agent.controllable_entities[entity_id]
            assert entity.is_alive, f"Refuel receiver {entity_id} should be alive"
        
        for entity_id in providers:
            entity = agent_legacy._sim_agent.controllable_entities[entity_id]
            assert entity.is_alive, f"Refuel provider {entity_id} should be alive"
        
        env.close()


class TestTerminationConditions:
    """Test termination conditions"""
    
    def test_capture_win_legacy(self):
        """Test termination based on legacy capture completion"""
        config = Config()
        config.capture_required_seconds = 20.0  # Requires 2 steps to capture
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Simulate legacy one timestep ahead of dynasty
        # Note: time_delta = frame_rate/60 = 600/60 = 10 seconds per step
        # Legacy at threshold (20), Dynasty one step behind (10)
        # Legacy completed at some previous step, Dynasty will complete this step
        # But Legacy should win because it completed FIRST
        env.capture_progress_by_faction[Faction.LEGACY] = config.capture_required_seconds  # 20.0
        env.capture_progress_by_faction[Faction.DYNASTY] = 10.0  # One step behind
        env.capture_completed_at_step[Faction.LEGACY] = env.current_step - 1  # Completed last step
        env.capture_completed_at_step[Faction.DYNASTY] = None  # Not yet completed
        
        # Step to trigger termination check
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify termination occurred
        assert terminations["legacy"] == True, "Capture completion should trigger termination"
        assert terminations["dynasty"] == True, "Both agents should terminate simultaneously"
        assert infos["legacy"]["termination_cause"] == "legacy_win"
        
        env.close()
    
    def test_capture_win_dynasty(self):
        """Test termination based on dynasty capture completion"""
        config = Config()
        config.capture_required_seconds = 20.0  # Requires 2 steps to capture
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Simulate dynasty one timestep ahead of legacy
        # Note: time_delta = frame_rate/60 = 600/60 = 10 seconds per step
        # Dynasty at threshold (20), Legacy one step behind (10)
        # Dynasty completed at some previous step, Legacy will complete this step
        # But Dynasty should win because it completed FIRST
        env.capture_progress_by_faction[Faction.DYNASTY] = config.capture_required_seconds  # 20.0
        env.capture_progress_by_faction[Faction.LEGACY] = 10.0  # One step behind
        env.capture_completed_at_step[Faction.DYNASTY] = env.current_step - 1  # Completed last step
        env.capture_completed_at_step[Faction.LEGACY] = None  # Not yet completed
        
        # Step to trigger termination check
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify termination occurred
        assert terminations["dynasty"] == True, "Capture completion should trigger termination"
        assert terminations["legacy"] == True, "Both agents should terminate simultaneously"
        assert infos["dynasty"]["termination_cause"] == "dynasty_win"
        
        env.close()
    
    def test_time_limit_truncation(self):
        """Test truncation based on time limit"""
        config = Config()
        config.max_game_time = 10.0
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Fast-forward time
        env.time_elapsed = config.max_game_time + 1.0
        
        # Step to trigger truncation check
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify truncation occurred
        assert truncations["legacy"] == True, "Time limit should trigger truncation"
        assert truncations["dynasty"] == True, "Both agents should truncate simultaneously"
        assert infos["legacy"]["termination_cause"] == "time_limit"
        
        env.close()
    
    def test_kill_ratio_win_legacy(self):
        """Test termination based on legacy achieving favorable kill ratio"""
        config = Config()
        config.kill_ratio_win_threshold = 2.0
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Simulate legacy achieving high kill ratio by manually manipulating kill sets
        # Legacy kills 10 dynasty entities, dynasty kills 2 legacy entities
        # Kill ratio for legacy = 10/2 = 5.0 > 2.0 threshold
        
        # Get entities and add to kill tracking
        dynasty_entities = list(agent_dynasty._sim_agent.controllable_entities.values())[:10]
        legacy_entities = list(agent_legacy._sim_agent.controllable_entities.values())[:2]
        
        # Manually track kills using the single source of truth: dead_entities_by_faction
        # Legacy killed 10 Dynasty entities
        for entity in dynasty_entities:
            env.dead_entities_by_faction[Faction.DYNASTY].add(entity)
        # Dynasty killed 2 Legacy entities
        for entity in legacy_entities:
            env.dead_entities_by_faction[Faction.LEGACY].add(entity)
        
        # Step to trigger termination check
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify kill ratio is favorable
        kill_ratio = infos["legacy"]["mission"]["kill_ratio"]
        assert kill_ratio >= config.kill_ratio_win_threshold, f"Kill ratio {kill_ratio} should be >= {config.kill_ratio_win_threshold}"
        
        # Verify termination occurred
        assert terminations["legacy"] == True, "Kill ratio win should trigger termination"
        assert terminations["dynasty"] == True, "Both agents should terminate simultaneously"
        assert infos["legacy"]["termination_cause"] == "legacy_win"
        
        env.close()
    
    def test_capture_progress_resets_when_settlers_lost(self):
        """Test that capture progress and completion step reset when faction loses all settlers"""
        config = Config()
        config.capture_required_seconds = 20.0
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Simulate legacy making progress toward capture
        env.capture_progress_by_faction[Faction.LEGACY] = 15.0  # 75% complete
        env.capture_completed_at_step[Faction.LEGACY] = None  # Not yet completed
        env.capture_possible_by_faction[Faction.LEGACY] = True  # Has settlers
        
        # Take a step - legacy should still have progress
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Legacy should still have progress (will be recalculated by update_capture_possible)
        # But if they actually have settlers, progress should be maintained or advanced
        
        # Now simulate legacy losing all settlers
        # Set capture_possible to False to simulate all settlers dead/gone
        env.capture_possible_by_faction[Faction.LEGACY] = False
        
        # Manually call update_capture_progress to simulate what happens in step()
        update_capture_progress(env)
        
        # Verify progress and completion step are RESET
        assert env.capture_progress_by_faction[Faction.LEGACY] == 0.0, \
            "Progress should reset to 0 when settlers are lost"
        assert env.capture_completed_at_step[Faction.LEGACY] is None, \
            "Completion step should reset to None when settlers are lost"
        
        # Simulate legacy getting settlers back and re-capturing
        env.capture_possible_by_faction[Faction.LEGACY] = True
        env.capture_progress_by_faction[Faction.LEGACY] = 18.0  # Close to completion
        
        # Advance to complete capture
        update_capture_progress(env)  # Will add time_delta (10 seconds)
        
        # Legacy should now have completed (18 + 10 = 28, capped at 20)
        assert env.capture_progress_by_faction[Faction.LEGACY] >= config.capture_required_seconds
        # And completion should be tracked at CURRENT step, not the old one
        assert env.capture_completed_at_step[Faction.LEGACY] == env.current_step, \
            "Completion step should be current step for new capture attempt"
        
        env.close()


class TestInfoDictUpdates:
    """Test that info dict is properly updated each step"""
    
    def test_time_progression(self):
        """Test that time values update correctly"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        initial_time = infos["legacy"]["time_elapsed"]
        initial_step = infos["legacy"]["step"]
        initial_time_remaining = infos["legacy"]["time_remaining"]
        
        # Take a step
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify time progression
        assert infos["legacy"]["time_elapsed"] > initial_time
        assert infos["legacy"]["step"] == initial_step + 1
        assert infos["legacy"]["time_remaining"] < initial_time_remaining
        
        # Verify both agents see same global time
        assert infos["legacy"]["time_elapsed"] == infos["dynasty"]["time_elapsed"]
        assert infos["legacy"]["time_remaining"] == infos["dynasty"]["time_remaining"]
        
        env.close()
    
    def test_entity_counts_update(self):
        """Test that entity counts are updated"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Verify entity counts exist and are reasonable
        assert infos["legacy"]["total_entities"] >= 0
        assert infos["legacy"]["my_entities_count"] >= 0
        assert infos["legacy"]["detected_targets_count"] >= 0
        
        assert infos["dynasty"]["total_entities"] >= 0
        assert infos["dynasty"]["my_entities_count"] >= 0
        assert infos["dynasty"]["detected_targets_count"] >= 0
        
        # Take steps and verify counts remain valid
        for _ in range(3):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            assert infos["legacy"]["my_entities_count"] >= 0
            assert infos["dynasty"]["my_entities_count"] >= 0
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()
    
    def test_capture_progress_updates(self):
        """Test that capture progress is tracked per faction"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Verify capture fields exist for both agents
        assert "my_capture_progress" in infos["legacy"]["mission"]
        assert "my_capture_possible" in infos["legacy"]["mission"]
        assert "enemy_capture_progress" in infos["legacy"]["mission"]
        assert "enemy_capture_possible" in infos["legacy"]["mission"]
        
        assert "my_capture_progress" in infos["dynasty"]["mission"]
        assert "my_capture_possible" in infos["dynasty"]["mission"]
        assert "enemy_capture_progress" in infos["dynasty"]["mission"]
        assert "enemy_capture_possible" in infos["dynasty"]["mission"]
        
        # Verify symmetry: legacy's enemy = dynasty's my
        assert infos["legacy"]["mission"]["enemy_capture_progress"] == infos["dynasty"]["mission"]["my_capture_progress"]
        assert infos["legacy"]["mission"]["enemy_capture_possible"] == infos["dynasty"]["mission"]["my_capture_possible"]
        
        env.close()
    
    def test_masks_update_each_step(self):
        """Test that action masks update properly"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        for _ in range(5):
            # Verify mask structure
            assert "valid_masks" in infos["legacy"]
            assert "action_types" in infos["legacy"]["valid_masks"]
            assert "controllable_entities" in infos["legacy"]["valid_masks"]
            assert "visible_targets" in infos["legacy"]["valid_masks"]
            assert "entity_target_matrix" in infos["legacy"]["valid_masks"]
            
            # No-op should always be available
            assert 0 in infos["legacy"]["valid_masks"]["action_types"]
            assert 0 in infos["dynasty"]["valid_masks"]["action_types"]
            
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()


class TestRewardCalculation:
    """Test reward calculation through agent.calculate_reward()"""
    
    def test_default_reward_is_zero(self):
        """Test that default CompetitionAgent reward is 0.0"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Take a step
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Default reward should be 0.0
        assert isinstance(rewards["legacy"], (int, float))
        assert isinstance(rewards["dynasty"], (int, float))
        assert rewards["legacy"] == 0.0
        assert rewards["dynasty"] == 0.0
        
        env.close()
    
    def test_custom_reward_agent(self):
        """Test that agents can implement custom rewards"""
        config = Config()
        
        class CustomRewardAgent(CompetitionAgent):
            def calculate_reward(self, env):
                return -0.1 * env.time_elapsed
        
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CustomRewardAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Take a step
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Legacy should have custom reward, dynasty should have default 0.0
        assert rewards["legacy"] < 0, "Custom reward should be negative"
        assert rewards["dynasty"] == 0.0, "SimpleAgent uses default reward"
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
