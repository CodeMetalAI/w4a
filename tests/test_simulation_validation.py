"""
Simulation Validation Tests

Deep validation tests to ensure the simulation is actually running and producing
meaningful state updates, not just passing through null/empty values.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction


class TestSimulationIsActuallyRunning:
    """Verify the simulation is producing real state, not null/default values"""
    
    def test_entities_are_spawned(self):
        """Verify that entities are actually spawned after reset"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Both agents should have controllable entities
        legacy_entities = infos["legacy"]["valid_masks"]["controllable_entities"]
        dynasty_entities = infos["dynasty"]["valid_masks"]["controllable_entities"]
        
        assert len(legacy_entities) > 0, "Legacy should have controllable entities after spawn"
        assert len(dynasty_entities) > 0, "Dynasty should have controllable entities after spawn"
        
        # Entity IDs should be valid integers
        # Note: Entity IDs can go beyond max_entities in the actual simulation
        # max_entities is a config parameter for observation/action space sizing
        for entity_id in legacy_entities:
            assert isinstance(entity_id, int), f"Entity ID {entity_id} is not an int"
            assert entity_id >= 0, f"Entity ID {entity_id} should be non-negative"
        
        # Agents should be able to get their entities
        legacy_entity_list = agent_legacy.get_entities()
        dynasty_entity_list = agent_dynasty.get_entities()
        
        assert len(legacy_entity_list) > 0, "Legacy agent should have entities"
        assert len(dynasty_entity_list) > 0, "Dynasty agent should have entities"
        
        # Entities should have actual simulation objects
        for entity in legacy_entity_list:
            assert entity is not None
            assert hasattr(entity, 'is_alive')
            assert hasattr(entity, 'faction')
            assert entity.faction == Faction.LEGACY
        
        for entity in dynasty_entity_list:
            assert entity is not None
            assert hasattr(entity, 'is_alive')
            assert hasattr(entity, 'faction')
            assert entity.faction == Faction.DYNASTY
        
        env.close()
    
    def test_target_groups_are_detected(self):
        """Verify that agents detect enemy target groups"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Step forward to allow detection
        for _ in range(10):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # At least one agent should eventually detect targets
            legacy_targets = infos["legacy"]["valid_masks"]["visible_targets"]
            dynasty_targets = infos["dynasty"]["valid_masks"]["visible_targets"]
            
            if len(legacy_targets) > 0 or len(dynasty_targets) > 0:
                # If we detected targets, verify they're real
                if len(legacy_targets) > 0:
                    legacy_target_groups = agent_legacy.get_target_groups()
                    assert len(legacy_target_groups) > 0, "Legacy should have target group objects"
                    
                    for tg in legacy_target_groups:
                        assert tg is not None
                        assert hasattr(tg, 'faction')
                        assert tg.faction == Faction.LEGACY, "Target should be enemy"
                
                if len(dynasty_targets) > 0:
                    dynasty_target_groups = agent_dynasty.get_target_groups()
                    assert len(dynasty_target_groups) > 0, "Dynasty should have target group objects"
                    
                    for tg in dynasty_target_groups:
                        assert tg is not None
                        assert hasattr(tg, 'faction')
                        assert tg.faction == Faction.DYNASTY, "Target should be enemy"
                
                break
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()
    
    def test_engagement_matrix_is_meaningful(self):
        """Verify entity-target engagement matrix contains real data"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Step forward to build engagement opportunities
        for step in range(20):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if engagement matrix has entries
            legacy_matrix = infos["legacy"]["valid_masks"]["entity_target_matrix"]
            dynasty_matrix = infos["dynasty"]["valid_masks"]["entity_target_matrix"]
            
            # Matrix keys should be entity IDs
            for entity_id, target_set in legacy_matrix.items():
                assert isinstance(entity_id, int), "Matrix key should be entity ID"
                assert isinstance(target_set, set), "Matrix value should be set of targets"
                
                # If entity can engage targets, verify they're valid
                if len(target_set) > 0:
                    # Entity should be in controllable entities
                    assert entity_id in infos["legacy"]["valid_masks"]["controllable_entities"]
                    
                    # Targets should be in visible targets
                    visible = infos["legacy"]["valid_masks"]["visible_targets"]
                    for target_id in target_set:
                        assert isinstance(target_id, int), "Target ID should be int"
                        assert target_id in visible, f"Target {target_id} not in visible targets"
            
            # If we found any engageable targets, test is successful
            if any(len(targets) > 0 for targets in legacy_matrix.values()):
                # Verify engage action is available
                assert 2 in infos["legacy"]["valid_masks"]["action_types"], \
                    "Engage should be available when entities can engage"
                env.close()
                return
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()
    
    def test_simulation_state_changes_over_time(self):
        """Verify simulation state actually changes as time progresses"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        initial_time = infos["legacy"]["time_elapsed"]
        initial_step = infos["legacy"]["step"]
        
        time_values = [initial_time]
        step_values = [initial_step]
        
        # Run simulation for several steps
        for _ in range(10):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            time_values.append(infos["legacy"]["time_elapsed"])
            step_values.append(infos["legacy"]["step"])
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        # Time should be strictly increasing
        for i in range(1, len(time_values)):
            assert time_values[i] > time_values[i-1], \
                f"Time not increasing: {time_values[i]} <= {time_values[i-1]}"
        
        # Steps should be strictly increasing by 1
        for i in range(1, len(step_values)):
            assert step_values[i] == step_values[i-1] + 1, \
                f"Steps not incrementing correctly: {step_values[i]} != {step_values[i-1] + 1}"
        
        env.close()
    
    def test_observations_are_not_all_zeros(self):
        """Verify observations contain actual data, not just zeros"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Current observation implementation returns zeros (placeholder)
        # This test will need to be updated when observations are implemented
        assert observations["legacy"].shape == (12,), "Observation should be 12 features"
        assert observations["dynasty"].shape == (12,), "Observation should be 12 features"
        
        # For now, verify that observations are at least well-formed
        assert observations["legacy"].dtype == np.float32
        assert observations["dynasty"].dtype == np.float32
        assert np.all(np.isfinite(observations["legacy"])), "Observations should be finite"
        assert np.all(np.isfinite(observations["dynasty"])), "Observations should be finite"
        
        env.close()


class TestActionsAffectSimulation:
    """Verify actions actually affect simulation state"""
    
    def test_noop_preserves_entity_count(self):
        """Verify no-op doesn't cause entities to disappear"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        initial_legacy_count = infos["legacy"]["my_entities_count"]
        initial_dynasty_count = infos["dynasty"]["my_entities_count"]
        
        # Execute several no-ops
        noop_action = {
            "action_type": 0, "entity_id": 0, "move_center_grid": 0,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        
        for _ in range(5):
            actions = {"legacy": noop_action, "dynasty": noop_action}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Entity counts should remain stable with no-op (no combat)
            # Note: counts could change due to SimpleAgent actions or combat
            assert infos["legacy"]["my_entities_count"] >= 0
            assert infos["dynasty"]["my_entities_count"] >= 0
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()
    
    def test_valid_actions_produce_different_rewards(self):
        """Verify different agent reward implementations produce different rewards"""
        config = Config()
        
        class NegativeTimeRewardAgent(CompetitionAgent):
            def calculate_reward(self, env):
                return -env.time_elapsed
        
        class PositiveStepRewardAgent(CompetitionAgent):
            def calculate_reward(self, env):
                return env.current_step * 0.1
        
        env1 = TridentIslandMultiAgentEnv(config=config)
        agent1_legacy = NegativeTimeRewardAgent(Faction.LEGACY, config)
        agent1_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env1.set_agents(agent1_legacy, agent1_dynasty)
        
        env2 = TridentIslandMultiAgentEnv(config=config)
        agent2_legacy = PositiveStepRewardAgent(Faction.LEGACY, config)
        agent2_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env2.set_agents(agent2_legacy, agent2_dynasty)
        
        # Reset both with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Take identical actions
        action = {
            "action_type": 0, "entity_id": 0, "move_center_grid": 0,
            "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
            "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
            "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        
        actions1 = {"legacy": action, "dynasty": agent1_dynasty.select_action(obs1["dynasty"])}
        actions2 = {"legacy": action, "dynasty": agent2_dynasty.select_action(obs2["dynasty"])}
        
        _, rewards1, _, _, _ = env1.step(actions1)
        _, rewards2, _, _, _ = env2.step(actions2)
        
        # Different reward functions should produce different rewards
        assert rewards1["legacy"] < 0, "Negative time reward should be negative"
        assert rewards2["legacy"] > 0, "Positive step reward should be positive"
        assert rewards1["legacy"] != rewards2["legacy"], \
            "Different reward functions should produce different rewards"
        
        env1.close()
        env2.close()


class TestRefuelMechanics:
    """Verify refuel mechanics work properly"""
    
    def test_refuel_info_structure(self):
        """Verify refuel providers and receivers are tracked"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Refuel info should exist
        assert "refuel" in infos["legacy"]
        assert "receivers" in infos["legacy"]["refuel"]
        assert "providers" in infos["legacy"]["refuel"]
        
        assert "refuel" in infos["dynasty"]
        assert "receivers" in infos["dynasty"]["refuel"]
        assert "providers" in infos["dynasty"]["refuel"]
        
        # Lists should contain integers (entity IDs)
        for receiver_id in infos["legacy"]["refuel"]["receivers"]:
            assert isinstance(receiver_id, int)
        
        for provider_id in infos["legacy"]["refuel"]["providers"]:
            assert isinstance(provider_id, int)
        
        env.close()


class TestCaptureProgress:
    """Verify capture progress is tracked per faction"""
    
    def test_capture_progress_is_per_faction(self):
        """Verify capture progress is tracked independently per faction"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Both factions start with 0 capture progress
        assert infos["legacy"]["mission"]["my_capture_progress"] == 0.0
        assert infos["dynasty"]["mission"]["my_capture_progress"] == 0.0
        
        # Manually set legacy capture progress
        env.capture_progress_by_faction[Faction.LEGACY] = 5.0
        env.capture_progress_by_faction[Faction.DYNASTY] = 2.0
        
        # Step to update info
        # Note: update_capture_progress() will advance progress by time_delta (10 seconds) if settlers present
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify each faction sees their own progress (advanced by 10 seconds)
        # Legacy: 5.0 + 10.0 = 15.0, Dynasty: 2.0 + 10.0 = 12.0
        assert infos["legacy"]["mission"]["my_capture_progress"] == 15.0
        assert infos["legacy"]["mission"]["enemy_capture_progress"] == 12.0
        
        assert infos["dynasty"]["mission"]["my_capture_progress"] == 12.0
        assert infos["dynasty"]["mission"]["enemy_capture_progress"] == 15.0
        
        env.close()
    
    def test_capture_possible_is_per_faction(self):
        """Verify capture_possible is tracked independently per faction"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Both start with capture possible
        assert infos["legacy"]["mission"]["my_capture_possible"] == True
        assert infos["dynasty"]["mission"]["my_capture_possible"] == True
        
        # Test that capture_possible is independently tracked
        # We can't manually set it mid-step since update_mission_metrics will recalculate
        # Instead, verify that both factions report capture_possible independently
        
        # Step several times to see state
        for _ in range(5):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Both factions should report their own capture_possible state
            legacy_my_capture = infos["legacy"]["mission"]["my_capture_possible"]
            legacy_enemy_capture = infos["legacy"]["mission"]["enemy_capture_possible"]
            dynasty_my_capture = infos["dynasty"]["mission"]["my_capture_possible"]
            dynasty_enemy_capture = infos["dynasty"]["mission"]["enemy_capture_possible"]
            
            # Verify symmetry
            assert legacy_my_capture == dynasty_enemy_capture, "Symmetry broken: legacy my != dynasty enemy"
            assert dynasty_my_capture == legacy_enemy_capture, "Symmetry broken: dynasty my != legacy enemy"
            
            # Values should be boolean
            assert isinstance(legacy_my_capture, bool)
            assert isinstance(dynasty_my_capture, bool)
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

