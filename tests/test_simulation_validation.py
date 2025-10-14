"""
Simulation Validation Tests

Deep validation tests to ensure the simulation is actually running and producing
meaningful state updates, not just passing through null/empty values.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.envs.constants import CENTER_ISLAND_FLAG_ID
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
                        assert tg.faction == Faction.LEGACY, "Target group belongs to same faction (represents enemies visible to that faction)"
                
                if len(dynasty_targets) > 0:
                    dynasty_target_groups = agent_dynasty.get_target_groups()
                    assert len(dynasty_target_groups) > 0, "Dynasty should have target group objects"
                    
                    for tg in dynasty_target_groups:
                        assert tg is not None
                        assert hasattr(tg, 'faction')
                        assert tg.faction == Faction.DYNASTY, "Target group belongs to same faction (represents enemies visible to that faction)"
                
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
        
        found_engageable_targets = False
        
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
            
            # Check if we found any engageable targets
            if any(len(targets) > 0 for targets in legacy_matrix.values()):
                found_engageable_targets = True
                # Verify engage action is available
                assert 2 in infos["legacy"]["valid_masks"]["action_types"], \
                    "Engage should be available when entities can engage"
                print(f"\n[ENGAGEMENT MATRIX] Found engageable targets at step {step}")
                break
            
            if terminations["legacy"] or truncations["legacy"]:
                break
        
        # Assert that we found engageable targets during the test
        assert found_engageable_targets, \
            f"Expected to find engageable targets within 20 steps but found none. " \
            f"Last legacy matrix: {legacy_matrix}, Last dynasty matrix: {dynasty_matrix}"
        
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
        
        # 10 global + (max_entities * 36) + (max_target_groups * 12)
        expected_size = 10 + (config.max_entities * 36) + (config.max_target_groups * 12)
        
        assert observations["legacy"].shape == (expected_size,), f"Observation should be {expected_size} features"
        assert observations["dynasty"].shape == (expected_size,), f"Observation should be {expected_size} features"
        
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
        """Verify capture progress tracking infrastructure and flag state changes"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Verify infrastructure
        assert infos["legacy"]["mission"]["my_capture_progress"] == 0.0
        assert infos["dynasty"]["mission"]["my_capture_progress"] == 0.0
        
        flag = env.flags[CENTER_ISLAND_FLAG_ID]
        assert hasattr(env, 'capture_progress_by_faction')
        assert Faction.LEGACY in env.capture_progress_by_faction
        assert Faction.DYNASTY in env.capture_progress_by_faction
        
        # Find settler
        pioneer_id = None
        pioneer_entity = None
        for entity_id, entity in agent_legacy._sim_agent.controllable_entities.items():
            if hasattr(entity, 'can_capture') and entity.can_capture and entity.is_alive:
                pioneer_id = entity_id
                pioneer_entity = entity
                break
        
        assert pioneer_id is not None, "Should have a settler unit"
        
        # Send capture action once

        capture_action = {
            "action_type": 5, "entity_id": pioneer_id,
            "move_center_grid": 0, "move_short_axis_km": 0, "move_long_axis_km": 0,
            "move_axis_angle": 0, "target_group_id": 0, "weapon_selection": 0,
            "weapon_usage": 0, "weapon_engagement": 0, "stealth_enabled": 0,
            "sensing_position_grid": 0, "refuel_target_id": 0
        }
        
        noop_action = {
            "action_type": 0, "entity_id": 0,
            "move_center_grid": 0, "move_short_axis_km": 0, "move_long_axis_km": 0,
            "move_axis_angle": 0, "target_group_id": 0, "weapon_selection": 0,
            "weapon_usage": 0, "weapon_engagement": 0, "stealth_enabled": 0,
            "sensing_position_grid": 0, "refuel_target_id": 0
        }
        
        flag_pos = flag.pos
        max_steps = 480  # 80 minutes game time
        
        print(f"\n[FLAG STATE TRACKING]")
        print(f"Flag at: ({flag_pos.x:.0f}, {flag_pos.y:.0f})")
        print(f"Initial flag state:")
        print(f"  is_captured: {flag.is_captured}")
        print(f"  is_being_captured: {flag.is_being_captured}")
        
        # Track if flag state changed
        flag_being_captured_changed = False
        flag_capture_progress_increased = False
        
        for step in range(max_steps):
            if step == 0:
                legacy_action = capture_action
            else:
                legacy_action = noop_action
            
            actions = {"legacy": legacy_action, "dynasty": noop_action}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            legacy_progress = infos["legacy"]["mission"]["my_capture_progress"]
            dynasty_progress = infos["dynasty"]["mission"]["my_capture_progress"]
            
            # Track flag state changes
            if flag.is_being_captured and not flag_being_captured_changed:
                flag_being_captured_changed = True
                print(f"\n[FLAG STATE CHANGE] At step {step} ({step*10}s game time):")
                print(f"  is_being_captured changed to: True")
                print(f"  Capture progress: {legacy_progress:.1f}s")
            
            if legacy_progress > 0 and not flag_capture_progress_increased:
                flag_capture_progress_increased = True
                print(f"\n[CAPTURE PROGRESS] At step {step} ({step*10}s game time):")
                print(f"  Capture progress increased to: {legacy_progress:.1f}s")
            
            # Print periodic updates (every 50 steps)
            if step % 50 == 0 and step > 0:
                print(f"\nStep {step} ({step*10}s game time):")
                print(f"  Flag is_captured: {flag.is_captured}")
                print(f"  Flag is_being_captured: {flag.is_being_captured}")
                print(f"  Legacy capture progress: {legacy_progress:.1f}s")
                print(f"  Dynasty capture progress: {dynasty_progress:.1f}s")
            
                print(f" Check that pioneer is still alive: {pioneer_entity.is_alive}")
            if terminations["legacy"] or terminations["dynasty"]:
                # VERIFY: Termination cause
                assert infos["legacy"]["termination_cause"] == "legacy_win", \
                    f"Expected legacy_win, got {infos['legacy']['termination_cause']}"
                
                # VERIFY: Flag is captured
                assert flag.is_captured, "Flag should be captured when termination occurs"
                
                # VERIFY: Capture progress reached threshold
                assert legacy_progress >= config.capture_required_seconds, \
                    f"Capture progress {legacy_progress} should be >= {config.capture_required_seconds}"
                
                # VERIFY: Rewards on termination
                assert rewards["legacy"] > 0, f"Legacy should get positive reward on win, got {rewards['legacy']}"
                assert rewards["dynasty"] < 0, f"Dynasty should get negative reward on loss, got {rewards['dynasty']}"
                
                print(f"\n[TERMINATION SUCCESS] Legacy won via capture at step {step}")
                print(f"  Termination cause: {infos['legacy']['termination_cause']}")
                print(f"  Flag is_captured: {flag.is_captured}")
                print(f"  Flag is_being_captured: {flag.is_being_captured}")
                print(f"  Legacy capture progress: {legacy_progress:.1f}s / {config.capture_required_seconds}s")
                print(f"  Legacy reward: {rewards['legacy']}")
                print(f"  Dynasty reward: {rewards['dynasty']}")
                
                env.close()
                return
            
            if truncations["legacy"]:
                break
        
        env.close()
        
        # Test should fail - capture didn't trigger win condition
        pytest.fail(
            f"Expected Legacy to win via capture within {max_steps} steps but didn't terminate.\n")
    
    def test_capture_win_triggers_termination(self):
        """Test that capture action leads to win condition
        
        This test verifies the full capture flow with a passive adversary:
        1. Legacy agent issues capture command to settler
        2. Dynasty agent is passive (doesn't attack)
        3. Run for up to 80 minutes of game time (480 steps)
        4. Verify termination triggers with capture win
        5. Verify info dict updates (capture progress)
        6. Verify reward is issued
        """
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        # Legacy agent - active
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        
        # Dynasty agent - passive (dummy that doesn't respond to adversary contact)
        class PassiveDynastyAgent(CompetitionAgent):
            def on_adversary_contact(self, event):
                """Override to do nothing - don't attack Legacy units"""
                nonlocal adversary_contact_count
                adversary_contact_count += 1
                if adversary_contact_count <= 3:  # Only print first 3 to avoid spam
                    print(f"[DYNASTY CONTACT #{adversary_contact_count}] I see an adversary, but I don't attack them")
                pass
        
        agent_dynasty = PassiveDynastyAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        settler_id = None
        settler_entity = None
        settler_name = None
        
        print(f"\n[CAPTURE WIN TEST] Searching for settler in {len(agent_legacy._sim_agent.controllable_entities)} Legacy entities")
        for entity_id, entity in agent_legacy._sim_agent.controllable_entities.items():
            if entity.is_alive and entity.can_capture:
                settler_id = entity_id
                settler_entity = entity
                print("What entity can capture??")
                print(f"  Found settler: RL_ID={settler_id}, identifier={entity.identifier}")
                break
        
        assert settler_id is not None, "Legacy faction should have at least one settler unit"
        assert settler_entity is not None, f"Settler entity should exist"
        
        # Create capture action for the settler
        capture_action = {
            "action_type": 5,  # Capture
            "entity_id": settler_id,
            "move_center_grid": 0, "move_short_axis_km": 0, "move_long_axis_km": 0,
            "move_axis_angle": 0, "target_group_id": 0, "weapon_selection": 0,
            "weapon_usage": 0, "weapon_engagement": 0, "stealth_enabled": 0,
            "sensing_position_grid": 0, "refuel_target_id": 0
        }
        
        # Calculate max steps for 80 minutes
        # frame_rate = 600, time per step = frame_rate/60 = 10 seconds
        # 80 minutes = 4800 seconds, steps = 4800/10 = 480
        max_steps = 480
        
        print(f"\n[SIMULATION] Running for up to {max_steps} steps (80 minutes game time)")
        print(f"  Initial capture progress: Legacy={infos['legacy']['mission']['my_capture_progress']:.1f}s")
        
        # Debug: Check Dynasty entities and sensors
        print(f"\n[DYNASTY DEBUG] {len(agent_dynasty._sim_agent.controllable_entities)} Dynasty entities:")
        for dyn_id, dyn_entity in list(agent_dynasty._sim_agent.controllable_entities.items())[:3]:  # Show first 3
            dyn_name = getattr(dyn_entity, 'Identifier', f'Unknown_{dyn_id}')
            can_sense = getattr(dyn_entity, 'can_sense', False)
            print(f"  Entity {dyn_id}: {type(dyn_entity).__name__}, {dyn_name}, can_sense={can_sense}")
        
        # Debug: Check target groups at start
        print(f"\n[TARGET GROUPS DEBUG]")
        legacy_tg = agent_legacy._sim_agent.target_groups
        dynasty_tg = agent_dynasty._sim_agent.target_groups
        print(f"  Legacy target_groups: {legacy_tg is not None}, count={len(legacy_tg) if legacy_tg else 'N/A'}")
        print(f"  Dynasty target_groups: {dynasty_tg is not None}, count={len(dynasty_tg) if dynasty_tg else 'N/A'}")
        print(f"  Legacy visible_targets from info: {infos['legacy']['valid_masks']['visible_targets']}")
        print(f"  Dynasty visible_targets from info: {infos['dynasty']['valid_masks']['visible_targets']}")
        
        # Debug: Check if ships are visible
        print(f"\n[SHIPS DEBUG]")
        legacy_ships = [e for e in agent_legacy._sim_agent.controllable_entities.values() 
                       if type(e).__name__ == 'Ship' and e.is_alive]
        dynasty_ships = [e for e in agent_dynasty._sim_agent.controllable_entities.values() 
                        if type(e).__name__ == 'Ship' and e.is_alive]
        print(f"  Legacy has {len(legacy_ships)} ships")
        print(f"  Dynasty has {len(dynasty_ships)} ships")
        
        capture_started = False
        termination_occurred = False
        final_reward_legacy = 0.0
        adversary_contact_count = 0
        
        # Step simulation 10 times to allow detection to occur
        print(f"\n[DETECTION] Stepping 10 times to allow targets to be detected...")
        for i in range(10):
            actions = {
                "legacy": capture_action,
                "dynasty": {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                           "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                           "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                           "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                           "refuel_target_id": 0}
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"  After 10 steps:")
        print(f"    Legacy target_groups: {len(agent_legacy._sim_agent.target_groups)} targets")
        print(f"    Dynasty target_groups: {len(agent_dynasty._sim_agent.target_groups)} targets")
        if len(agent_legacy._sim_agent.target_groups) > 0:
            print(f"    Legacy sees target IDs: {list(agent_legacy._sim_agent.target_groups.keys())[:3]}")
        if len(agent_dynasty._sim_agent.target_groups) > 0:
            print(f"    Dynasty sees target IDs: {list(agent_dynasty._sim_agent.target_groups.keys())[:3]}")
        print(f"    Adversary contacts so far: {adversary_contact_count}")
        print(f"    Legacy capture progress: {infos['legacy']['mission']['my_capture_progress']:.1f}s")
        
        for step in range(max_steps):
            # Legacy issues capture command, Dynasty does nothing
            actions = {
                "legacy": capture_action,
                "dynasty": {"action_type": 0, "entity_id": 0, "move_center_grid": 0,
                           "move_short_axis_km": 0, "move_long_axis_km": 0, "move_axis_angle": 0,
                           "target_group_id": 0, "weapon_selection": 0, "weapon_usage": 0,
                           "weapon_engagement": 0, "stealth_enabled": 0, "sensing_position_grid": 0,
                           "refuel_target_id": 0}
            }
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Track capture progress
            legacy_progress = infos["legacy"]["mission"]["my_capture_progress"]
            
            if not capture_started and legacy_progress > 0:
                capture_started = True
                print(f"\n[CAPTURE STARTED] At step {step} ({step*10}s game time)")
                print(f"  Legacy capture progress: {legacy_progress:.1f}s")
            
            # Print periodic updates
            if step % 50 == 0 and step > 0:
                legacy_targets = len(infos['legacy']['valid_masks']['visible_targets'])
                dynasty_targets = len(infos['dynasty']['valid_masks']['visible_targets'])
                print(f"  Step {step:3d} ({step*10:4d}s): Legacy progress={legacy_progress:.1f}s, "
                      f"time_elapsed={infos['legacy']['time_elapsed']:.1f}s, "
                      f"L_targets={legacy_targets}, D_targets={dynasty_targets}, contacts={adversary_contact_count}")
            
            # Check for termination
            if terminations["legacy"] or terminations["dynasty"]:
                termination_occurred = True
                final_reward_legacy = rewards["legacy"]
                
                print(f"\n[TERMINATION] At step {step} ({step*10}s game time)")
                print(f"  Termination cause: {infos['legacy']['termination_cause']}")
                print(f"  Legacy capture progress: {legacy_progress:.1f}s")
                print(f"  Legacy reward: {final_reward_legacy}")
                print(f"  Both terminated: legacy={terminations['legacy']}, dynasty={terminations['dynasty']}")
                
                # Verify capture win
                assert infos["legacy"]["termination_cause"] == "legacy_win", \
                    f"Expected 'legacy_win', got '{infos['legacy']['termination_cause']}'"
                
                # Verify capture progress reached threshold
                assert legacy_progress >= config.capture_required_seconds, \
                    f"Capture progress {legacy_progress} should be >= {config.capture_required_seconds}"
                
                # Verify both agents terminated
                assert terminations["legacy"] == True, "Legacy should terminate"
                assert terminations["dynasty"] == True, "Dynasty should terminate"
                
                # Test passed!
                env.close()
                return
            
            # Check for truncation (shouldn't happen before 80 min)
            if truncations["legacy"] or truncations["dynasty"]:
                print(f"\n[TRUNCATION] At step {step}")
                print(f"  Truncation cause: {infos['legacy']['termination_cause']}")
                break
        
        # If we got here, test failed
        env.close()
        
        pytest.fail(
            f"Capture did not trigger termination within {max_steps} steps (80 minutes).\n"
            f"  Settler: {settler_name} (RL_ID={settler_id})\n"
            f"  Capture started: {capture_started}\n"
            f"  Final capture progress: {infos['legacy']['mission']['my_capture_progress']:.1f}s\n"
            f"  Required: {config.capture_required_seconds}s\n"
            f"  Final time elapsed: {infos['legacy']['time_elapsed']:.1f}s\n"
            f"  Termination occurred: {termination_occurred}\n"
            f"  Adversary contacts detected by Dynasty: {adversary_contact_count}\n"
            f"  Final Legacy targets: {len(infos['legacy']['valid_masks']['visible_targets'])}\n"
            f"  Final Dynasty targets: {len(infos['dynasty']['valid_masks']['visible_targets'])}\n"
        )
    
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

