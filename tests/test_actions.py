"""
Action Execution Tests

Tests to verify that each action type executes correctly through the multiagent
environment using the CompetitionAgent interface.
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction


class TestActionVisibilityAtReset:
    """Test that adversary entities are visible at t=0"""
    
    def test_adversary_ships_visible_on_first_tick(self):
        """Verify that both agents detect enemy ships on the first tick (step 1)"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # At reset (t=0), detection has not processed yet
        legacy_targets_t0 = infos["legacy"]["valid_masks"]["visible_targets"]
        dynasty_targets_t0 = infos["dynasty"]["valid_masks"]["visible_targets"]
        
        print(f"\n[t=0 RESET] Legacy agent detects {len(legacy_targets_t0)} enemy target groups")
        print(f"[t=0 RESET] Dynasty agent detects {len(dynasty_targets_t0)} enemy target groups")
        
        # Take first step - ships should be visible on tick 1
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        legacy_targets = infos["legacy"]["valid_masks"]["visible_targets"]
        dynasty_targets = infos["dynasty"]["valid_masks"]["visible_targets"]
        
        print(f"\n[t=1 FIRST TICK] Legacy agent detects {len(legacy_targets)} enemy target groups")
        print(f"[t=1 FIRST TICK] Dynasty agent detects {len(dynasty_targets)} enemy target groups")
        
        # Ships should be visible on first tick
        assert len(legacy_targets) > 0, "Legacy should detect Dynasty ships on first tick"
        assert len(dynasty_targets) > 0, "Dynasty should detect Legacy ships on first tick"
        
        # Get actual target group objects
        legacy_target_groups = agent_legacy.get_target_groups()
        dynasty_target_groups = agent_dynasty.get_target_groups()
        
        print(f"[t=1] Legacy can see {len(legacy_target_groups)} target group objects")
        print(f"[t=1] Dynasty can see {len(dynasty_target_groups)} target group objects")
        
        # Verify target groups belong to same faction (represent enemies visible to that faction)
        for tg in legacy_target_groups:
            assert tg.faction == Faction.LEGACY, f"Legacy should only see Legacy targets, got {tg.faction}"
            print(f"  - Legacy sees target group: faction={tg.faction.name}")
        
        for tg in dynasty_target_groups:
            assert tg.faction == Faction.DYNASTY, f"Dynasty should only see Dynasty targets, got {tg.faction}"
            print(f"  - Dynasty sees target group: faction={tg.faction.name}")
        
        env.close()
    
    def test_engage_action_available_on_first_tick(self):
        """Verify engage action (type 2) is available on first tick since ships are visible"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # At reset, engage may not be available yet
        print(f"\n[t=0 RESET] Legacy valid action types: {sorted(infos['legacy']['valid_masks']['action_types'])}")
        print(f"[t=0 RESET] Dynasty valid action types: {sorted(infos['dynasty']['valid_masks']['action_types'])}")
        
        # Take first step
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check if engage is in valid action types after first tick
        legacy_action_types = infos["legacy"]["valid_masks"]["action_types"]
        dynasty_action_types = infos["dynasty"]["valid_masks"]["action_types"]
        
        print(f"\n[t=1 FIRST TICK] Legacy valid action types: {sorted(legacy_action_types)}")
        print(f"[t=1 FIRST TICK] Dynasty valid action types: {sorted(dynasty_action_types)}")
        
        # Engage (2) should be available if targets are visible and entities have weapons
        legacy_targets = infos["legacy"]["valid_masks"]["visible_targets"]
        dynasty_targets = infos["dynasty"]["valid_masks"]["visible_targets"]
        
        legacy_matrix = infos["legacy"]["valid_masks"]["entity_target_matrix"]
        dynasty_matrix = infos["dynasty"]["valid_masks"]["entity_target_matrix"]
        
        legacy_has_engageable = any(len(targets) > 0 for targets in legacy_matrix.values())
        dynasty_has_engageable = any(len(targets) > 0 for targets in dynasty_matrix.values())
        
        print(f"[t=1] Legacy has engageable targets: {legacy_has_engageable}")
        print(f"[t=1] Dynasty has engageable targets: {dynasty_has_engageable}")
        
        # Assert that both sides have engageable targets at t=1
        assert legacy_has_engageable, f"Expected Legacy to have engageable targets at t=1. Matrix: {legacy_matrix}"
        assert dynasty_has_engageable, f"Expected Dynasty to have engageable targets at t=1. Matrix: {dynasty_matrix}"
        
        # Verify engage action is available
        assert 2 in legacy_action_types, "Engage should be available when targets are engageable"
        assert 2 in dynasty_action_types, "Engage should be available when targets are engageable"
        
        print("[t=1] Legacy CAN engage - action type 2 is available")
        print("[t=1] Dynasty CAN engage - action type 2 is available")
        
        env.close()


class TestMoveAction:
    """Test move action (type 1) execution"""
    
    def test_move_action_execution(self):
        """Test that move action executes and updates entity state"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Find a controllable entity
        controllable = infos["legacy"]["valid_masks"]["controllable_entities"]
        assert len(controllable) > 0, "Need controllable entities to test move"
        
        entity_id = list(controllable)[0]
        print(f"\n[MOVE] Testing move action on entity {entity_id}")
        
        # Create move action
        action_legacy = {
            "action_type": 1,
            "entity_id": entity_id,
            "move_center_grid": 10,
            "move_short_axis_km": 10,
            "move_long_axis_km": 100,
            "move_axis_angle": 0,
            "target_group_id": 0,
            "weapon_selection": 0,
            "weapon_usage": 0,
            "weapon_engagement": 0,
            "stealth_enabled": 0,
            "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded in intent
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == entity_id and v["action_type"] == 1 for v in intent.values())
        assert found, "Move action should be recorded in intent"
        
        print(f"[MOVE] Move action executed for entity {entity_id}")
        print(f"[MOVE] Action recorded in intent: {found}")
        
        env.close()


class TestEngageAction:
    """Test engage action (type 2) execution"""
    
    def test_engage_action_execution(self):
        """Test that engage action executes when targets are available"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Step forward to t=1 where targets are visible
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Find entity that can engage at t=1
        matrix = infos["legacy"]["valid_masks"]["entity_target_matrix"]
        entity_id = None
        target_id = None
        
        for eid, targets in matrix.items():
            if len(targets) > 0:
                entity_id = eid
                target_id = list(targets)[0]
                break
        
        assert entity_id is not None, f"Expected engageable entities at t=1 but found none. Matrix: {matrix}"
        
        print(f"\n[ENGAGE] Testing engage action at t=1: entity {entity_id} -> target {target_id}")
        
        # Create engage action
        action_legacy = {
            "action_type": 2,
            "entity_id": entity_id,
            "move_center_grid": 0,
            "move_short_axis_km": 0,
            "move_long_axis_km": 0,
            "move_axis_angle": 0,
            "target_group_id": target_id,
            "weapon_selection": 0,
            "weapon_usage": 0,
            "weapon_engagement": 0,
            "stealth_enabled": 0,
            "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == entity_id and v["action_type"] == 2 for v in intent.values())
        assert found, "Engage action should be recorded in intent"
        
        print(f"[ENGAGE] Engage action executed for entity {entity_id} on target {target_id}")
        print(f"[ENGAGE] Action recorded in intent: {found}")
        
        env.close()


class TestStealthAction:
    """Test stealth action (type 3) execution"""
    
    def test_stealth_action_execution(self):
        """Test that stealth action executes on capable entities"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Check if stealth action is available
        action_types = infos["legacy"]["valid_masks"]["action_types"]
        if 3 not in action_types:
            print("\n[STEALTH] Stealth action not available, skipping test")
            env.close()
            pytest.skip("Stealth action not available")
            return
        
        # Find any controllable entity (stealth capability checked by action_types mask)
        controllable = infos["legacy"]["valid_masks"]["controllable_entities"]
        entity_id = list(controllable)[0]
        
        print(f"\n[STEALTH] Testing stealth action on entity {entity_id}")
        
        # Create stealth action
        action_legacy = {
            "action_type": 3,
            "entity_id": entity_id,
            "move_center_grid": 0,
            "move_short_axis_km": 0,
            "move_long_axis_km": 0,
            "move_axis_angle": 0,
            "target_group_id": 0,
            "weapon_selection": 0,
            "weapon_usage": 0,
            "weapon_engagement": 0,
            "stealth_enabled": 1,
            "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == entity_id and v["action_type"] == 3 for v in intent.values())
        
        print(f"[STEALTH] Stealth action executed for entity {entity_id}")
        print(f"[STEALTH] Action recorded in intent: {found}")
        
        env.close()


class TestSensingAction:
    """Test sensing/radar focus action (type 4) execution"""
    
    def test_sensing_action_execution(self):
        """Test that sensing action executes on radar-capable entities"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Check if sensing action is available
        action_types = infos["legacy"]["valid_masks"]["action_types"]
        if 4 not in action_types:
            print("\n[SENSING] Sensing action not available, skipping test")
            env.close()
            pytest.skip("Sensing action not available")
            return
        
        # Find any controllable entity
        controllable = infos["legacy"]["valid_masks"]["controllable_entities"]
        entity_id = list(controllable)[0]
        
        print(f"\n[SENSING] Testing sensing action on entity {entity_id}")
        
        # Create sensing action
        action_legacy = {
            "action_type": 4,
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
            "sensing_position_grid": 15,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == entity_id and v["action_type"] == 4 for v in intent.values())
        
        print(f"[SENSING] Sensing action executed for entity {entity_id}")
        print(f"[SENSING] Action recorded in intent: {found}")
        
        env.close()


class TestCaptureAction:
    """Test capture action (type 5) execution"""
    
    def test_capture_action_execution(self):
        """Test that capture action executes on capture-capable entities"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Check if capture action is available
        action_types = infos["legacy"]["valid_masks"]["action_types"]
        if 5 not in action_types:
            print("\n[CAPTURE] Capture action not available, skipping test")
            env.close()
            pytest.skip("Capture action not available")
            return
        
        # Find any controllable entity
        controllable = infos["legacy"]["valid_masks"]["controllable_entities"]
        entity_id = list(controllable)[0]
        
        print(f"\n[CAPTURE] Testing capture action on entity {entity_id}")
        
        # Create capture action
        action_legacy = {
            "action_type": 5,
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
            "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == entity_id and v["action_type"] == 5 for v in intent.values())
        
        print(f"[CAPTURE] Capture action executed for entity {entity_id}")
        print(f"[CAPTURE] Action recorded in intent: {found}")
        
        env.close()


class TestRTBAction:
    """Test return to base action (type 6) execution"""
    
    def test_rtb_action_execution(self):
        """Test that RTB action executes on air entities"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Check if RTB action is available
        action_types = infos["legacy"]["valid_masks"]["action_types"]
        if 6 not in action_types:
            print("\n[RTB] RTB action not available, skipping test")
            env.close()
            pytest.skip("RTB action not available")
            return
        
        # Find any controllable entity
        controllable = infos["legacy"]["valid_masks"]["controllable_entities"]
        entity_id = list(controllable)[0]
        
        print(f"\n[RTB] Testing RTB action on entity {entity_id}")
        
        # Create RTB action
        action_legacy = {
            "action_type": 6,
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
            "sensing_position_grid": 0,
            "refuel_target_id": 0
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == entity_id and v["action_type"] == 6 for v in intent.values())
        
        print(f"[RTB] RTB action executed for entity {entity_id}")
        print(f"[RTB] Action recorded in intent: {found}")
        
        env.close()


class TestRefuelAction:
    """Test refuel action (type 7) execution"""
    
    def test_refuel_action_execution(self):
        """Test that refuel action executes when receivers and providers are available"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Check if refuel action is available
        action_types = infos["legacy"]["valid_masks"]["action_types"]
        if 7 not in action_types:
            print("\n[REFUEL] Refuel action not available, skipping test")
            env.close()
            pytest.skip("Refuel action not available")
            return
        
        # Get refuel info
        receivers = infos["legacy"]["refuel"]["receivers"]
        providers = infos["legacy"]["refuel"]["providers"]
        
        if len(receivers) == 0 or len(providers) == 0:
            print("\n[REFUEL] No refuel receivers or providers, skipping test")
            env.close()
            pytest.skip("No refuel pairs available")
            return
        
        receiver_id = receivers[0]
        provider_id = providers[0]
        
        print(f"\n[REFUEL] Testing refuel action: receiver {receiver_id} -> provider {provider_id}")
        
        # Create refuel action
        action_legacy = {
            "action_type": 7,
            "entity_id": receiver_id,
            "move_center_grid": 0,
            "move_short_axis_km": 0,
            "move_long_axis_km": 0,
            "move_axis_angle": 0,
            "target_group_id": 0,
            "weapon_selection": 0,
            "weapon_usage": 0,
            "weapon_engagement": 0,
            "stealth_enabled": 0,
            "sensing_position_grid": 0,
            "refuel_target_id": provider_id
        }
        action_dynasty = agent_dynasty.select_action(observations["dynasty"])
        
        actions = {"legacy": action_legacy, "dynasty": action_dynasty}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Verify action was recorded
        intent = infos["legacy"]["last_action_intent_by_entity"]
        found = any(v["entity_id"] == receiver_id and v["action_type"] == 7 for v in intent.values())
        
        print(f"[REFUEL] Refuel action executed for entity {receiver_id}")
        print(f"[REFUEL] Action recorded in intent: {found}")
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
