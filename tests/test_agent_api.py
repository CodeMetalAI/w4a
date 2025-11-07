"""
Tests for CompetitionAgent public API methods.

Verifies that direct lookup and capability query API methods work correctly and are faction-scoped.
"""

import pytest
from w4a import Config
from w4a.envs import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction


class TestDirectLookups:
    """Test direct lookup methods (get_entity_by_id, get_target_group_by_id)."""
    
    def test_get_entity_by_id_valid(self):
        """Verify get_entity_by_id returns correct entity for valid ID."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get valid entity IDs from controllable_entities
        legacy_entity_ids = list(infos["legacy"]["valid_masks"]["controllable_entities"])
        
        assert len(legacy_entity_ids) > 0, "Should have entities"
        
        # Test lookup by ID
        entity_id = legacy_entity_ids[0]
        entity = agent_legacy.get_entity_by_id(entity_id)
        
        assert entity is not None, f"Should find entity with ID {entity_id}"
        assert entity.faction == Faction.LEGACY, "Entity should belong to Legacy"
        assert entity.is_alive, "Entity should be alive"
        
        env.close()
    
    def test_get_entity_by_id_invalid(self):
        """Verify get_entity_by_id returns None for invalid ID."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Test with invalid ID (far beyond max_entities)
        entity = agent_legacy.get_entity_by_id(9999)
        
        assert entity is None, "Should return None for invalid ID"
        
        env.close()
    
    def test_get_entity_by_id_faction_scoped(self):
        """Verify get_entity_by_id only returns entities from agent's faction."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = CompetitionAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get entity IDs for both factions
        legacy_ids = list(infos["legacy"]["valid_masks"]["controllable_entities"])
        dynasty_ids = list(infos["dynasty"]["valid_masks"]["controllable_entities"])
        
        # Legacy agent should only see Legacy entities
        for entity_id in legacy_ids:
            entity = agent_legacy.get_entity_by_id(entity_id)
            assert entity is not None
            assert entity.faction == Faction.LEGACY
        
        # Dynasty agent should only see Dynasty entities
        for entity_id in dynasty_ids:
            entity = agent_dynasty.get_entity_by_id(entity_id)
            assert entity is not None
            assert entity.faction == Faction.DYNASTY
        
        # Cross-faction lookup should return None (IDs are independent per agent)
        # Since IDs are per-agent, Dynasty ID 0 is different from Legacy ID 0
        
        env.close()
    
    def test_get_target_group_by_id_valid(self):
        """Verify get_target_group_by_id returns correct target group for valid ID."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Step forward to allow detection
        found_target = False
        for step in range(20):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if any target groups detected
            legacy_target_ids = list(infos["legacy"]["valid_masks"]["visible_targets"])
            
            if len(legacy_target_ids) > 0:
                # Test lookup by ID
                group_id = legacy_target_ids[0]
                target_group = agent_legacy.get_target_group_by_id(group_id)
                
                assert target_group is not None, f"Should find target group with ID {group_id}"
                assert target_group.faction == Faction.LEGACY, "Target group should be tagged with Legacy faction"
                found_target = True
                break
            
            if terminations["legacy"]:
                break
        
        assert found_target, "Should have detected at least one target group within 20 steps"
        env.close()
    
    def test_get_target_group_by_id_invalid(self):
        """Verify get_target_group_by_id returns None for invalid ID."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Test with invalid ID
        target_group = agent_legacy.get_target_group_by_id(9999)
        
        assert target_group is None, "Should return None for invalid ID"
        
        env.close()


class TestCapabilityQueries:
    """Test capability query methods."""
    
    def test_get_capture_capable_entities(self):
        """Verify get_capture_capable_entities returns only entities with can_capture=True."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get capture-capable entities
        capture_entities = agent_legacy.get_capture_capable_entities()
        
        # Verify all have can_capture=True
        for entity in capture_entities:
            assert entity.can_capture, f"Entity {entity.identifier} should have can_capture=True"
            assert entity.is_alive, f"Entity {entity.identifier} should be alive"
            assert entity.faction == Faction.LEGACY, f"Entity {entity.identifier} should be Legacy"
        
        # Verify we're not missing any (compare with manual filter)
        all_entities = agent_legacy.get_alive_entities()
        manual_capture = [e for e in all_entities if e.can_capture]
        
        assert len(capture_entities) == len(manual_capture), "Should return all capture-capable entities"
        
        env.close()
    
    def test_get_refuelable_entities(self):
        """Verify get_refuelable_entities returns only entities with can_refuel=True."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get refuelable entities
        refuelable = agent_legacy.get_refuelable_entities()
        
        # Verify all have can_refuel=True
        for entity in refuelable:
            assert entity.can_refuel, f"Entity {entity.identifier} should have can_refuel=True"
            assert entity.is_alive, f"Entity {entity.identifier} should be alive"
            assert entity.faction == Faction.LEGACY, f"Entity {entity.identifier} should be Legacy"
        
        # Verify we're not missing any (compare with manual filter)
        all_entities = agent_legacy.get_alive_entities()
        manual_refuelable = [e for e in all_entities if e.can_refuel]
        
        assert len(refuelable) == len(manual_refuelable), "Should return all refuelable entities"
        
        env.close()
    
    def test_capability_queries_faction_scoped(self):
        """Verify capability queries are faction-scoped."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = CompetitionAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get capture-capable entities for both factions
        legacy_capture = agent_legacy.get_capture_capable_entities()
        dynasty_capture = agent_dynasty.get_capture_capable_entities()
        
        # Verify all Legacy entities are Legacy faction
        for entity in legacy_capture:
            assert entity.faction == Faction.LEGACY
        
        # Verify all Dynasty entities are Dynasty faction
        for entity in dynasty_capture:
            assert entity.faction == Faction.DYNASTY
        
        env.close()


class TestOperationChecks:
    """Test operation check methods."""
    
    def test_is_entity_capturing_with_id(self):
        """Verify is_entity_capturing works with entity ID without errors."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get entity IDs from masks
        entity_ids = list(infos["legacy"]["valid_masks"]["controllable_entities"])
        assert len(entity_ids) > 0, "Should have at least one entity"
        
        entity_id = entity_ids[0]
        
        # Verify method works with entity ID (returns False initially)
        result = agent_legacy.is_entity_capturing(entity_id)
        assert isinstance(result, bool), "Should return boolean"
        assert not result, "Entity should not be capturing initially"
        
        # Verify method works after stepping (should not crash)
        for _ in range(5):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Method should work without errors
            result = agent_legacy.is_entity_capturing(entity_id)
            assert isinstance(result, bool)
            
            if terminations["legacy"]:
                break
        
        env.close()
    
    def test_is_entity_capturing_all_entities(self):
        """Verify is_entity_capturing works for all entity IDs."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get all entity IDs
        entity_ids = list(infos["legacy"]["valid_masks"]["controllable_entities"])
        assert len(entity_ids) > 0, "Should have entities"
        
        # Verify method works for all entities
        for entity_id in entity_ids:
            result = agent_legacy.is_entity_capturing(entity_id)
            assert isinstance(result, bool), f"Should return boolean for entity {entity_id}"
            assert not result, f"Entity {entity_id} should not be capturing initially"
        
        env.close()
    
    def test_is_entity_refueling_with_id(self):
        """Verify is_entity_refueling works with entity ID and returns False initially."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get all entity IDs from masks
        entity_ids = list(infos["legacy"]["valid_masks"]["controllable_entities"])
        assert len(entity_ids) > 0, "Should have entities"
        
        # Check none are refueling initially
        for entity_id in entity_ids:
            result = agent_legacy.is_entity_refueling(entity_id)
            assert isinstance(result, bool), f"Should return boolean for entity {entity_id}"
            assert not result, f"Entity {entity_id} should not be refueling initially"
        
        env.close()
    
    def test_is_entity_refueling_all_entities(self):
        """Verify is_entity_refueling works for all entity IDs."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Get all entity IDs
        entity_ids = list(infos["legacy"]["valid_masks"]["controllable_entities"])
        assert len(entity_ids) > 0, "Should have entities"
        
        # Verify method works for all entities
        for entity_id in entity_ids:
            result = agent_legacy.is_entity_refueling(entity_id)
            assert isinstance(result, bool), f"Should return boolean for entity {entity_id}"
            assert not result, f"Entity {entity_id} should not be refueling initially"
        
        env.close()
    
    def test_operation_checks_with_invalid_input(self):
        """Verify operation checks handle invalid input gracefully."""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # Test with invalid entity ID
        assert not agent_legacy.is_entity_capturing(9999), "Should return False for invalid ID"
        assert not agent_legacy.is_entity_refueling(9999), "Should return False for invalid ID"
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

