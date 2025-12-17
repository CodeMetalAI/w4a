"""
Simple rule-based agent for testing and as adversary.

Matches the original SimpleAgent behavior exactly while extending CompetitionAgent.
"""

from .competition_agent import CompetitionAgent
from SimulationInterface import (
    Faction, PlayerEventCommit, AdversaryContact, EntitySpawned, 
    ComponentSpawned, ControllableEntity, UnitEngagement, UnitWeaponUsage
)


class SimpleAgent(CompetitionAgent):
    """
    Simple heuristic agent that auto-engages detected enemies.
    
    This agent matches the original SimpleAgent behavior:
    - Automatically commits weapons when enemies are detected  
    - Logs simulation events
    - Uses aggressive engagement parameters
    - Serves as baseline adversary for testing
    """
    
    def __init__(self, faction: Faction, config):
        super().__init__(faction, config)
        
        self.log("Constructed")
        
        # Track entities (matches original: uses set not dict)
        self.entities = set()
        
        # Player events list (matches original)
        self.player_events = []
        
        # Override event handlers (matches original handlers)
        self._sim_agent.simulation_event_handlers[EntitySpawned] = self.__entity_spawned
        self._sim_agent.simulation_event_handlers[AdversaryContact] = self.__adversary_contact
        self._sim_agent.simulation_event_handlers[ComponentSpawned] = self.__component_spawned
    
    @property
    def log_prefix(self):
        """Prefix for log messages (matches original)."""
        # frame_index comes from SimAgent base class
        return f"{self.faction.name} {self.__class__.__name__} (frame {self._sim_agent.frame_index}): "
    
    def log(self, message):
        """Log a message with faction prefix (matches original)."""
        print(f"{self.log_prefix}{message}")
    
    def select_action(self, observation):
        """
        SimpleAgent doesn't use RL actions - it responds via events.
        Return noop action for PettingZoo compatibility.
        """
        return {
            'action_type': 0,
            'entity_id': 0,
            'move_center_grid': 0,
            'move_short_axis_km': 0,
            'move_long_axis_km': 0,
            'move_axis_angle': 0,
            'target_group_id': 0,
            'weapon_selection': 0,
            'weapon_usage': 0,
            'weapon_engagement': 0,
            'stealth_enabled': 0,
            'sensing_position_grid': 0,
            'refuel_target_id': 0,
        }
    
    def __entity_spawned(self, event):
        """Handle entity spawned events (matches original)."""
        # Call parent to track in _sim_agent
        self._sim_agent._on_entity_spawned(event)
        
        # Original behavior
        if issubclass(event.entity.__class__, ControllableEntity):
            self.__controllable_entity_spawned(event)
    
    def __component_spawned(self, event):
        """Handle component spawned events (matches original)."""
        self.log(f"{event.component.__class__.__name__} {event.component.entity.identifier} spawned")
    
    def __controllable_entity_spawned(self, event):
        """Handle controllable entity spawned (matches original)."""
        if event.entity.has_parent:
            return
        
        self.log(f"{event.entity.__class__.__name__} {event.entity.identifier} spawned")
        self.entities.add(event.entity)
    
    def __adversary_contact(self, event):
        """Auto-engage on adversary contact (matches original)."""
        # Call parent to track in _sim_agent
        self._sim_agent._on_adversary_contact(event)
        
        # Original auto-engage logic
        selected_weapons = event.entity.select_weapons(event.target_group, False)
        if len(selected_weapons) == 0:
            return
        
        commit = PlayerEventCommit()
        commit.entity = event.entity
        commit.target_group = event.target_group
        commit.manouver_data.throttle = 1.0
        commit.manouver_data.engagement = UnitEngagement(2)
        commit.manouver_data.weapon_usage = UnitWeaponUsage(2)
        commit.manouver_data.weapons = selected_weapons.keys()
        commit.manouver_data.wez_scale = 1
        
        self.player_events.append(commit)

