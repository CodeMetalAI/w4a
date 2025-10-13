"""
Internal simulation agent implementation. Users should inherit from CompetitionAgent.

This module handles all SimulationInterface.Agent plumbing including event
processing, entity tracking, and force laydown.
"""

from SimulationInterface import Agent as SimAgent
from SimulationInterface import (
    Faction, EntitySpawned, AdversaryContact, Victory, ControllableEntity, 
    TargetGroup, Flag, CAPManouver, NonCombatManouverQueue
)


class _SimulationAgentImpl(SimAgent):
    """
    Private implementation handling all simulation interface plumbing.
    
    This class is not exposed to users. It handles:
    - Event processing (EntitySpawned, AdversaryContact, etc.)
    - Per-agent entity tracking with stable IDs
    - Per-agent target group tracking with stable IDs
    - Force laydown finalization
    
    Users interact with CompetitionAgent which wraps this class.
    """
    
    def __init__(self, faction: Faction, config):
        super().__init__()
        self.faction = faction
        self.config = config
        
        # Per-agent tracking (instead of environment-global)
        self.controllable_entities = {}  # Dict[entity_id -> controllable entity]
        self._entity_id_by_ptr = {}
        self._next_entity_id = 0
        
        self.target_groups = {}  # Dict[group_id -> target_group]
        self._target_group_id_by_ptr = {}
        self._next_target_group_id = 0
        
        self.flags = {}  # Flags this agent can see
        
        # Event handlers
        self.simulation_event_handlers = {
            EntitySpawned: self._on_entity_spawned,
            AdversaryContact: self._on_adversary_contact,
            Victory: self._on_victory,
        }
    
    def start_force_laydown(self, force_laydown):
        """Called by simulation to start force setup."""
        self.force_laydown = force_laydown
    
    def finalize_force_laydown(self):
        """
        Spawn entities for this faction.
        
        This is the default force laydown implementation.
        """
        force_laydown = self.force_laydown
        entities = []
        
        # Ground forces
        for entity in force_laydown.ground_forces_entities:
            spawn_location = force_laydown.get_random_ground_force_spawn_location()
            entity.pos = spawn_location.pos
            entity.rot = spawn_location.rot
            entities.append(entity)
        
        # Sea forces
        for entity in force_laydown.sea_forces_entities:
            spawn_location = force_laydown.get_random_sea_force_spawn_location()
            entity.pos = spawn_location.pos
            entity.rot = spawn_location.rot
            entities.append(entity)
        
        # Air forces
        for entity in force_laydown.air_forces_entities:
            spawn_location = force_laydown.get_random_air_force_spawn_location()
            entity.pos = spawn_location.pos
            entity.rot = spawn_location.rot
            
            # Set up CAP maneuver for air units
            NonCombatManouverQueue.create(entity.pos, lambda: 
                CAPManouver.create_from_spline_points(force_laydown.get_random_cap().spline_points))
            
            entities.append(entity)
        
        return entities
    
    def pre_simulation_tick(self, simulation_data):
        """Called before each simulation tick."""
        self._process_simulation_events(simulation_data.simulation_events)
    
    def tick(self, simulation_data):
        """Called during each simulation tick."""
        self._process_simulation_events(simulation_data.simulation_events)
    
    def _process_simulation_events(self, events):
        """Dispatch events to handlers."""
        for event in events:
            handler = self.simulation_event_handlers.get(type(event))
            if handler:
                handler(event)
    
    def _on_entity_spawned(self, event):
        """
        Track entities that spawn for this faction.
        
        Only tracks ControllableEntity instances that belong to this faction
        and are actually controllable (excludes carriers, stationary objects, etc).
        Assigns stable entity IDs that persist across the episode.
        """
        entity = event.entity
        
        # Handle flags (shared resource, track separately)
        if isinstance(entity, Flag):
            self.flags[entity.faction] = entity
            return
        
        # Only track our faction's controllable entities
        if not isinstance(entity, ControllableEntity):
            return
        
        if entity.faction != self.faction:
            return
        
        # Only track entities that are actually controllable (not stationary/carriers)
        if not entity.Controllable:
            return
        
        # Assign stable ID
        ptr = id(entity)
        if ptr not in self._entity_id_by_ptr:
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            self._entity_id_by_ptr[ptr] = entity_id
            self.controllable_entities[entity_id] = entity
    
    def _on_adversary_contact(self, event):
        """
        Track enemy target groups detected by this faction.
        
        Assigns stable target group IDs that persist across the episode.
        """
        group = event.target_group
        if group is None:
            return
        
        # Don't track our own faction's groups
        if group.faction == self.faction:
            return
        
        # Assign stable ID
        ptr = id(group)
        if ptr not in self._target_group_id_by_ptr:
            group_id = self._next_target_group_id
            self._next_target_group_id += 1
            self._target_group_id_by_ptr[ptr] = group_id
            self.target_groups[group_id] = group
    
    def _on_victory(self, event):
        """Handle victory events."""
        pass

