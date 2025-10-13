"""
Internal simulation agent implementation. Users should inherit from CompetitionAgent.

This module handles all SimulationInterface.Agent plumbing including event
processing, entity tracking, and force laydown.
"""

from SimulationInterface import Agent as SimAgent
from SimulationInterface import (
    Faction, EntitySpawned, EntityDespawned, AdversaryContact, Victory, ControllableEntity, 
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
            EntityDespawned: self._on_entity_despawned,
            AdversaryContact: self._on_adversary_contact,
            Victory: self._on_victory,
        }

        self.entity_spawned_handlers = {
            ControllableEntity: self._on_controllable_entity_spawned,
            TargetGroup: self._on_target_group_spawned,
            Flag: self._on_flag_spawned
        }

        self.entity_despawned_handlers = {
            TargetGroup: self._on_target_group_despawned, # Nothing else despawns at this moment
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

        cls = type(event.entity)

        while cls != None:
            handler = self.entity_spawned_handlers.get(cls)

            if handler:
                handler(event)
                return
            else:
                cls = cls.__base__

    def _on_entity_despawned(self, event):
        handler = self.entity_despawned_handlers.get(type(event.entity))

        if handler:
            handler(event)

        entity = event.entity
        
    def _on_flag_spawned(self, event):
        flag = event.entity

        # Handle flags (shared resource, track separately)
        self.flags[flag.faction] = flag

    def _on_target_group_spawned(self, event):
        """
        Track enemy target groups detected by this faction.
        
        Assigns stable target group IDs that persist across the episode.
        """

        group = event.entity
        ptr = id(group)

        assert group.faction == self.faction, f"Agent {self.faction.name} got a target group spawn of faction {group.faction}"

        if ptr not in self._target_group_id_by_ptr:
            group_id = self._next_target_group_id
            self._next_target_group_id += 1
            self._target_group_id_by_ptr[ptr] = group_id
            self.target_groups[group_id] = group

    def _on_target_group_despawned(self, event):
        group = event.entity
        ptr = id(group)

        assert group.faction == self.faction, f"Agent {self.faction.name} got a target group despawn of faction {group.faction}"

        if ptr in self._target_group_id_by_ptr:
            group_id = _target_group_id_by_ptr[ptr] # TODO: We should recycle this number

            del self._target_group_id_by_ptr[ptr]  # Is this ok to do? 
            del self.target_groups[group_id]  # Is this ok to do? 

    def _on_controllable_entity_spawned(self, event):
        entity = event.entity

        assert entity.faction == self.faction, f"Agent {self.faction.name} got an entity spawn of faction {entity.faction} {entity.identifier}"
                
        # Only track entities that are actually controllable (not stationary/carriers)
        if not entity.Controllable: # @Sanjna: I think this is always true in the current setup.
            return
        
        # Assign stable ID
        ptr = id(entity)
        if ptr not in self._entity_id_by_ptr:
            entity_id = self._next_entity_id
            self._next_entity_id += 1
            self._entity_id_by_ptr[ptr] = entity_id
            self.controllable_entities[entity_id] = entity

    def _on_adversary_contact(self, event):
        pass    # Nothing to do here, since we moved the logic to the targetgroup spawned (easier to manage)
    
    def _on_victory(self, event):
        """Handle victory events."""
        pass

