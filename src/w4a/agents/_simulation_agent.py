"""
Internal simulation agent implementation. Users should inherit from CompetitionAgent.

This module handles all SimulationInterface.Agent plumbing including event
processing, entity tracking, and force laydown.
"""

from SimulationInterface import Agent as SimAgent
from SimulationInterface import (
    Faction, EntitySpawned, EntityDespawned, AdversaryContact, Victory, ComponentSpawned, ControllableEntity, 
    TargetGroup, Flag, CAPManouver, NonCombatManouverQueue,
    CaptureFlagComponent, EntitySpawnComponent, RefuelComponent, RefuelingComponent
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

        self.entity_spawn_entities = set()              # All units capable of spawning new entities (carriers)
        self.refueling_entities = set()                 # All units capable of refueling others
        self.refuelable_entities = set()                # All units capable of refueling
        self.capture_entities = set()                   # All units capable of capturing a flag
        
        # Active operation tracking (entities currently performing operations)
        # Store by RL ID for fast lookup without recomputing ptr->id mapping
        self.active_capturing_entities = {}             # Dict[rl_id -> entity] currently capturing flags
        self.active_refuel_receivers = {}               # Dict[rl_id -> entity] currently receiving fuel
        self.active_refuel_providers = {}               # Dict[rl_id -> entity] currently providing fuel
        
        # Event handlers
        self.simulation_event_handlers = {
            EntitySpawned: self._on_entity_spawned,
            EntityDespawned: self._on_entity_despawned,
            ComponentSpawned: self._on_component_spawned,
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

        self.component_spawned_handlers = {
            CaptureFlagComponent: self._on_capture_flag_component_spawned,
            EntitySpawnComponent: self._on_entity_spawn_component_spawned,
            RefuelComponent: self._on_refuel_component_spawned,
            RefuelingComponent: self._on_refueling_component_spawned,
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

    def _on_component_spawned(self, event):
        handler = self.component_spawned_handlers.get(type(event.component))

        if handler:
            handler(event)
        
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
            group_id = self._target_group_id_by_ptr[ptr] # TODO: We should recycle this number

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

    def _on_capture_flag_component_spawned(self, event):
        self.capture_entities.add(event.component.entity)

    def _on_entity_spawn_component_spawned(self, event):
        self.entity_spawn_entities.add(event.component.entity)

    def _on_refuel_component_spawned(self, event):
        self.refuelable_entities.add(event.component.entity)

    def _on_refueling_component_spawned(self, event):
        self.refueling_entities.add(event.component.entity)
    
    # === Event Hooks (can be called by external systems) ===
    
    def on_refuel_started(self, receiver_entity, provider_entity):
        """Hook: Called when refueling operation starts."""
        receiver_ptr = id(receiver_entity)
        provider_ptr = id(provider_entity)
        
        # Look up RL IDs (similar to _on_controllable_entity_spawned pattern)
        if receiver_ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[receiver_ptr]
            self.active_refuel_receivers[entity_id] = receiver_entity
        
        if provider_ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[provider_ptr]
            self.active_refuel_providers[entity_id] = provider_entity
    
    def on_refuel_completed(self, receiver_entity, provider_entity):
        """Hook: Called when refueling operation completes."""
        receiver_ptr = id(receiver_entity)
        provider_ptr = id(provider_entity)
        
        # Look up RL IDs
        if receiver_ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[receiver_ptr]
            self.active_refuel_receivers.pop(entity_id, None)
        
        if provider_ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[provider_ptr]
            self.active_refuel_providers.pop(entity_id, None)
    
    def on_refuel_interrupted(self, receiver_entity, provider_entity):
        """Hook: Called when refueling operation is interrupted."""
        receiver_ptr = id(receiver_entity)
        provider_ptr = id(provider_entity)
        
        # Look up RL IDs
        if receiver_ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[receiver_ptr]
            self.active_refuel_receivers.pop(entity_id, None)
        
        if provider_ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[provider_ptr]
            self.active_refuel_providers.pop(entity_id, None)
    
    def on_capture_started(self, entity, flag):
        """Hook: Called when entity starts capturing a flag."""
        ptr = id(entity)
        
        # Look up RL ID
        if ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[ptr]
            self.active_capturing_entities[entity_id] = entity
    
    def on_capture_completed(self, entity, flag):
        """Hook: Called when capture completes."""
        ptr = id(entity)
        
        # Look up RL ID
        if ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[ptr]
            self.active_capturing_entities.pop(entity_id, None)
    
    def on_capture_interrupted(self, entity, flag, reason=None):
        """Hook: Called when capture is interrupted."""
        ptr = id(entity)
        
        # Look up RL ID
        if ptr in self._entity_id_by_ptr:
            entity_id = self._entity_id_by_ptr[ptr]
            self.active_capturing_entities.pop(entity_id, None)