"""
Simulation Utilities

Utility functions for setting up and managing FFSim simulations.
"""

import json
from pathlib import Path

import SimulationInterface
from SimulationInterface import (
    Simulation, SimulationConfig, SimulationData, ForceLaydown, Faction, EntitySpawnData, FactionConfiguration, EntityList
)

from ..entities import w4a_entities


def _load_scenario_data(legacy_entity_list_path, dynasty_entity_list_path, 
                       legacy_spawn_data_path, dynasty_spawn_data_path):
    """
    Load scenario data from JSON files
    
    Args:
        legacy_entity_list_path: Path to Legacy faction entity list JSON
        dynasty_entity_list_path: Path to Dynasty faction entity list JSON  
        legacy_spawn_data_path: Path to Legacy faction spawn data JSON
        dynasty_spawn_data_path: Path to Dynasty faction spawn data JSON
        
    Returns:
        tuple: (mission_events_data, faction_entity_spawn_data, faction_entity_data)
    """
    # TODO: This file loading and json parsing should be done in a preprocess step instead of repeating it every time
    scenario_path = Path(__file__).parent.parent / "scenarios" / "trident_island"   # TODO: this should not be hardcoded here

    # Load scenario data (base mission setup and spawn data per faction)
    with open(scenario_path / "MissionEvents.json") as f:
        mission_events_data = f.read()

    faction_entity_spawn_data = {}
    with open(legacy_spawn_data_path) as f:
        faction_entity_spawn_data[Faction.LEGACY] = EntitySpawnData.import_json(f.read())

    with open(dynasty_spawn_data_path) as f:
        faction_entity_spawn_data[Faction.DYNASTY] = EntitySpawnData.import_json(f.read())

    # Load entities lists that have emerged from the auction
    faction_entity_data = {}

    with open(legacy_entity_list_path) as f:
        faction_entity_data[Faction.LEGACY] = EntityList().load_json(f.read())

    with open(dynasty_entity_list_path) as f:
        faction_entity_data[Faction.DYNASTY] = EntityList().load_json(f.read())

    # TODO: All stuff above should be part of the preprocess step
    
    return mission_events_data, faction_entity_spawn_data, faction_entity_data


def _create_simulation_config(env, seed):
    """
    Create simulation configuration
    
    Args:
        env: The environment instance (TridentIslandEnv)
        seed: Optional random seed for simulation
        
    Returns:
        SimulationConfig: Configured simulation config object
    """
    # Create simulation config
    config = SimulationConfig()
    config.name = env.scenario_name
    config.random_seed = seed if seed is not None else (env.config.seed or 42)
    config.log_json = env.enable_replay
    
    return config


def _setup_agents(env):
    """
    Setup and add agents to the simulation
    
    Args:
        env: The environment instance (TridentIslandEnv)
    """
    # TODO: @Sanjna initialize the actual RL agents here
    # env.legacy_agent = SimpleAgent(Faction.LEGACY)
    # env.dynasty_agent = SimpleAgent(Faction.DYNASTY)

    env.simulation.add_agent(env.legacy_agent)
    env.simulation.add_agent(env.dynasty_agent)


def _setup_mission_events(env, mission_events_data):
    """
    Setup mission events and simulation data
    
    Args:
        env: The environment instance (TridentIslandEnv)
        mission_events_data: Raw mission events data from JSON
    """
    env.sim_data = SimulationData()
    env.sim_data.add_mission_events(Simulation.create_mission_events(mission_events_data))

    # Add the mission events to the simulation. Agents will get this data too.
    pre_simulation_tick(env)


def _create_force_laydowns(env, faction_entity_spawn_data, faction_entity_data):
    """
    Create force laydowns for all factions
    
    Args:
        env: The environment instance (TridentIslandEnv)
        faction_entity_spawn_data: Spawn data for each faction
        faction_entity_data: Entity data for each faction
        
    Returns:
        dict: Force laydowns keyed by faction
    """
    # Create force laydowns
    force_laydowns = {}

    for faction in [Faction.LEGACY, Faction.DYNASTY]:
        force_laydown = ForceLaydown()
        force_laydown.entity_spawn_data = faction_entity_spawn_data[faction]
        force_laydown.entity_data = FactionConfiguration().create_entities(faction_entity_data[faction], lambda type: env.simulation.create_mission_event(w4a_entities.get_entity(type)))

        force_laydowns[faction] = force_laydown
        
    return force_laydowns


def _execute_force_laydown(env, force_laydowns):
    """
    Execute the force laydown phase
    
    Args:
        env: The environment instance (TridentIslandEnv)
        force_laydowns: Force laydowns for all factions
    """
    # Debug: report agent capabilities and laydown contents
    legacy_has_finalize = callable(getattr(env.legacy_agent, 'finalize_force_laydown', None))
    dynasty_has_finalize = callable(getattr(env.dynasty_agent, 'finalize_force_laydown', None))

    try:
        legacy_laydown = force_laydowns[Faction.LEGACY]
        dynasty_laydown = force_laydowns[Faction.DYNASTY]
        def _counts(l):
            return (
                len(getattr(l, 'ground_forces_entities', [])),
                len(getattr(l, 'sea_forces_entities', [])),
                len(getattr(l, 'air_force_squadrons', [])),
                len(getattr(l, 'air_force_packages', [])),
            )
        lg, ls, la, lp = _counts(legacy_laydown)
        dg, ds, da, dp = _counts(dynasty_laydown)
        print(f"[LAYDOWN] Legacy finalize={legacy_has_finalize} counts: ground={lg} sea={ls} air_squadrons={la} air_packages={lp}")
        print(f"[LAYDOWN] Dynasty finalize={dynasty_has_finalize} counts: ground={dg} sea={ds} air_squadrons={da} air_packages={dp}")
    except Exception as _e:
        # Non-fatal; diagnostics only
        pass

    # Ensure passive agents expose minimal laydown hooks so the simulation can query them
    _ensure_passive_agent_laydown(env)

    # Start force laydown phase
    env.simulation.start_force_laydown(force_laydowns)
    
    # Finalize force laydown phase. Theoretically, we could give the agents time in between these steps, but let's make it immediate for now.
    env.simulation.finalize_force_laydown(env.sim_data)

    sim_data = env.sim_data

    # Debug: report event summary from finalize
    try:
        events = getattr(sim_data, 'simulation_events', [])
        print(f"[LAYDOWN] finalize produced {len(events)} events")
    except Exception:
        pass

    # Process all events coming out of this
    process_simulation_events(env, sim_data.simulation_events)


def _ensure_passive_agent_laydown(env):
    """
    Attach minimal start/finalize force laydown hooks to agents that don't implement them,
    so spawning is driven by the environment (not the agent wrapper) while keeping RL agents passive.
    """

    def _attach(agent):
        has_start = callable(getattr(agent, 'start_force_laydown', None))
        has_finalize = callable(getattr(agent, 'finalize_force_laydown', None))
        if has_start and has_finalize:
            return

        def start_force_laydown(self, force_laydown):
            setattr(self, '_force_laydown', force_laydown)

        def finalize_force_laydown(self):
            laydown = getattr(self, '_force_laydown', None)
            if laydown is None:
                return []

            entities = []

            for entity in getattr(laydown, 'ground_forces_entities', []):
                loc = laydown.get_random_ground_force_spawn_location()
                entity.pos = loc.pos
                entity.rot = loc.rot
                entities.append(entity)

            for entity in getattr(laydown, 'sea_forces_entities', []):
                loc = laydown.get_random_sea_force_spawn_location()
                entity.pos = loc.pos
                entity.rot = loc.rot
                entities.append(entity)

            # Spawn individual air units as controllable entities
            for unit in getattr(laydown, 'air_force_units', []):
                loc = laydown.get_random_air_force_spawn_location()
                unit.pos = loc.pos
                unit.rot = loc.rot
                entities.append(unit)

            # Packages (if any) are handled by SimpleAgent; keep RL side minimal
            return entities

        # Bind methods to instance
        setattr(agent, 'start_force_laydown', start_force_laydown.__get__(agent, agent.__class__))
        setattr(agent, 'finalize_force_laydown', finalize_force_laydown.__get__(agent, agent.__class__))

    _attach(env.legacy_agent)
    _attach(env.dynasty_agent)


def setup_simulation_from_json(env, legacy_entity_list_path, dynasty_entity_list_path, 
                              legacy_spawn_data_path, dynasty_spawn_data_path, seed=None):
    """
    Set up simulation using JSON files for force composition and spawn data
    
    Args:
        env: The environment instance (TridentIslandEnv)
        legacy_entity_list_path: Path to Legacy faction entity list JSON
        dynasty_entity_list_path: Path to Dynasty faction entity list JSON  
        legacy_spawn_data_path: Path to Legacy faction spawn data JSON
        dynasty_spawn_data_path: Path to Dynasty faction spawn data JSON
        seed: Optional random seed for simulation
        
    Returns:
        Nothing (simulation is being injected into the environment)
    """
    # Load all scenario data from JSON files
    mission_events_data, faction_entity_spawn_data, faction_entity_data = _load_scenario_data(
        legacy_entity_list_path, dynasty_entity_list_path, 
        legacy_spawn_data_path, dynasty_spawn_data_path
    )
    
    # Create and configure the simulation
    config = _create_simulation_config(env, seed)
    env.simulation = SimulationInterface.create_simulation(config)
    
    # Setup agents
    _setup_agents(env)
    
    # Setup mission events and simulation data
    _setup_mission_events(env, mission_events_data)
    
    # Create force laydowns for all factions
    force_laydowns = _create_force_laydowns(env, faction_entity_spawn_data, faction_entity_data)
    
    # Execute the force laydown phase
    _execute_force_laydown(env, force_laydowns)


def process_simulation_events(env, events):
    """
    Process events from simulation step
    
    Args:
        env: The environment instance 
        events: List of simulation events to process
    """
    for event in events:
        handler = getattr(env, 'simulation_event_handlers', {}).get(type(event))
        if handler:
            handler(event)
        else:
            # Uncomment for debugging:
            # print(f"Frame {getattr(env.simulation, 'frame_index', 0)}: Unhandled {event.__class__.__name__}")
            pass

def pre_simulation_tick(env):
    sim_data = env.sim_data

    env.simulation.pre_simulation_tick(sim_data)

    # Set up the simulation data for the next frame
    env.sim_data = SimulationData()

    # Process all events the simulation generated
    process_simulation_events(env, sim_data.simulation_events)

def tick_simulation(env):
    """
    Advance simulation by one time step and process resulting events.
    
    Executes queued player events, advances simulation physics, and processes
    any events generated (entity spawns, deaths, victories, etc.).
    
    Args:
        env: Environment instance with simulation handle and event data
        
    Side Effects:
        - Advances simulation state by env.frame_rate time units
        - Processes all generated events through event handlers
        - Resets env.sim_data for next frame's events
    """

    sim_data = env.sim_data

    env.simulation.tick(sim_data, env.frame_rate)

    # Set up the simulation data for the next frame
    env.sim_data = SimulationData()

    # Process all events the simulation generated
    process_simulation_events(env, sim_data.simulation_events)