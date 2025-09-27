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

from .simple_agent import SimpleAgent

from ..entities import w4a_entities


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
        Created simulation handle
    """

    # TODO: This file loading and json parsing should be done in a preprocess step instead of repeating it every time
    scenario_path = Path(__file__).parent.parent / "scenarios" / "trident_island"   # TODO: this should not be hardcoded here

    with open(scenario_path / "MissionEvents.json") as f:
        mission_events = Simulation.create_mission_events(f.read())

    faction_entity_spawn_data = {}
    with open(legacy_spawn_data_path) as f:
        faction_entity_spawn_data[Faction.LEGACY] = EntitySpawnData.import_json(f.read())

    with open(dynasty_spawn_data_path) as f:
        faction_entity_spawn_data[Faction.DYNASTY] = EntitySpawnData.import_json(f.read())

    faction_entity_data = {}

    with open(legacy_entity_list_path) as f:
        faction_entity_data[Faction.LEGACY] = EntityList().load_json(f.read())

    with open(dynasty_entity_list_path) as f:
        faction_entity_data[Faction.DYNASTY] = EntityList().load_json(f.read())

    # Create simulation config
    config = SimulationConfig()
    config.name = env.scenario_name
    config.random_seed = seed if seed is not None else (env.config.seed or 42)
    config.log_json = env.enable_replay

    simulation = SimulationInterface.create_simulation(config)

    # TODO: @Sanjna initialize the actual RL agents here
    simulation.add_agent(SimpleAgent(Faction.LEGACY))
    simulation.add_agent(SimpleAgent(Faction.DYNASTY))

    sim_data = SimulationData()
    sim_data.add_mission_events(mission_events)    # Don't think this holds up the second time. We might need to recrete them from json every time

    # Load JSON files
    #legacy_entities = _load_entity_list_json(legacy_entity_list_path)
    #dynasty_entities = _load_entity_list_json(dynasty_entity_list_path)
    #legacy_spawn_data = _load_spawn_data_json(legacy_spawn_data_path)
    #dynasty_spawn_data = _load_spawn_data_json(dynasty_spawn_data_path)

    # Create force laydowns
    force_laydowns = {}
    for faction in [Faction.LEGACY, Faction.DYNASTY]:
        force_laydown = ForceLaydown()
        force_laydown.entity_spawn_data = faction_entity_spawn_data[faction]
        force_laydown.entity_data = FactionConfiguration().create_entities(faction_entity_data[faction], lambda type: simulation.create_mission_event(w4a_entities.get_entity(type)))

        force_laydowns[faction] = force_laydown

    # Process the mission setup
    simulation.start_force_laydown(force_laydowns)
    
    sim_data = SimulationData()
    simulation.finalize_force_laydown(sim_data)

    env.sim_data = SimulationData()

    # Process all events coming out of this
    process_simulation_events(env, sim_data.simulation_events)
    
    return simulation


def _load_entity_list_json(file_path):
    """Load entity list from JSON file"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # TODO: Convert JSON data to proper entity data format
    # The JSON has: {"Entities": [{"Type": "F-35C", "Amount": 2, "Identifiers": ["SCAT", "BONG"]}]}
    # Need to convert to format expected by simulation
    return data


def _load_spawn_data_json(file_path):
    """Load entity spawn data from JSON file"""  
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # TODO: Convert JSON data to EntitySpawnData format
    # The JSON has spawn areas, locations, CAPs etc.
    # Need to convert to format expected by simulation
    entity_spawn_data = EntitySpawnData.import_json(json.dumps(data))
    return entity_spawn_data


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