"""
Simulation Utilities

Utility functions for setting up and managing FFSim simulations.
"""

import json
from pathlib import Path

import SimulationInterface
from SimulationInterface import (
    SimulationConfig, SimulationData, ForceLaydown, Faction, EntitySpawnData
)



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
    # Create simulation config
    config = SimulationConfig()
    config.name = env.scenario_name
    config.random_seed = seed if seed is not None else (env.config.seed or 42)
    config.log_json = env.enable_replay

    simulation = SimulationInterface.create_simulation(config)


    # TODO: Do we need to add agents?
    # legacy_agent = SimpleAgent(Faction.LEGACY)
    # dynasty_agent = SimpleAgent(Faction.DYNASTY) 
    # simulation.add_agent(legacy_agent)
    # simulation.add_agent(dynasty_agent)

    # TODO: Do we load in mission events here as well?

    # Load JSON files
    legacy_entities = _load_entity_list_json(legacy_entity_list_path)
    dynasty_entities = _load_entity_list_json(dynasty_entity_list_path)
    legacy_spawn_data = _load_spawn_data_json(legacy_spawn_data_path)
    dynasty_spawn_data = _load_spawn_data_json(dynasty_spawn_data_path)

    # Create force laydowns
    force_laydowns = {}
    
    # Legacy faction
    legacy_laydown = ForceLaydown()
    legacy_laydown.entity_spawn_data = legacy_spawn_data
    legacy_laydown.entity_data = legacy_entities
    force_laydowns[Faction.LEGACY] = legacy_laydown
    
    # Dynasty faction  
    dynasty_laydown = ForceLaydown()
    dynasty_laydown.entity_spawn_data = dynasty_spawn_data
    dynasty_laydown.entity_data = dynasty_entities
    force_laydowns[Faction.DYNASTY] = dynasty_laydown

    # Process the mission setup
    simulation.start_force_laydown(force_laydowns)
    
    sim_data = SimulationData()
    simulation.finalize_force_laydown(sim_data)

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

    env.simulation_handle.tick(sim_data, env.frame_rate)

    # Set up the simulation data for the next frame
    env.sim_data = SimulationData()

    # Process all events the simulation generated
    process_simulation_events(env, sim_data.simulation_events)