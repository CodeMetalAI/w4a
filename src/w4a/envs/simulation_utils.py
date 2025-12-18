"""
Simulation Utilities

Utility functions for setting up and managing FFSim simulations.
"""

import json
from pathlib import Path

import SimulationInterface
from SimulationInterface import (
    Simulation, SimulationConfig, SimulationData, ForceLaydown, Faction, EntitySpawnData, FactionConfiguration, EntityList, SatelliteSweep
)

from ..entities import w4a_entities


def _load_scenario_data(legacy_force_laydown_path, dynasty_force_laydown_path):
    """
    Load scenario data from JSON files
    
    Args:
        legacy_entity_list_path: Path to Legacy faction entity list JSON
        dynasty_entity_list_path: Path to Dynasty faction entity list JSON  
        legacy_spawn_data_path: Path to Legacy faction spawn data JSON
        dynasty_spawn_data_path: Path to Dynasty faction spawn data JSON
        
    Returns:
        tuple: (mission_events_data, force_laydown_data)
    """
    scenario_path = Path(__file__).parent.parent / "scenarios" / "trident_island"

    # Load scenario data (base mission setup and spawn data per faction)
    with open(scenario_path / "MissionEvents.json") as f:
        mission_events_data = f.read()

    force_laydown_data = {}
    with open(legacy_force_laydown_path) as f:
        force_laydown_data[Faction.LEGACY] = FactionConfiguration.import_json(f.read())

    with open(dynasty_force_laydown_path) as f:
        force_laydown_data[Faction.DYNASTY] = FactionConfiguration.import_json(f.read())

    return mission_events_data, force_laydown_data


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


def _setup_agents(env, legacy_agent, dynasty_agent):
    """
    Setup and add agents to the simulation
    
    Args:
        env: The environment instance
        legacy_agent: _SimulationAgentImpl for Legacy faction
        dynasty_agent: _SimulationAgentImpl for Dynasty faction
    """
    env.simulation.add_agent(legacy_agent)
    env.simulation.add_agent(dynasty_agent)
    
    # Store references for _ensure_passive_agent_laydown
    env.legacy_agent = legacy_agent
    env.dynasty_agent = dynasty_agent


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


def _create_force_laydowns(env, faction_force_laydown_data):
    """
    Create force laydowns for all factions
    
    Args:
        env: The environment instance (TridentIslandEnv)
        faction_force_laydown_data: Spawn data for each faction
        
    Returns:
        dict: Force laydowns keyed by faction
    """
    # Create force laydowns
    force_laydowns = {}

    for faction in [Faction.LEGACY, Faction.DYNASTY]:
        force_laydown = ForceLaydown()
        force_laydown.entity_data = faction_force_laydown_data[faction]

        force_laydowns[faction] = force_laydown
        
    return force_laydowns


def _execute_force_laydown(env, force_laydowns):
    """
    Execute the force laydown phase
    
    Args:
        env: The environment instance (TridentIslandEnv)
        force_laydowns: Force laydowns for all factions
    """
    # Debug: report laydown contents
    try:
        legacy_laydown = force_laydowns[Faction.LEGACY]
        dynasty_laydown = force_laydowns[Faction.DYNASTY]
        def _counts(l):
            return (
                len(l.ground_forces_entities),
                len(l.sea_forces_entities),
                len(l.air_force_squadrons),
                len(l.air_force_packages),
            )
        lg, ls, la, lp = _counts(legacy_laydown)
        dg, ds, da, dp = _counts(dynasty_laydown)
        print(f"[LAYDOWN] Legacy finalize=True counts: ground={lg} sea={ls} air_squadrons={la} air_packages={lp}")
        print(f"[LAYDOWN] Dynasty finalize=True counts: ground={dg} sea={ds} air_squadrons={da} air_packages={dp}")
    except Exception as _e:
        # Non-fatal; diagnostics only
        pass

    # Start force laydown phase
    env.simulation.start_force_laydown(force_laydowns)
    
    # Finalize force laydown phase. Theoretically, we could give the agents time in between these steps, but let's make it immediate for now.
    env.simulation.finalize_force_laydown(env.sim_data)

    # Debug: report event summary from finalize
    try:
        events = env.sim_data.simulation_events
        print(f"[LAYDOWN] finalize produced {len(events)} events")
    except Exception:
        pass

    # Satellite sweep
    if len(env.satellites) != 0:
        player_events = []

        for satellite in env.satellites:
            print(f"Sweeping {satellite.faction.name} satellite!!!")

            sweep = SatelliteSweep()
            sweep.entity = satellite
            
            player_events.append(sweep)

        env.sim_data = SimulationData()
        env.sim_data.player_events = player_events

        env.simulation.pre_simulation_tick(env.sim_data)

        process_simulation_events(env, env.sim_data.simulation_events)

        # Debug: report event summary from satellite sweep
        try:
            events = env.sim_data.simulation_events
            print(f"[LAYDOWN] satellite sweep produced {len(events)} events")
        except Exception:
            pass

    env.sim_data = SimulationData()

    # Process all events coming out of this
    process_simulation_events(env, env.sim_data.simulation_events)


def setup_simulation_from_json(env, legacy_force_laydown, dynasty_force_laydown, 
                              legacy_agent=None, dynasty_agent=None, seed=None):
    """
    Set up simulation using JSON files for force composition and spawn data
    
    Args:
        env: The environment instance (TridentIslandEnv or TridentIslandMultiAgentEnv)
        legacy_force_laydown: Path to Legacy faction faction configuration JSON
        dynasty_force_laydown: Path to Dynasty faction configuration JSON
        legacy_agent: Optional _SimulationAgentImpl for Legacy faction
        dynasty_agent: Optional _SimulationAgentImpl for Dynasty faction
        seed: Optional random seed for simulation
        
    Returns:
        Nothing (simulation is being injected into the environment)
    """
    # Load all scenario data from JSON files
    mission_events_data, faction_force_laydown_data = _load_scenario_data(
        legacy_force_laydown, dynasty_force_laydown
    )
    
    # Create and configure the simulation
    config = _create_simulation_config(env, seed)
    env.simulation = SimulationInterface.create_simulation(config)
    
    # Setup agents
    _setup_agents(env, legacy_agent, dynasty_agent)
    
    # Setup mission events and simulation data
    _setup_mission_events(env, mission_events_data)
    
    # Create force laydowns for all factions
    force_laydowns = _create_force_laydowns(env, faction_force_laydown_data)
    
    # Execute the force laydown phase
    _execute_force_laydown(env, force_laydowns)


def process_simulation_events(env, events):
    """
    Process events from simulation step
    
    Args:
        env: The environment instance 
        events: List of simulation events to process
    """
    # Store all events for debugging
    env.simulation_events = events
    
    for event in events:
        handler = env.simulation_event_handlers.get(type(event))
        if handler:
            handler(event)

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