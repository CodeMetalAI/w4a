"""
Mission Metrics Management

This module provides functions to track and update mission progress metrics including
kill counts, capture progress, contestation status, and capture capability. These
metrics represent the current tactical situation and progress toward mission objectives.

The metrics are updated each simulation step and used for reward calculation,
termination conditions, and agent observations.
"""

from typing import Dict, List, Set, Any
from SimulationInterface import Entity, EntityDomain
from .utils import *


def update_dead_entities(env: Any) -> None:
    """Track dead entities and update kill counters.

    Scans all entities from both agents and adds newly dead units to the 
    appropriate kill set based on faction.
    
    Args:
        env: Environment instance with agents and kill counters
    """
    from SimulationInterface import Faction
    
    # Iterate over entities from both agents (no list allocation)
    for entity in env.agent_legacy._sim_agent.controllable_entities.values():
        if not entity.is_alive and entity not in env.friendly_kills and entity not in env.enemy_kills:
            if entity.faction == Faction.LEGACY:
                env.friendly_kills.add(entity)
            else:
                env.enemy_kills.add(entity)
    
    for entity in env.agent_dynasty._sim_agent.controllable_entities.values():
        if not entity.is_alive and entity not in env.friendly_kills and entity not in env.enemy_kills:
            if entity.faction == Faction.DYNASTY:
                env.friendly_kills.add(entity)
            else:
                env.enemy_kills.add(entity)


def update_capture_progress(env: Any) -> None:
    """Update capture timer progress based on current capture conditions.
    
    Advances capture progress per-faction when units are actively capturing
    the objective. Progress resets if capture is interrupted.
    
    Args:
        env: Environment instance with per-faction capture state
    """
    # Calculate time delta from simulation parameters
    time_delta = env.frame_rate / 60.0  # frame_rate frames per step / 60 frames per second
    
    # TODO: Implement per-faction capture progress tracking
    # For each faction:
    #   - Get settler units for that faction
    #   - Check if they're in the capture area
    #   - If capturing, advance env.capture_progress_by_faction[faction]
    #   - If not, reset progress
    # TODO: Do we keep track of time for each settler unit? Is there an exposed function for time on target?

def update_island_contested(env: Any) -> None:
    """Check if the capture area is contested by enemy forces.
    
    The area is considered contested when both friendly and enemy forces
    are present within the capture zone simultaneously.
    
    Args:
        env: Environment instance with contestation flag
    """
    # TODO: Implement based on actual capture area definition
    # TODO: Check if our settler units are capturing
    # Is there a function we can use to check if we see enemies capturing?




def update_capture_possible(env: Any) -> None:
    """Check if capture is currently possible for each faction.

    Capture is possible per faction when at least one settler unit from that
    faction is alive and capable of capturing the objective.
    
    Args:
        env: Environment instance with per-faction capture_possible flags
    """
    from SimulationInterface import Faction
    
    # Check Legacy faction for settler units
    legacy_can_capture = False
    for entity in env.agent_legacy._sim_agent.controllable_entities.values():
        if entity.is_alive and entity.can_capture:
            legacy_can_capture = True
            break
    env.capture_possible_by_faction[Faction.LEGACY] = legacy_can_capture
    
    # Check Dynasty faction for settler units
    dynasty_can_capture = False
    for entity in env.agent_dynasty._sim_agent.controllable_entities.values():
        if entity.is_alive and entity.can_capture:
            dynasty_can_capture = True
            break
    env.capture_possible_by_faction[Faction.DYNASTY] = dynasty_can_capture



def update_all_mission_metrics(env: Any) -> None:
    """Update all mission metrics in the correct order.
    
    This function should be called once per simulation step to maintain
    consistent mission progress tracking across all metrics.
    
    Args:
        env: Environment instance
    """
    # Order matters: some functions depend on others
    update_dead_entities(env)
    update_capture_possible(env) 
    update_island_contested(env)
    update_capture_progress(env)


def reset_mission_metrics(env: Any) -> None:
    """Reset all mission metrics to initial values.
    
    This function should be called during environment reset to initialize
    fresh mission tracking for a new episode.
    
    Args:
        env: Environment instance
    """
    from SimulationInterface import Faction
    
    env.friendly_kills = set()  # Set of dead friendly entities
    env.enemy_kills = set()     # Set of dead enemy entities
    
    # Reset per-faction capture tracking
    env.capture_progress_by_faction = {
        Faction.LEGACY: 0.0,
        Faction.DYNASTY: 0.0
    }
    env.capture_possible_by_faction = {
        Faction.LEGACY: True,
        Faction.DYNASTY: True
    }
    env.island_contested = False
