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

    Scans all entities and adds newly dead units to the appropriate kill set
    based on faction (friendly casualties vs enemy kills).
    
    Args:
        env: Environment instance with entities dict and kill counters
    """
    for entity in env.entities.values():
        if not entity.is_alive:
            if entity.faction.value == env.config.our_faction:
                env.friendly_kills.add(entity)  # Our casualties
            else:
                env.enemy_kills.add(entity)     # Enemy casualties


def update_capture_progress(env: Any) -> None:
    """Update capture timer progress based on current capture conditions.
    
    Advances capture progress only when friendly units are actively capturing
    the objective. Progress resets if capture is interrupted.
    
    Args:
        env: Environment instance with capture state variables
    """
    # Calculate time delta from simulation parameters
    time_delta = env.frame_rate / 60.0  # frame_rate frames per step / 60 frames per second
    capture_timer_progress = 0.0
    # TODO: Implement this
    
    # get settler units
    # for each settler unit, check if its at capture area
    # if is currently capturing, advance progress
    # if not, reset progress
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
    """Check if capture is currently possible.

    Capture is possible when at least one friendly settler unit is alive
    and capable of capturing the objective.
    
    Args:
        env: Environment instance with capture_possible flag  
    """
    for settler_unit in get_settler_units(env.entities, env.config.our_faction):
        if settler_unit.is_alive:
            env.capture_possible = True
            return    
    env.capture_possible = False



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
    env.friendly_kills = set()  # Set of dead friendly entities
    env.enemy_kills = set()     # Set of dead enemy entities  
    env.capture_timer_progress = 0.0
    env.island_contested = False
    env.capture_possible = True
