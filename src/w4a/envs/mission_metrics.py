"""
Mission Metrics Management

Functions to update and track mission progress metrics like kills, capture progress,
contestation status, and capture capability.

These metrics represent the tactical situation and mission objectives progress.
"""

from typing import Dict, List, Set, Any
from SimulationInterface import Entity, EntityDomain
from .utils import *


def update_dead_entities(env: Any) -> None:
    """
    Track dead entities and update kill sets.

    
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
    """
    Update capture timer progress based on current capture conditions.
    
    Only advance progress if friendly units are currently capturing.
    
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
    """
    Check if the island/capture area is contested by enemy forces.
    
    Island is contested if both friendly and enemy forces are present

    
    Args:
        env: Environment instance with contestation flag
    """
    # TODO: Implement based on actual capture area definition
    # TODO: Check if our settler units are capturing
    # Is there a function we can use to check if we see enemies capturing?




def update_capture_possible(env: Any) -> None:
    """
    Check if capture is currently possible (we have capture-capable units alive).

    Capture is possible if we have at least one alive unit that can capture.
    
    Args:
        env: Environment instance with capture_possible flag  
    """
    for settler_unit in get_settler_units(env.entities, env.config.our_faction):
        if settler_unit.is_alive:
            env.capture_possible = True
            return    
    env.capture_possible = False



def update_all_mission_metrics(env: Any) -> None:
    """
    Update all mission metrics in correct order.
    
    Call this once per step to maintain consistent mission progress tracking.
    
    Args:
        env: Environment instance
    """
    # Order matters: some functions depend on others
    update_dead_entities(env)
    update_capture_possible(env) 
    update_island_contested(env)
    update_capture_progress(env)


def reset_mission_metrics(env: Any) -> None:
    """
    Reset all mission metrics to initial values.
    
    Call this during environment reset to start fresh mission tracking.
    
    Args:
        env: Environment instance
    """
    env.friendly_kills = set()  # Set of dead friendly entities
    env.enemy_kills = set()     # Set of dead enemy entities  
    env.capture_timer_progress = 0.0
    env.island_contested = False
    env.capture_possible = True
