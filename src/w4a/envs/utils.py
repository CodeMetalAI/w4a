"""
Environment Utilities

This module provides pure utility functions used across multiple environment modules.

"""

from typing import Set
from SimulationInterface import Entity


def get_time_elapsed(frame_index: int) -> float:
    """Convert simulation frame index to elapsed time in seconds.
    
    The simulation runs at 60 frames per second, so this function
    converts frame indices to real-time seconds for timing calculations.
    
    Args:
        frame_index: Current simulation frame index
        
    Returns:
        Elapsed time in seconds (frame_index / 60)
    """
    return frame_index / 60  # seconds


def get_settler_units(entities, our_faction) -> Set[Entity]:
    """Get all settler units belonging to our faction.
    
    Settler units are entities capable of capturing objectives in the mission.
    This function filters the entity collection to find only alive settler units
    from the specified faction.
    
    Args:
        entities: Dictionary of all entities in the simulation
        our_faction: Faction value to filter for
        
    Returns:
        Set of settler entities that are alive and belong to our faction
    """
    return {entity for entity in entities.values() if entity.is_alive and entity.faction.value == our_faction and entity.can_capture}