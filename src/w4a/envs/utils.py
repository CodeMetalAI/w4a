"""
Environment Utilities

Pure utility functions used across multiple environment modules.
These are stateless helper functions for common operations.
"""

from typing import Set
from SimulationInterface import Entity


def get_time_elapsed(frame_index: int) -> float:
    """
    Convert frame index to elapsed time in seconds.
    
    Args:
        frame_index: Current simulation frame index
        
    Returns:
        Elapsed time in seconds (frame_index / 60)
    """
    return frame_index / 60  # seconds


def get_settler_units(entities, our_faction) -> Set[Entity]:
    """Get settler units from entities"""
    return {entity for entity in entities.values() if entity.is_alive and entity.faction.value == our_faction and entity.is_settler}