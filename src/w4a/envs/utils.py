"""
Environment Utilities

This module provides pure utility functions used across multiple environment modules.

"""

from typing import Set, Tuple, Any
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


def calculate_max_grid_positions(config: Any) -> int:
    """Calculate maximum number of grid positions for the map.
    
    Args:
        config: Environment configuration with map and grid parameters
        
    Returns:
        Total number of discrete grid positions available
    """
    grid_size = int((config.map_size_km[0]) / config.grid_resolution_km)
    return grid_size * grid_size


def grid_to_position(grid_index: int, config: Any) -> Tuple[float, float]:
    """Convert discrete grid index to world coordinates.
    
    Transforms the agent's discrete position choice into continuous
    world coordinates for use in the simulation.
    
    Args:
        grid_index: Discrete grid position index
        config: Environment configuration with grid parameters
        
    Returns:
        Tuple of (x, y) world coordinates in meters
    """
    grid_size = int(config.map_size_km[0] / config.grid_resolution_km)  # Grid size in cells
    
    grid_x = grid_index % grid_size
    grid_y = grid_index // grid_size
    
    # Convert to world coordinates (meters)
    # First convert grid position to meters, then center by subtracting half map size in meters
    world_x = (grid_x * config.grid_resolution_km * 1000) - (config.map_size_km[0] * 1000 // 2)
    world_y = (grid_y * config.grid_resolution_km * 1000) - (config.map_size_km[1] * 1000 // 2)
    
    return world_x, world_y


def position_in_bounds(x: float, y: float, config: Any) -> bool:
    """Check if world position is within map boundaries.
    
    Args:
        x, y: World coordinates in meters
        config: Environment configuration with map size
        
    Returns:
        True if position is within map bounds
    """
    half_map = config.map_size_km[0] * 1000 // 2
    return abs(x) <= half_map and abs(y) <= half_map