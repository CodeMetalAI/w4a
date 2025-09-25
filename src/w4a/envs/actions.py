"""
Action execution and validation functions for TridentIslandEnv.

Pure functions with explicit dependencies for better testability and maintainability.
"""

from typing import Dict, List, Tuple
from ..config import Config

import math

from SimulationInterface import (
    PlayerEventCommit, NonCombatManouverQueue, MoveManouver, CAPManouver, RTBManouver,
    SetRadarFocus, ClearRadarFocus, SetRadarStrength,
    Vector3, Formation, ControllableEntity, EntityDomain,
)


def execute_action(action: Dict, entities: Dict, target_groups: Dict, config: Config) -> List:
    """Execute action and return list of player events.
    
    Args:
        action: Action dictionary from agent
        entities: Dict of entity_id -> entity objects
        target_groups: Dict of target_group_id -> target_group objects  
        config: Environment configuration
        
    Returns:
        List of PlayerEvent objects to submit to FFSim
    """
    action_type = action["action_type"]
    entity_id = action["entity_id"]
    
    if not is_valid_action(action, entities, target_groups, config):
        return []
    
    player_events = []
    
    if action_type == 0:  # No-op
        pass
    elif action_type == 1:  # Move
        event = execute_move_action(entity_id, action, entities, config)
        player_events.append(event)
    elif action_type == 2:  # Engage
        event = execute_engage_action(entity_id, action, entities, target_groups)
        player_events.append(event)
    elif action_type == 3:  # Sense
        event = execute_set_radar_focus_action(entity_id, action, entities, config) # @Sanjna: we need the other ones here too.
        player_events.append(event)
    elif action_type == 4:  # Land
        event = execute_land_action(entity_id, action, entities)
        player_events.append(event)
    elif action_type == 5:  # RTB
        event = execute_rtb_action(entity_id, action, entities)
        player_events.append(event)
    elif action_type == 6:  # Refuel
        event = execute_refuel_action(entity_id, action, entities)
        player_events.append(event)
    
    return player_events


def is_valid_action(action: Dict, entities: Dict, target_groups: Dict, config: Config) -> bool:
    """Validate action against FFSim constraints and game state.
    
    Args:
        action: Action dictionary from agent
        entities: Dict of entity_id -> entity objects
        target_groups: Dict of target_group_id -> target_group objects
        config: Environment configuration
        
    Returns:
        True if action is valid, False otherwise
    """
    if not validate_entity(action, entities, config):
        return False
    
    action_type = action["action_type"]
    
    if action_type == 0:  # No-op
        return True
    elif action_type == 1:  # Move
        return validate_move_action(action, entities, config)
    elif action_type == 2:  # Engage
        return validate_engage_action(action, entities, target_groups)
    elif action_type == 3:  # Sense
        return validate_sense_action(action, entities, config)
    elif action_type == 4:  # Land
        return validate_land_action(action, entities)
    elif action_type == 5:  # RTB
        return validate_rtb_action(action, entities)
    elif action_type == 6:  # Refuel
        return validate_refuel_action(action, entities)
    
    return False


# =============================================================================
# ACTION EXECUTION FUNCTIONS
# =============================================================================

def execute_move_action(entity_id: int, action: Dict, entities: Dict, config: Config):
    """Execute move action - create CAP maneuver"""
    center_x, center_y = grid_to_position(action["move_center_grid"], config)
    short_axis_m = action["move_short_axis_km"] * 1000
    long_axis_m = action["move_long_axis_km"] * 1000 #config.min_patrol_axis_km + (action["move_long_axis_km"] * config.patrol_axis_increment_km) * 1000
    axis_angle = math.radians(action["move_axis_angle"] * config.angle_resolution_degrees)
    
    # Create CAP route using FFSim pattern
    entity = entities[entity_id]

    center = Vector3(center_x, center_y, entity.pos.z)

    axis = Vector3(math.cos(axis_angle), math.sin(axis_angle), 0)

    cap_maneuver = NonCombatManouverQueue.create(entity.pos, lambda: CAPManouver.create_race_track(center, short_axis_m, long_axis_m, axis, 32))
    cap_maneuver.entity = entity

    return cap_maneuver


def execute_engage_action(entity_id: int, action: Dict, entities: Dict, target_groups: Dict):
    """Execute engage action - create combat commit"""
    target_group_id = action["target_group_id"]
    weapon_selection = action["weapon_selection"]
    weapon_usage = action["weapon_usage"]
    weapon_engagement = action["weapon_engagement"]
    
    entity = entities[entity_id]
    target_group = target_groups[target_group_id]
    
    # Get weapons compatible with target group (FFSim determines domain compatibility)
    available_weapons = entity.select_weapons(target_group, False)
    
    # RL agent selects which compatible weapons to use
    selected_weapons = select_weapons_from_available(available_weapons, weapon_selection)
    
    # TODO: Check if selected weapons have ammo available
    
    commit = PlayerEventCommit()
    commit.entity = entity
    commit.target_group = target_group
    commit.manouver_data.throttle = 1.0  # Always max throttle
    commit.manouver_data.engagement = weapon_engagement
    commit.manouver_data.weapon_usage = weapon_usage  # 0=1/unit, 1=1/adversary, 2=2/adversary
    commit.manouver_data.weapons = selected_weapons.keys()
    commit.manouver_data.wez_scale = 1  # Always 1
    
    return commit

def execute_set_radar_focus_action(entity_id: int, action: Dict, entities: Dict, config: Config):
    """Execute sense action - point radar at location"""
    sense_x, sense_y = grid_to_position(action["sense_grid"], config)
    
    entity = entities[entity_id]
    
    event = SetRadarFocus()
    event.entity = entity
    event.position = Vector3(sense_x, sense_y, entity.pos.z)
    
    return event

def execute_clear_radar_focus_action(entity_id: int, action: Dict, entities: Dict, config: Config):
    """Execute sense action - radar back to default"""

    entity = entities[entity_id]
    
    event = ClearRadarFocus()
    event.entity = entity

    return event

def execute_set_radar_strength_action(entity_id: int, action: Dict, entities: Dict, config: Config):
    """Execute sense action - set radar strength"""
    radar_strength = action["radar_strength"]
    
    entity = entities[entity_id]
    
    event = SetRadarStrength()
    event.entity = entity
    event.strength = radar_strength
    
    return event

def execute_land_action(entity_id: int, action: Dict, entities: Dict):
    """Execute land action - land at nearest friendly airbase"""
    entity = entities[entity_id]
    
    # TODO: Create PlayerEventLand when available in FFSim
    # land_event = PlayerEventLand()
    # land_event.entity = entity
    
    # Placeholder for now
    return None


def execute_rtb_action(entity_id: int, action: Dict, entities: Dict):
    """Execute RTB action - return to base"""
    entity = entities[entity_id]
    
    rtb_maneuver = RTBManouver()
    # TODO: Configure RTB maneuver with entity's home base
    # rtb_maneuver.target_base = entity.home_base
    
    return rtb_maneuver


def execute_refuel_action(entity_id: int, action: Dict, entities: Dict):
    """Execute refuel action - refuel from another entity"""
    refuel_target_id = action["refuel_target_id"]
    
    entity = entities[entity_id]
    refuel_target = entities[refuel_target_id]
    
    # TODO: Create PlayerEventRefuel when available in FFSim
    # refuel_event = PlayerEventRefuel()
    # refuel_event.entity = entity
    # refuel_event.refuel_source = refuel_target
    
    # Placeholder for now
    return None


# =============================================================================
# ACTION VALIDATION FUNCTIONS
# =============================================================================

def validate_entity(action: Dict, entities: Dict, config: Config) -> bool:
    """Validate entity exists and is controllable."""
    entity_id = action["entity_id"]
    
    # Check entity exists
    if entity_id not in entities:
        return False
    
    entity = entities[entity_id]
    
    # Check entity is controllable
    if not isinstance(entity, ControllableEntity):
        return False
        
    # Check entity is alive
    if not entity.is_alive:
        return False
        
    # Check entity belongs to our faction
    if entity.faction.value != config.our_faction:
        return False
        
    return True


def validate_move_action(action: Dict, entities: Dict, config: Config) -> bool:
    """Validate move action parameters."""
    entity_id = action["entity_id"]
    center_grid = action["move_center_grid"]
    
    entity = entities[entity_id]
    
    # Check grid position is within map bounds
    max_grid_positions = calculate_max_grid_positions(config)
    if center_grid >= max_grid_positions:
        return False
    
    # Convert to world coordinates and check bounds
    world_x, world_y = grid_to_position(center_grid, config)
    if not position_in_bounds(world_x, world_y, config):
        return False
    
    # Check entity is capable of movement (air units for CAP)
    if entity.domain != EntityDomain.AIR:  # Only air units can do CAP
        return False
    
    return True


def validate_engage_action(action: Dict, entities: Dict, target_groups: Dict) -> bool:
    """Validate engage action parameters."""
    entity_id = action["entity_id"]
    target_group_id = action["target_group_id"]
    weapon_selection = action["weapon_selection"]
    
    entity = entities[entity_id]
    
    # Check target group exists
    if target_group_id not in target_groups:
        return False
    
    target_group = target_groups[target_group_id]
    
    # Check target group is enemy faction
    if target_group.faction.value == entity.faction.value:
        return False
    
    # Check entity has weapons compatible with target
    available_weapons = entity.select_weapons(target_group, False)
    if len(available_weapons) == 0:
        return False
    
    # Check weapon selection is valid for available weapons
    valid_combinations = get_valid_weapon_combinations(available_weapons)
    if weapon_selection not in valid_combinations:
        return False
    
    return True


def validate_sense_action(action: Dict, entities: Dict, config: Config) -> bool:
    """Validate sense action parameters."""
    entity_id = action["entity_id"]
    sense_grid = action["sense_grid"]
    
    entity = entities[entity_id]
    
    # Check grid position is within map bounds
    max_grid_positions = calculate_max_grid_positions(config)
    if sense_grid >= max_grid_positions:
        return False
    
    # Convert to world coordinates and check bounds
    world_x, world_y = grid_to_position(sense_grid, config)
    if not position_in_bounds(world_x, world_y, config):
        return False
    
    # Check entity has radar/sensors
    if not entity.has_radar:
        return False
    
    return True


def validate_land_action(action: Dict, entities: Dict) -> bool:
    """Validate land action parameters."""
    entity_id = action["entity_id"]
    entity = entities[entity_id]
    
    # Check entity is aircraft
    if entity.domain != EntityDomain.AIR:
        return False
    
    # Check entity can land
    if not entity.can_land:
        return False
    
    return True


def validate_rtb_action(action: Dict, entities: Dict) -> bool:
    """Validate RTB action parameters."""
    entity_id = action["entity_id"]
    entity = entities[entity_id]
    
    # Check entity is aircraft
    if entity.domain != EntityDomain.AIR:
        return False
    
    return True


def validate_refuel_action(action: Dict, entities: Dict) -> bool:
    """Validate refuel action parameters."""
    entity_id = action["entity_id"]
    refuel_target_id = action["refuel_target_id"]
    
    entity = entities[entity_id]
    
    # Check refuel target exists
    if refuel_target_id not in entities:
        return False
    
    refuel_target = entities[refuel_target_id]
    
    # Check both entities are same faction
    if entity.faction.value != refuel_target.faction.value:
        return False
    
    # Check refuel target can provide fuel
    if not refuel_target.can_refuel_others: # TODO: Check its a refuel unit
        return False
    
    return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def select_weapons_from_available(available_weapons: Dict, selection_index: int) -> Dict:
    """Select specific weapons from FFSim-compatible weapons using combinatorial selection.
    
    Args:
        available_weapons: Dict from entity.select_weapons() containing compatible weapons
        selection_index: Agent's weapon combination choice (0 to max_weapon_combinations-1)
        
    Returns:
        Dict containing selected weapons
    """
    available_keys = list(available_weapons.keys())
    
    if len(available_keys) == 0:
        return {}  # No compatible weapons available
    
    # Convert selection_index to binary combination
    # selection_index 0 maps to combination 1 (first weapon only)
    # selection_index 1 maps to combination 2 (second weapon only)  
    # selection_index 2 maps to combination 3 (first + second weapons)
    # etc.
    num_available = len(available_keys)
    max_combinations = (1 << num_available) - 1  # 2^n - 1
    
    # Ensure selection_index is valid and avoid empty selection
    combination_index = (selection_index % max_combinations) + 1
    
    result = {}
    for i, key in enumerate(available_keys):
        if combination_index & (1 << i):  # Check if bit i is set
            result[key] = available_weapons[key]
    
    return result


def get_valid_weapon_combinations(available_weapons: Dict) -> List[int]:
    """Get valid combination indices for current available weapons.
    
    Args:
        available_weapons: Dict from entity.select_weapons() containing compatible weapons
        
    Returns:
        List of valid selection_index values for current available weapons
    """
    num_available = len(available_weapons)
    if num_available == 0:
        return []
    
    # Valid combinations: 1 to 2^num_available - 1 (mapped to selection indices 0 to 2^n-2)
    max_combinations = (1 << num_available) - 1  # 2^n - 1
    return list(range(max_combinations))  # [0, 1, 2, ...] for selection indices


def grid_to_position(grid_index: int, config: Config) -> Tuple[float, float]:
    """Convert grid index to world coordinates"""

    grid_size = int(config.map_size[0] / config.grid_resolution_km)  # Grid size in cells
    
    grid_x = grid_index % grid_size
    grid_y = grid_index // grid_size
    
    # Convert to world coordinates (meters)
    world_x = (grid_x * config.grid_resolution_km * 1000) - (config.map_size[0] // 2)
    world_y = (grid_y * config.grid_resolution_km * 1000) - (config.map_size[1] // 2)
    
    return world_x, world_y


def position_in_bounds(x: float, y: float, config: Config) -> bool:
    """Check if position is within map boundaries."""
    half_map = config.map_size[0] // 2
    return abs(x) <= half_map and abs(y) <= half_map


def calculate_max_grid_positions(config: Config) -> int:
    """Calculate maximum number of grid positions for the map."""
    grid_size = int((config.map_size[0] / 1000) / config.grid_resolution_km)
    return grid_size * grid_size
