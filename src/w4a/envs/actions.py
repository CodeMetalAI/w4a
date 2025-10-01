"""
Action System

This module handles the execution and validation of hierarchical actions in the
TridentIslandEnv. It provides a comprehensive action space that covers tactical
military operations including movement, engagement, sensing, and support actions.

The action system validates all actions against current game state and entity
capabilities before execution, ensuring that only valid tactical decisions
are processed by the simulation engine.

Action types supported:
- Movement: CAP routes and positioning
- Engagement: Target selection and weapon employment  
- Sensing: Radar direction and stealth modes
- Additional: Refueling and return-to-base operations
- Objectives: Capture and hold operations
"""

from typing import Dict, List, Tuple
from ..config import Config

import math

from SimulationInterface import (
    PlayerEventCommit, NonCombatManouverQueue, MoveManouver, CAPManouver, RTBManouver,
    SetRadarFocus, ClearRadarFocus, SetRadarEnabled, CaptureFlag, Refuel,
    RefuelComponent,
    Vector3, Formation, ControllableEntity, EntityDomain, Faction,
)

FACTION_FLAG_IDS = { Faction.LEGACY: 2, Faction.DYNASTY: 1, Faction.NEUTRAL: 3 } #Ideally, we would not hardcode this
CENTER_ISLAND_FLAG_ID = FACTION_FLAG_IDS[Faction.NEUTRAL]

def execute_action(action: Dict, entities: Dict, target_groups: Dict, config: Config) -> List:
    """Execute a hierarchical action and return corresponding player events.
    
    Validates the action against current game state and entity capabilities,
    then converts it into appropriate player events for execution.
    
    Args:
        action: Hierarchical action dictionary from agent
        entities: Dict of entity_id -> entity objects
        target_groups: Dict of target_group_id -> target_group objects  
        config: Environment configuration
        
    Returns:
        List of PlayerEvent objects to submit, empty if action invalid
    """
    action_type = action["action_type"]
    entity_id = action["entity_id"]

    player_event = []
    
    if not is_valid_action(action, entities, target_groups, config):
        return player_event
    
    if action_type == 0:  # No-op
        return player_event
    elif action_type == 1:  # Move
        event = execute_move_action(entity_id, action, entities, config)
        player_event = event
    elif action_type == 2:  # Engage
        event = execute_engage_action(entity_id, action, entities, target_groups)
        player_event = event
    elif action_type == 3:  # Stealth
        event = execute_stealth_action(entity_id, action, entities)
        player_event = event
    elif action_type == 4:  # Sensing Direction
        event = execute_set_radar_focus_action(entity_id, action, entities, config)
        player_event = event
    elif action_type == 5:  # Capture
        event = execute_capture_action(entity_id, action, entities)
        player_event = event
    elif action_type == 6:  # RTB
        event = execute_rtb_action(entity_id, action, entities)
        player_event = event
    elif action_type == 7:  # Refuel
        event = execute_refuel_action(entity_id, action, entities)
        player_event = event
    
    return player_event


def is_valid_action(action: Dict, entities: Dict, target_groups: Dict, config: Config) -> bool:
    """Validate action against simulation constraints and current game state.
    
    Performs comprehensive validation including entity existence, capability checks,
    parameter bounds, and tactical feasibility before allowing action execution.
    
    Args:
        action: Action dictionary from agent
        entities: Dict of entity_id -> entity objects
        target_groups: Dict of target_group_id -> target_group objects
        config: Environment configuration
        
    Returns:
        True if action is valid and can be executed, False otherwise
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
    elif action_type == 3:  # Stealth
        return validate_stealth_action(action, entities)
    elif action_type == 4:  # Sensing Position
        return validate_sensing_position_action(action, entities, config)
    elif action_type == 5:  # Capture
        return validate_capture_action(action, entities)
    elif action_type == 6:  # RTB
        return validate_rtb_action(action, entities)
    elif action_type == 7:  # Refuel
        return validate_refuel_action(action, entities)
    
    return False


def execute_move_action(entity_id: int, action: Dict, entities: Dict, config: Config):
    """Execute move action by creating a CAP (Combat Air Patrol) maneuver.
    
    Args:
        entity_id: ID of entity to move
        action: Action dictionary with movement parameters
        entities: Dict of entity objects
        config: Environment configuration
        
    Returns:
        PlayerEvent for CAP maneuver
    """
    center_x, center_y = grid_to_position(action["move_center_grid"], config)

    # Convert discrete action values to actual patrol axis lengths
    short_axis_km = config.min_patrol_axis_km + (action["move_short_axis_km"] * config.patrol_axis_increment_km)
    long_axis_km = config.min_patrol_axis_km + (action["move_long_axis_km"] * config.patrol_axis_increment_km)
    
    # Convert to meters
    short_axis_m = short_axis_km * 1000
    long_axis_m = long_axis_km * 1000
    
    axis_angle = math.radians(action["move_axis_angle"] * config.angle_resolution_degrees)
    
    entity = entities[entity_id]

    center = Vector3(center_x, center_y, entity.pos.z)
    axis = Vector3(math.cos(axis_angle), math.sin(axis_angle), 0)

    event = NonCombatManouverQueue.create(entity.pos, lambda: CAPManouver.create_race_track(center, short_axis_m, long_axis_m, axis, 32))
    event.entity = entity

    return event



def execute_engage_action(entity_id: int, action: Dict, entities: Dict, target_groups: Dict):
    """Execute engage action by creating a combat commit event.
    
    Args:
        entity_id: ID of entity to engage
        action: Action dictionary with engagement parameters
        entities: Dict of entity objects
        target_groups: Dict of target group objects
        
    Returns:
        PlayerEventCommit for combat engagement
    """
    target_group_id = action["target_group_id"]
    weapon_selection = action["weapon_selection"]
    weapon_usage = action["weapon_usage"]
    weapon_engagement = action["weapon_engagement"]
    
    entity = entities[entity_id]
    target_group = target_groups[target_group_id]
    
    # Get weapons compatible with target group
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
    """Execute radar focus action to direct sensing at specific location.
    
    Args:
        entity_id: ID of entity with radar
        action: Action dictionary with sensing parameters
        entities: Dict of entity objects
        config: Environment configuration
        
    Returns:
        PlayerEvent for radar focus or clear focus
    """

    max_grid_positions = calculate_max_grid_positions(config)
    if action["sensing_position_grid"] == max_grid_positions: # Default sensing (forward)
        entity = entities[entity_id]
        
        event = ClearRadarFocus()
        event.entity = entity

        return event

    sense_x, sense_y = grid_to_position(action["sensing_position_grid"], config)
    
    entity = entities[entity_id]
    
    event = SetRadarFocus()
    event.entity = entity
    event.position = Vector3(sense_x, sense_y, entity.pos.z)
    
    return event

def execute_stealth_action(entity_id: int, action: Dict, entities: Dict, config: Config):
    """Execute stealth action by setting radar emission strength.
    
    Args:
        entity_id: ID of entity to set stealth mode
        action: Action dictionary with stealth parameters
        entities: Dict of entity objects
        config: Environment configuration
        
    Returns:
        PlayerEvent for radar strength setting
    """
    stealth_enabled = action["stealth_enabled"]
    
    entity = entities[entity_id]
    
    event = SetRadarEnabled()
    event.entity = entity
    event.enabled = stealth_enabled
    
    return event

def execute_capture_action(entity_id: int, action: Dict, entities: Dict):
    # """Execute land action - land at nearest friendly airbase"""
    entity = entities[entity_id]
    flag = entities[FACTION_FLAG_IDS[Faction.NEUTRAL]]

    event = CaptureFlag()
    event.entity = entity
    event.flag = flag

    return event

def execute_rtb_action(entity_id: int, action: Dict, entities: Dict):
    """Execute RTB action - return to base"""
    entity = entities[entity_id]
    flag = entities[FACTION_FLAG_IDS[entity.faction]]

    event = RTBManouver()
    event.entity = entity
    event.flag = flag
    
    return event

def execute_refuel_action(entity_id: int, action: Dict, entities: Dict):
    """Execute refuel action to refuel from another entity.
    
    Args:
        entity_id: ID of entity that needs fuel
        action: Action dictionary with refuel parameters
        entities: Dict of entity objects
        
    Returns:
        PlayerEvent for refueling operation
    """
    refuel_target_id = action["refuel_target_id"]
    
    entity = entities[entity_id]
    refuel_target = entities[refuel_target_id]

    event = Refuel()
    event.component = entity.find_component_by_class(RefuelComponent)
    event.entity = entity
    event.refueling_entity = refuel_target
    
    return event

def validate_entity(action: Dict, entities: Dict, config: Config) -> bool:
    """Validate that the target entity exists and is controllable.
    
    Args:
        action: Action dictionary containing entity_id
        entities: Dict of entity objects
        config: Environment configuration
        
    Returns:
        True if entity is valid and controllable
    """
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
    """Validate move action parameters and entity capabilities.
    
    Args:
        action: Action dictionary with movement parameters
        entities: Dict of entity objects
        config: Environment configuration
        
    Returns:
        True if move action is valid
    """
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
    """Validate engage action parameters and target availability.
    
    Args:
        action: Action dictionary with engagement parameters
        entities: Dict of entity objects
        target_groups: Dict of target group objects
        
    Returns:
        True if engage action is valid
    """
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


def validate_stealth_action(action: Dict, entities: Dict) -> bool:
    """Validate stealth action parameters and entity radar capability.
    
    Args:
        action: Action dictionary with stealth parameters
        entities: Dict of entity objects
        
    Returns:
        True if stealth action is valid
    """
    entity_id = action["entity_id"]
    entity = entities[entity_id]
    
    if not entity.has_radar:
        return False
    
    return True


def validate_sensing_position_action(action: Dict, entities: Dict, config: Config) -> bool:
    """Validate sensing position action parameters and radar capability.
    
    Args:
        action: Action dictionary with sensing parameters
        entities: Dict of entity objects
        config: Environment configuration
        
    Returns:
        True if sensing action is valid
    """
    entity_id = action["entity_id"]
    sensing_position_grid = action["sensing_position_grid"]
    
    entity = entities[entity_id]
    
    # Check grid position is within map bounds
    max_grid_positions = calculate_max_grid_positions(config)
    if sensing_position_grid > max_grid_positions:
        return False
    
    if sensing_position_grid == max_grid_positions:
        return True  # Default sensing (forward)

    # Convert to world coordinates and check bounds
    world_x, world_y = grid_to_position(sensing_position_grid, config)
    if not position_in_bounds(world_x, world_y, config):
        return False
    
    # Check entity has radar/sensors
    if not entity.has_radar:
        return False
    
    return True

def validate_capture_action(action: Dict, entities: Dict) -> bool:
    """Validate capture action parameters and entity capability.
    
    Args:
        action: Action dictionary with capture parameters
        entities: Dict of entity objects
        
    Returns:
        True if capture action is valid
    """
    entity_id = action["entity_id"]
    entity = entities[entity_id]

    flag_id = FACTION_FLAG_IDS[Faction.NEUTRAL]
    flag = entities[flag_id]
    
    # Check entity is aircraft
    if entity.domain != EntityDomain.AIR:
        return False

    # Check if flag is neutral
    if flag.faction != Faction.NEUTRAL:
        return False
    
    # Check entity can capture
    if not entity.can_capture:
        return False
    
    return True


def validate_rtb_action(action: Dict, entities: Dict) -> bool:
    """Validate return-to-base action parameters.
    
    Args:
        action: Action dictionary with RTB parameters
        entities: Dict of entity objects
        
    Returns:
        True if RTB action is valid
    """
    entity_id = action["entity_id"]
    entity = entities[entity_id]

    flag_id = FACTION_FLAG_IDS[entity.faction]
    flag = entities[flag_id]
    
    # Check entity is aircraft
    if entity.domain != EntityDomain.AIR:
        return False

    # Check if flag faction is the same as the aircraft
    if flag.faction != entity.faction:
        return False
    
    return True


def validate_refuel_action(action: Dict, entities: Dict) -> bool:
    """Validate refuel action parameters and entity capabilities.
    
    Args:
        action: Action dictionary with refuel parameters
        entities: Dict of entity objects
        
    Returns:
        True if refuel action is valid
    """
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

    if not entity.can_refuel:
        return False    
    
    # Check refuel target can provide fuel
    if not refuel_target.can_refuel_others:
        return False
    
    return True

def select_weapons_from_available(available_weapons: Dict, selection_index: int) -> Dict:
    """Select specific weapons using combinatorial selection from available options.
    
    Converts the agent's discrete weapon selection choice into a specific
    subset of available weapons using binary combination encoding.
    
    Args:
        available_weapons: Dict from entity.select_weapons() containing compatible weapons
        selection_index: Agent's weapon combination choice (0 to max_weapon_combinations-1)
        
    Returns:
        Dict containing selected weapons for engagement
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
    """Get valid weapon combination indices for current available weapons.
    
    Generates all possible weapon combinations that can be selected from
    the currently available weapons for this entity and target.
    
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
    world_x = (grid_x * config.grid_resolution_km * 1000) - (config.map_size_km[0] // 2)
    world_y = (grid_y * config.grid_resolution_km * 1000) - (config.map_size_km[1] // 2)
    
    return world_x, world_y


def position_in_bounds(x: float, y: float, config: Config) -> bool:
    """Check if world position is within map boundaries.
    
    Args:
        x, y: World coordinates in meters
        config: Environment configuration with map size
        
    Returns:
        True if position is within map bounds
    """
    half_map = config.map_size_km[0] * 1000 // 2
    return abs(x) <= half_map and abs(y) <= half_map


def calculate_max_grid_positions(config: Config) -> int:
    """Calculate maximum number of grid positions for the map.
    
    Args:
        config: Environment configuration with map and grid parameters
        
    Returns:
        Total number of discrete grid positions available
    """
    grid_size = int((config.map_size_km[0]) / config.grid_resolution_km)
    return grid_size * grid_size
