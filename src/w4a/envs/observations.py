"""
Observation System

This module provides comprehensive observation encoding for tactical simulation environments.
It transforms raw simulation state into normalized feature vectors suitable for machine learning.

The observation system encodes three main categories:
- Global features: mission state, time, objectives, and overall tactical situation
- Friendly features: our units' capabilities, positions, and status
- Enemy features: detected enemy units and threat assessments based on sensing tiers

Total observation size: 11 + (max_entities * 36) + (max_target_groups * 12)
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from w4a.envs.constants import CENTER_ISLAND_FLAG_ID
from w4a.envs.utils import calculate_max_grid_positions

from SimulationInterface import Faction, EntityDomain, ControllableEntityManouver

def build_observation_space(config) -> spaces.Box:
    """Build the complete observation space for the environment.

    Creates a normalized Box space containing all observation features.
    
    Observation structure:
    - Global features: 11 (mission state, objectives, time, flag faction)
    - Friendly entity features: max_entities * 36 (capabilities, kinematics, status, engagement)
    - Enemy target group features: max_target_groups * 12 (position, velocity, domain)
    
    NOTE (ID-indexed layout):
    - Entities: shape (config.max_entities, 36), row i corresponds to entity_id i
    - Target groups: shape (config.max_target_groups, 12), row j corresponds to target_group_id j
    - Zero rows for IDs that are not assigned/visible
    - This keeps observation indices aligned with action/mask IDs across steps
    
    Returns:
        Box space with normalized values in [0, 1]
    """
    global_features = 11
    friendly_features = config.max_entities * 36
    enemy_features = config.max_target_groups * 12
    
    total_features = global_features + friendly_features + enemy_features
    
    low = np.zeros((total_features,), dtype=np.float32)
    high = np.ones((total_features,), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def compute_observation(env: Any, agent: Any) -> np.ndarray:
    """
    Compute observation for a specific agent.
    
    Extracts and normalizes features from the agent's perspective, including:
    - Global mission state (shared by both agents)
    - Friendly entity features (our controllable units)
    - Enemy features (detected target groups)
    
    Args:
        env: Environment instance (for global info like time, flags)
        agent: CompetitionAgent instance to compute observation for
        
    Returns:
        Normalized observation vector with values in [0, 1]
        Shape: (11 + max_entities*36 + max_target_groups*12,)
    """

    # Compute global features (mission state, objectives, time)
    global_features = _compute_global_features(env, agent)
    
    # Compute friendly entity features (our units)
    friendly_features = _compute_friendly_features(env, agent)
    
    # Compute enemy features (detected target groups)
    enemy_features = _compute_enemy_features(env, agent)
        
    # Concatenate all features into single observation vector
    observation = np.concatenate([
        global_features,
        friendly_features,
        enemy_features
    ], dtype=np.float32)
    
    return observation


def _compute_global_features(env: Any, agent: Any) -> np.ndarray:
    """Compute global mission state features for a specific agent.
    
    Extracts high-level mission status from the agent's perspective
    All features are normalized to [0,1] for consistent learning.
    
    Features (11 total):
    - time_remaining_norm: Mission time remaining [0,1]
    - my_casualties_norm: Our casualties normalized by max entities
    - enemy_casualties_norm: Enemy casualties normalized by max entities
    - kill_ratio_norm: Our kill ratio (enemy kills / our casualties) normalized by threshold
    - capture_progress_norm: Our objective capture progress [0,1]
    - enemy_capture_progress_norm: Enemy objective capture progress [0,1]
    - capture_possible_flag: Our capture capability status [0,1]
    - flag_faction_norm: Center flag faction (0.0=neutral, 0.33=legacy, 0.66=dynasty)
    - enemy_capture_possible_flag: Enemy capture capability status [0,1]
    - island_center_x_norm: Center island X coordinate normalized
    - island_center_y_norm: Center island Y coordinate normalized
    
    Args:
        env: Environment instance
        agent: CompetitionAgent instance (to get faction-specific capture info)
    
    Returns:
        Array of shape (11,) with normalized values in [0,1]
    """

    
    # Time remaining normalized (in seconds)
    time_remaining = max(0, env.config.max_game_time - env.time_elapsed)
    time_remaining_norm = float(time_remaining) / float(env.config.max_game_time)

    # Kill tallies and ratio (cached per-faction in mission_metrics)
    my_faction = agent.faction
    enemy_faction = Faction.DYNASTY if my_faction == Faction.LEGACY else Faction.LEGACY
    
    my_casualties = float(env.casualties_by_faction[my_faction])
    enemy_casualties = float(env.kills_by_faction[my_faction])  # My kills = enemy casualties
    
    denom = float(max(1, env.config.max_entities))
    my_casualties_norm = my_casualties / denom
    enemy_casualties_norm = enemy_casualties / denom
    
    # Kill ratio normalized by threshold (for win condition awareness)
    kill_ratio = env.kill_ratio_by_faction[my_faction]
    threshold = env.config.kill_ratio_threshold
    kill_ratio_norm = kill_ratio / threshold

    # Capture progress normalized by required capture time (agent-specific)
    required_capture_time = env.config.capture_required_seconds
    my_capture_progress = float(env.capture_progress_by_faction[my_faction])
    enemy_capture_progress = float(env.capture_progress_by_faction[enemy_faction])
    
    if required_capture_time <= 0.0:
        capture_progress_norm = 0.0  # 0.0 = "no capture mechanic"
        enemy_capture_progress_norm = 0.0
    else:
        capture_progress_norm = float(my_capture_progress / required_capture_time)
        enemy_capture_progress_norm = float(enemy_capture_progress / required_capture_time)

    # Capture possible flags
    capture_possible_flag = 1.0 if env.capture_possible_by_faction[my_faction] else 0.0
    enemy_capture_possible_flag = 1.0 if env.capture_possible_by_faction[enemy_faction] else 0.0
    
    # Flag faction (neutral=0.0, legacy=0.33, dynasty=0.66, captured by me=1.0)
    # Encoded as: neutral -> 0.0, legacy -> 0.33, dynasty -> 0.66
    flag = env.flags[CENTER_ISLAND_FLAG_ID]
    flag_current_faction = flag.faction
    
    if flag_current_faction == Faction.NEUTRAL:
        flag_faction_norm = 0.0
    elif flag_current_faction == Faction.LEGACY:
        flag_faction_norm = 0.33
    elif flag_current_faction == Faction.DYNASTY:
        flag_faction_norm = 0.66

    # Center island coordinates (from neutral flag)
    island_center_x = flag.pos.x
    island_center_y = flag.pos.y

    
    # Normalize to [0, 1] based on map bounds
    # Map coordinates are centered at 0, so adjust for offset
    map_width, map_height = env.config.map_size_km
    island_center_x_norm = (island_center_x / 1000.0 + map_width / 2.0) / map_width
    island_center_y_norm = (island_center_y / 1000.0 + map_height / 2.0) / map_height

    return np.array([
        time_remaining_norm,
        my_casualties_norm,
        enemy_casualties_norm,
        kill_ratio_norm,
        capture_progress_norm,
        enemy_capture_progress_norm,
        capture_possible_flag,
        flag_faction_norm,
        enemy_capture_possible_flag,
        island_center_x_norm,
        island_center_y_norm,
    ], dtype=np.float32)


def _compute_friendly_features(env: Any, agent: Any) -> np.ndarray:
    """Compute friendly entity features for observation encoding.
    
    Extracts detailed information about our units including capabilities,
    status, and tactical situation. Features are organized by category
    for each entity up to the maximum entity limit.
    
    Feature categories per entity (36 features total):
    - Identity & Intent: can_engage, can_sense, can_refuel, can_capture, domain (air/surface/land) - 7 features
    - Kinematics: position (2), velocity (3), rotation quaternion (4) - 9 features
    - Egocentric: distance/bearing to island - 2 features
    - Status: health, radar state, fuel - 4 features
    - Weapons: weapon types, ammo counts - 3 features
    - Engagement: current engagement state, target info, weapons mode - 11 features
    
    Args:
        env: Environment instance
        agent: CompetitionAgent instance (to get controllable entities)
    
    Returns:
        Array of shape (max_entities * 36,) with zero padding for unused slots
    """
    # Build an ID-indexed array: row i corresponds to entity_id i. Zero rows for unused IDs.
    friendly_entity_features = np.zeros((int(env.config.max_entities), 36), dtype=np.float32)

    # Iterate through agent's controllable entities
    for entity_id, entity in agent._sim_agent.controllable_entities.items():

        # Compute each feature category
        identity_features = compute_friendly_identity_features(entity)
        kinematic_features = compute_friendly_kinematic_features(env, entity)
        egocentric_features = compute_friendly_egocentric_features(env, entity)
        status_features = compute_friendly_status_features(entity, env)
        weapon_features = compute_friendly_weapon_features(entity, env)
        engagement_features = compute_friendly_engagement_features(env, entity)

        # Concatenate all features for this entity
        features = np.concatenate([
            identity_features,
            kinematic_features,
            egocentric_features,
            status_features,
            weapon_features,
            engagement_features
        ])

        # Assign into the stable ID-indexed row
        if 0 <= entity_id < int(env.config.max_entities):
            friendly_entity_features[entity_id] = features

    # Convert (max_entities, 36) -> (max_entities * 36,)
    return friendly_entity_features.flatten()


def compute_friendly_identity_features(entity: Any) -> np.ndarray:
    """Compute identity and intent features for a friendly entity.
    
    Encodes the entity's basic capabilities and role in the mission.
    
    Features: can_engage, can_sense, can_refuel, can_capture, domain_air, domain_surface, domain_land (7 features)
    
    Returns:
        Array of shape (7,) with binary capability flags and domain one-hot encoding
    """

    # can_engage = 1.0 if hasattr(entity, 'can_engage') and entity.can_engage else 0.0 # TODO: Revisit this?
    can_engage = 1.0 if entity.has_radar else 0.0
    can_sense = 1.0 if entity.has_radar else 0.0 
    can_refuel = 1.0 if entity.can_refuel else 0.0
    can_capture = 1.0 if entity.can_capture else 0.0
    
    # Domain one-hot encoding
    domain_air = 1.0 if entity.domain == EntityDomain.AIR else 0.0
    domain_surface = 1.0 if entity.domain == EntityDomain.SURFACE else 0.0
    domain_land = 1.0 if entity.domain == EntityDomain.LAND else 0.0
    
    return np.array([can_engage, can_sense, can_refuel, can_capture, 
                     domain_air, domain_surface, domain_land], dtype=np.float32)
    

def compute_friendly_kinematic_features(env: Any, entity: Any) -> np.ndarray:
    """Compute kinematic state features for a friendly entity.
    
    Encodes position, velocity, and rotation in a normalized format suitable
    for neural networks. Position is discretized to match action space.
    
    Features: grid_x_norm, grid_y_norm, vel_x_norm, vel_y_norm, vel_z_norm,
              rot_x_norm, rot_y_norm, rot_z_norm, rot_w_norm (9 features)
    
    Returns:
        Array of shape (9,) with normalized kinematic state
    """
    # Position (2 features)
    # Convert position to grid coordinates (matching action space)
    grid_index = position_to_grid(entity.pos.x, entity.pos.y, env.config)
    grid_size = env.grid_size
    grid_x = grid_index % grid_size
    grid_y = grid_index // grid_size
    
    # Normalize grid coordinates [0, grid_size-1] -> [0, 1]
    grid_x_norm = grid_x / (grid_size - 1) if grid_size > 1 else 0.0
    grid_y_norm = grid_y / (grid_size - 1) if grid_size > 1 else 0.0
    
    # Velocity (3 features)
    # Normalize by max_velocity, convert from [-1, 1] to [0, 1]
    vel_x_norm = (entity.vel.x / env.max_velocity + 1.0) / 2.0
    vel_y_norm = (entity.vel.y / env.max_velocity + 1.0) / 2.0
    vel_z_norm = (entity.vel.z / env.max_velocity + 1.0) / 2.0
    
    # Rotation quaternion (4 features)
    # Clip quaternion components to [-1, 1], then convert to [0, 1]
    rot_x_norm = (entity.rot.x + 1.0) / 2.0
    rot_y_norm = (entity.rot.y + 1.0) / 2.0
    rot_z_norm = (entity.rot.z + 1.0) / 2.0
    rot_w_norm = (entity.rot.w + 1.0) / 2.0
    
    return np.array([grid_x_norm, grid_y_norm, 
                     vel_x_norm, vel_y_norm, vel_z_norm,
                     rot_x_norm, rot_y_norm, rot_z_norm, rot_w_norm], dtype=np.float32)
    


def compute_friendly_egocentric_features(env: Any, entity: Any) -> np.ndarray:
    """Compute egocentric spatial relationships for a friendly entity.
    
    Encodes the entity's spatial relationship to key mission objectives,
    providing context for tactical decision making.
    
    Features: island_range_norm, island_bearing_norm (2 features)
    
    Returns:
        Array of shape (2,) with normalized spatial relationships
    """
    # Egocentric (2 features)
    # Calculate map diagonal for normalization (max possible distance in km)
    map_width_km, map_height_km = env.config.map_size_km
    map_diagonal_km = np.sqrt(map_width_km * map_width_km + map_height_km * map_height_km)
    
    island_range = distance_to_island(env, entity.pos)  # in km
    island_range_norm = island_range / map_diagonal_km
    
    island_bearing = bearing_to_island(env, entity.pos)  # in degrees [0, 360)
    island_bearing_norm = island_bearing / 360.0
    
    return np.array([island_range_norm, island_bearing_norm], dtype=np.float32)
    


def compute_friendly_status_features(entity: Any, env: Any) -> np.ndarray:
    """Compute entity status features for a friendly entity.
    
    Encodes the current operational status and configuration of the entity.
    
    Features: health_ok, radar_on, radar_focus_grid_norm, fuel_remaining_norm (4 features)
    
    Args:
        entity: Entity to compute features for
        env: Environment instance (for config to normalize radar focus)
    
    Returns:
        Array of shape (4,) with normalized status indicators
    """
    health_ok = 1.0 if entity.is_alive else 0.0
    radar_on = 1.0 if hasattr(entity, 'radar_enabled') and entity.radar_enabled else 0.0
    
    # Radar focus position as grid index (1 feature) - matches action space representation
    if entity.has_radar_focus_position:
        # Convert radar focus position to grid index
        radar_focus_pos = entity.radar_focus_position
        radar_focus_grid_idx = position_to_grid(radar_focus_pos.x, radar_focus_pos.y, env.config)
        
        # Normalize by max grid positions
        max_grid = calculate_max_grid_positions(env.config)
        radar_focus_grid_norm = float(radar_focus_grid_idx) / float(max_grid) if max_grid > 0 else 0.0
    else:
        # No radar focus set - use 0 as default
        radar_focus_grid_norm = 0.0
    
    fuel_norm = entity.relative_fuel_left
    
    return np.array([health_ok, radar_on, radar_focus_grid_norm, fuel_norm], dtype=np.float32)
    

def compute_friendly_weapon_features(entity: Any, env: Any) -> np.ndarray:
    """Compute weapon system features for a friendly entity.
    
    Encodes the entity's weapon capabilities and ammunition status.
    
    Features: has_air_weapons, has_surface_weapons, ammo_norm (3 features)
    
    Args:
        entity: Entity to compute features for
        env: Environment instance (for config to get max_ammo)
    
    Returns:
        Array of shape (3,) with weapon capability and ammo status
    """

    # Check if entity can engage air targets (bitwise check on target_domains)
    has_air_weapons = 1.0 if (entity.target_domains & EntityDomain.AIR.value) != 0 else 0.0
    
    # Check if entity can engage surface targets (bitwise check on target_domains)
    has_surface_weapons = 1.0 if (entity.target_domains & EntityDomain.SURFACE.value) != 0 else 0.0
    
    total_ammo = float(entity.ammo) if entity.ammo is not None else 0.0
    ammo_norm = total_ammo / env.max_ammo
    
    return np.array([has_air_weapons, has_surface_weapons, ammo_norm], dtype=np.float32)


def compute_friendly_engagement_features(env: Any, entity: Any) -> np.ndarray:
    """Compute current engagement status features for a friendly entity.
    
    Encodes the entity's current combat engagement state and target information.
    
    Features: currently_engaging, shots_fired_this_commit, target_range_norm, target_bearing_norm,
             target_domain_air, target_domain_surface, target_domain_land, time_until_shoot_norm,
             weapons_tight, weapons_selective, weapons_free (11 features)
    
    Args:
        env: Environment instance (for map size to normalize range)
        entity: Entity to compute features for
    
    Returns:
        Array of shape (11,) with engagement status and target information
    """
    # Check if entity is currently engaging by examining current_manouver
    engaging = entity.current_manouver == ControllableEntityManouver.COMBAT

    target_group = entity.current_target_group if engaging else None
    currently_engaging = 1.0 if target_group is not None else 0.0
    
    # Shots fired this engagement/commit (only valid during an engagement)
    shots_fired_this_commit = 0
    if engaging:
        shots_fired_this_commit = entity.num_shots_fired
    
    if target_group is not None:
        # Calculate range to target group (positions in meters)
        dx = target_group.pos.x - entity.pos.x
        dy = target_group.pos.y - entity.pos.y
        target_range_m = np.sqrt(dx * dx + dy * dy)
        target_range_km = target_range_m / 1000.0  # Convert to km
        
        # Normalize by map diagonal (both in km)
        map_width_km, map_height_km = env.config.map_size_km
        map_diagonal_km = np.sqrt(map_width_km * map_width_km + map_height_km * map_height_km)
        target_range_norm = target_range_km / map_diagonal_km if map_diagonal_km > 0 else 0.0
        
        # Calculate bearing to target group
        bearing_rad = np.arctan2(dy, dx)
        bearing_deg = np.degrees(bearing_rad)
        if bearing_deg < 0:
            bearing_deg += 360.0
        target_bearing_norm = bearing_deg / 360.0
        
        # Target domain (one-hot encoding)
        target_domain_air = 1.0 if target_group.domain == EntityDomain.AIR else 0.0
        target_domain_surface = 1.0 if target_group.domain == EntityDomain.SURFACE else 0.0
        target_domain_land = 1.0 if target_group.domain == EntityDomain.LAND else 0.0
    else:
        # Not engaging - all features are 0
        target_range_norm = 0.0
        target_bearing_norm = 0.0
        target_domain_air = 0.0
        target_domain_surface = 0.0
        target_domain_land = 0.0
    
    # Time on target information
    # get_estimated_time_until_shoot() returns -1.0 when entity can't shoot (not ready/no target)
    # We map this to 0.0 (not ready) and clamp positive values to [0, 60] seconds
    time_until_shoot = entity.get_estimated_time_until_shoot()
    if time_until_shoot < 0:
        time_until_shoot_norm = 0.0  # Not ready to shoot
    else:
        time_until_shoot_norm = min(time_until_shoot / 60.0, 1.0)  # Normalize to max 60 seconds

    # Weapons usage mode (one-hot encoding: tight/selective/free)
    weapons_mode = entity.weapons_usage_mode
    weapons_tight = 1.0 if weapons_mode == 0 else 0.0 # tight
    weapons_selective = 1.0 if weapons_mode == 1 else 0.0  # selective
    weapons_free = 1.0 if weapons_mode == 2 else 0.0 # free

    return np.array([
        currently_engaging,
        shots_fired_this_commit,
        target_range_norm,
        target_bearing_norm,
        target_domain_air,
        target_domain_surface,
        target_domain_land,
        time_until_shoot_norm,
        weapons_tight,
        weapons_selective,
        weapons_free,
    ], dtype=np.float32)


def _compute_enemy_features(env: Any, agent: Any) -> np.ndarray:
    """Compute enemy entity features from detected target groups.
    
    Uses TargetGroup API provided by simulation engine. Detection and sensing
    tiers are handled automatically by the simulation - if a target group is
    in agent.target_groups, it has been detected.
    
    Features per enemy target group (12 features total):
    - Detection: is_detected (1.0 if in target_groups)
    - Position: pos_x_norm, pos_y_norm (grid coordinates from target_group.pos)
    - Velocity: vel_x_norm, vel_y_norm (velocity components normalized by typical max speed)
    - Domain: domain_air, domain_surface, domain_land (one-hot from target_group.domain)
    - Count: num_units_norm (from target_group.num_known_alive_units)
    - Egocentric: island_range_norm, island_bearing_norm (distance/bearing to objective)
    - Uncertainty: is_ghost (detection quality from target_group.is_ghost)
    
    Args:
        env: Environment instance
        agent: CompetitionAgent instance (to get detected target groups)
    
    Returns:
        Array of shape (max_target_groups * 12,) with zero padding for undetected groups
    """
    num_features = 12
    enemy_group_features = np.zeros((env.config.max_target_groups, num_features), dtype=np.float32)
    
    # Populate rows by stable target_group_id; array size stays fixed
    for group_id, target_group in agent._sim_agent.target_groups.items():
        # Detection (1 feature)
        # If it's in the target_groups dict, it's detected by the simulation
        is_detected = 1.0
        
        # Position (2 features)
        # Get position from TargetGroup.pos (Vector3)
        center_x = target_group.pos.x
        center_y = target_group.pos.y
        
        # Convert to grid coordinates
        grid_pos = position_to_grid(center_x, center_y, env.config)
        grid_size = env.grid_size
        pos_x_norm = float(grid_pos % grid_size) / max(1, grid_size - 1)
        pos_y_norm = float(grid_pos // grid_size) / max(1, grid_size - 1)
        
        # Speed (2 features)
        # Normalize by max_velocity, convert from [-1, 1] to [0, 1]
        speed_x_norm = (target_group.vel.x / env.max_velocity + 1.0) / 2.0
        speed_y_norm = (target_group.vel.y / env.max_velocity + 1.0) / 2.0
        
        # Domain (3 features)
        # One-hot encoding from TargetGroup.domain (EntityDomain enum)
        domain_air = 1.0 if target_group.domain == EntityDomain.AIR else 0.0
        domain_surface = 1.0 if target_group.domain == EntityDomain.SURFACE else 0.0
        domain_land = 1.0 if target_group.domain == EntityDomain.LAND else 0.0
        
        # Count (1 feature)
        # Number of known alive units from TargetGroup.num_known_alive_units
        num_units = float(target_group.num_known_alive_units)
        num_units_norm = num_units / env.max_entities
        
        # Egocentric (2 features)
        # Distance and bearing to center island objective (both in km)
        map_width_km, map_height_km = env.config.map_size_km
        map_diagonal_km = np.sqrt(map_width_km * map_width_km + map_height_km * map_height_km)
        
        enemy_island_range = distance_to_island(env, target_group.pos)  # in km
        enemy_island_range_norm = enemy_island_range / map_diagonal_km
        
        enemy_island_bearing = bearing_to_island(env, target_group.pos)
        enemy_island_bearing_norm = enemy_island_bearing / 360.0
        
        # Uncertainty (1 feature)
        # Detection quality from TargetGroup.is_ghost
        is_ghost = 1.0 if target_group.is_ghost else 0.0
        
        features = np.array([
            is_detected,                                          # Detection (1)
            pos_x_norm, pos_y_norm,                              # Position (2)
            speed_x_norm, speed_y_norm,                              # Velocity (2)
            domain_air, domain_surface, domain_land,             # Domain (3)
            num_units_norm,                                      # Count (1)
            enemy_island_range_norm, enemy_island_bearing_norm,  # Egocentric (2)
            is_ghost                                             # Uncertainty (1)
        ], dtype=np.float32)  # Total: 12 features
        
        # Assign into the stable ID-indexed row; keep array size fixed
        if 0 <= group_id < int(env.config.max_target_groups):
            enemy_group_features[group_id] = features
    
    return enemy_group_features.flatten()  # Shape: (max_target_groups * 12,)


def position_to_grid(x: float, y: float, config: Any) -> int:
    """Convert world position to grid index for discretized action space.
    
    Transforms continuous world coordinates into discrete grid indices
    that match the action space representation.
    
    Args:
        x, y: World coordinates in meters
        config: Environment configuration with grid parameters
        
    Returns:
        Grid index corresponding to the position
    """
    # Convert world coordinates to grid coordinates
    # map_size_km is in km, so grid_size = 25,000 km / 50 km = 500
    grid_size = int(config.map_size_km[0] / config.grid_resolution_km)
    
    # Adjust for map center offset (positions are in meters, so convert map_size_km to meters)
    adjusted_x = x + (config.map_size_km[0] * 1000 // 2)
    adjusted_y = y + (config.map_size_km[1] * 1000 // 2)
    
    # Convert to grid indices (grid_resolution_km * 1000 = grid cell size in meters)
    grid_x = int(adjusted_x / (config.grid_resolution_km * 1000))
    grid_y = int(adjusted_y / (config.grid_resolution_km * 1000))
    
    # Clamp to valid range
    grid_x = max(0, min(grid_x, grid_size - 1))
    grid_y = max(0, min(grid_y, grid_size - 1))
    
    return grid_y * grid_size + grid_x


def distance_to_island(env: Any, position: Any) -> float:
    """Calculate distance from entity position to the mission objective.
    
    Args:
        env: Environment instance (to get island position)
        position: Entity position object with x, y coordinates (in meters)
        
    Returns:
        Distance in kilometers to island center
    """
    # Positions are in meters, convert to km
    island_center_x_km = env.flags[CENTER_ISLAND_FLAG_ID].pos.x / 1000.0
    island_center_y_km = env.flags[CENTER_ISLAND_FLAG_ID].pos.y / 1000.0
    pos_x_km = position.x / 1000.0
    pos_y_km = position.y / 1000.0
    
    dx = pos_x_km - island_center_x_km
    dy = pos_y_km - island_center_y_km
    return np.sqrt(dx * dx + dy * dy)


def bearing_to_island(env: Any, position: Any) -> float:
    """Calculate bearing from entity position to the mission objective.
    
    Args:
        env: Environment instance (to get island position)
        position: Entity position object with x, y coordinates (in meters)
        
    Returns:
        Bearing in degrees [0, 360)
    """
    # Positions are in meters, but bearing calculation doesn't depend on units
    island_center_x = env.flags[CENTER_ISLAND_FLAG_ID].pos.x
    island_center_y = env.flags[CENTER_ISLAND_FLAG_ID].pos.y
    
    dx = island_center_x - position.x
    dy = island_center_y - position.y
    
    bearing_rad = np.arctan2(dy, dx)
    bearing_deg = np.degrees(bearing_rad)
    
    # Normalize to [0, 360)
    if bearing_deg < 0:
        bearing_deg += 360.0
        
    return bearing_deg