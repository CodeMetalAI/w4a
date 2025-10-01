"""
Observation System

This module provides comprehensive observation encoding for tactical simulation environments.
It transforms raw simulation state into normalized feature vectors suitable for machine learning.

The observation system encodes three main categories:
- Global features: mission state, time, objectives, and overall tactical situation
- Friendly features: our units' capabilities, positions, and status
- Enemy features: detected enemy units and threat assessments based on sensing tiers

The current implementation focuses on global features with structured placeholders
for entity-level features that can be expanded as needed.
"""

from typing import Any

import numpy as np
from gymnasium import spaces


def build_observation_space(config) -> spaces.Box:
    """Build the complete observation space for the environment.

    Creates a normalized Box space containing all observation features.
    Currently includes 7 global features with placeholders for entity features.
    
    Returns:
        Box space with normalized values in [0, 1]
    """
    # Current: 7 global features
    # NOTE (ID-indexed layout):
    # When adding per-entity or per-target-group features, index rows by stable IDs
    # assigned in the environment, not by iteration order. That is:
    # - Entities: shape (config.max_entities, features_per_entity), row i corresponds
    #             to entity_id i. Zero rows for IDs that are not assigned.
    # - Target groups: shape (config.max_target_groups, features_per_group), row j
    #                  corresponds to target_group_id j. Zero rows when not visible.
    # This keeps observation indices aligned with action/mask IDs across steps.
    
    total_features = 7  # globals only for now
    low = np.zeros((total_features,), dtype=np.float32)
    high = np.ones((total_features,), dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def compute_observation(env: Any) -> np.ndarray:
    """Compute the complete observation vector from current environment state.
    
    Extracts and normalizes all relevant features from the simulation state,
    combining global mission status with entity-specific information.
    
    Args:
        env: Environment instance containing simulation state
        
    Returns:
        Normalized observation vector with values in [0, 1]
    """
    # Compute feature groups
    global_features = _compute_global_features(env)
    friendly_features = _compute_friendly_features(env)
    enemy_features = _compute_enemy_features(env)
    
    # Concatenate all features
    # Currently only globals are implemented. Future per-entity/group features should
    # be ID-indexed arrays (see build_observation_space note above).
    obs = np.concatenate([
        global_features,
        # friendly_features,  # TODO: Uncomment when implemented
        # enemy_features,     # TODO: Uncomment when implemented
    ], dtype=np.float32)
    
    return obs


def _compute_global_features(env: Any) -> np.ndarray:
    """Compute global mission state features.
    
    Extracts high-level mission status that affects overall tactical decisions.
    All features are normalized to [0,1] for consistent learning.
    
    Features:
    - time_remaining_norm: Mission time remaining [0,1]
    - friendly_kills_norm: Our casualties normalized by max entities
    - enemy_kills_norm: Enemy casualties normalized by max entities
    - awacs_alive_flag: AWACS availability [0,1]
    - capture_progress_norm: Objective capture progress [0,1]
    - island_contested_flag: Area contestation status [0,1]
    - capture_possible_flag: Capture capability status [0,1]
    
    Returns:
        Array of shape (7,) with normalized values in [0,1]
    """
    # Time remaining normalized (in seconds)
    time_remaining = max(0, env.config.max_game_time - env.time_elapsed)
    time_remaining_norm = float(time_remaining) / float(env.config.max_game_time)

    # Kill tallies normalized by max_entities
    denom = float(max(1, env.config.max_entities))
    friendly_kills = float(len(env.friendly_kills))
    enemy_kills = float(len(env.enemy_kills))
    friendly_kills_norm = np.clip(friendly_kills / denom, 0.0, 1.0)
    enemy_kills_norm = np.clip(enemy_kills / denom, 0.0, 1.0)

    # AWACS alive flag
    awacs_alive_flag = 1.0 if _awacs_alive(env) else 0.0

    # Capture progress normalized by required capture time
    required_capture_time = env.config.capture_required_seconds
    capture_timer_progress = float(env.capture_timer_progress)
    if required_capture_time <= 0.0:
        capture_progress_norm = 0.0  # 0.0 = "no capture mechanic"
    else:
        capture_progress_norm = float(np.clip(capture_timer_progress / required_capture_time, 0.0, 1.0))

    # Island contested and capture possible flags
    island_contested_flag = 1.0 if env.island_contested else 0.0
    capture_possible_flag = 1.0 if env.capture_possible else 0.0

    return np.array([
        time_remaining_norm,
        friendly_kills_norm,
        enemy_kills_norm,
        awacs_alive_flag,
        capture_progress_norm,
        island_contested_flag,
        capture_possible_flag,
    ], dtype=np.float32)


def _get_friendly_entities(env: Any) -> list:
    """Get list of friendly entities (helper for placeholder code).
    
    Returns:
        List of entities belonging to our faction
    """
    return [entity for entity in env.entities.values() if entity.faction.value == env.config.our_faction]


def _compute_friendly_features(env: Any) -> np.ndarray:
    """Compute friendly entity features for observation encoding.
    
    Extracts detailed information about our units including capabilities,
    status, and tactical situation. Features are organized by category
    for each entity up to the maximum entity limit.
    
    Feature categories per entity (24 features total):
    - Identity & Intent: can_engage_now, can_sense_now, can_refuel
    - Kinematics: position, heading, speed
    - Egocentric: distance/bearing to island
    - Status: health, radar state, fuel
    - Weapons: weapon types, ammo counts  
    - Engagement: current engagement state, target info
    
    Returns:
        Array of shape (max_entities * 24,) with zero padding for unused slots
    """
    # TODO: Implement this as an ID-indexed array sized to max_entities
    # Build an ID-indexed array: row i corresponds to entity_id i. Zero rows for unused IDs.
    friendly_entity_features = np.zeros((int(env.config.max_entities), 24), dtype=np.float32)

    for entity_id, entity in env.entities.items():
        # Include only friendly faction entities
        if entity.faction.value != env.config.our_faction:
            continue

        # Compute each feature category
        identity_features = compute_friendly_identity_features(entity)
        kinematic_features = compute_friendly_kinematic_features(env, entity)
        egocentric_features = compute_friendly_egocentric_features(env, entity)
        status_features = compute_friendly_status_features(entity)
        weapon_features = compute_friendly_weapon_features(entity)
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

    # Convert (max_entities, 24) -> (max_entities * 24,)
    return friendly_entity_features.flatten()


def compute_friendly_identity_features(entity: Any) -> np.ndarray:
    """Compute identity and intent features for a friendly entity.
    
    Encodes the entity's basic capabilities and role in the mission.
    
    Features: can_engage_now, can_sense_now, can_refuel (3 features)
    
    Returns:
        Array of shape (3,) with binary capability flags
    """
    # Identity & Intent (3 features)
    # TODO: Implement these features

    can_engage = 1.0 if entity.can_engage else 0.0
    can_sense = 1.0 if entity.has_radar else 0.0 
    can_refuel = 1.0 if entity.can_refuel else 0.0
    return np.array([can_engage, can_sense, can_refuel], dtype=np.float32)
    

def compute_friendly_kinematic_features(env: Any, entity: Any) -> np.ndarray:
    """Compute kinematic state features for a friendly entity.
    
    Encodes position, heading, and speed in a normalized format suitable
    for neural networks. Position is discretized to match action space.
    
    Features: grid_x_norm, grid_y_norm, heading_sin, heading_cos, speed_norm (5 features)
    
    Returns:
        Array of shape (5,) with normalized kinematic state
    """
    # Kinematics (5 features)
    # Convert position to grid coordinates (matching action space)
    grid_index = position_to_grid(entity.position.x, entity.position.y, env.config)
    grid_size = env.grid_size
    grid_x = grid_index % grid_size
    grid_y = grid_index // grid_size
    
    # Normalize grid coordinates [0, grid_size-1] -> [0, 1]
    grid_x_norm = grid_x / (grid_size - 1) if grid_size > 1 else 0.0
    grid_y_norm = grid_y / (grid_size - 1) if grid_size > 1 else 0.0
    
    # Heading as sin/cos to avoid wraparound issues
    heading_sin = (np.sin(entity.heading) + 1.0) / 2.0  # [-1,1] -> [0,1]
    heading_cos = (np.cos(entity.heading) + 1.0) / 2.0  # [-1,1] -> [0,1]
    
    # Speed normalized by max speed
    speed_norm = entity.speed / entity.max_speed if entity.max_speed > 0 else 0.0
    
    return np.array([grid_x_norm, grid_y_norm, heading_sin, heading_cos, speed_norm], dtype=np.float32)
    


def compute_friendly_egocentric_features(env: Any, entity: Any) -> np.ndarray:
    """Compute egocentric spatial relationships for a friendly entity.
    
    Encodes the entity's spatial relationship to key mission objectives,
    providing context for tactical decision making.
    
    Features: island_range_norm, island_bearing_norm (2 features)
    
    Returns:
        Array of shape (2,) with normalized spatial relationships
    """
    # Egocentric (2 features)
    # Calculate map diagonal for normalization (max possible distance)
    # TODO: Does this make sense?
    map_width, map_height = env.config.map_size
    map_diagonal = np.sqrt(map_width * map_width + map_height * map_height)
    
    island_range = distance_to_island(entity.position)
    island_range_norm = island_range / map_diagonal
    
    island_bearing = bearing_to_island(entity.position)
    island_bearing_norm = island_bearing / 360.0
    
    return np.array([island_range_norm, island_bearing_norm], dtype=np.float32)
    


def compute_friendly_status_features(entity: Any) -> np.ndarray:
    """Compute entity status features for a friendly entity.
    
    Encodes the current operational status and configuration of the entity.
    
    Features: health_ok, radar_on, radar_direction_norm, fuel_remaining_norm (4 features)
    
    Returns:
        Array of shape (4,) with normalized status indicators
    """
    health_ok = 1.0 if entity.is_alive else 0.0
    radar_on = 1.0 if entity.radar_active else 0.0
    # TODO: Instead of radar dir, using radar point (as used in action space)
    radar_dir_norm = entity.radar_direction / 360.0
    fuel_norm = entity.fuel_remaining / entity.max_fuel
    return np.array([health_ok, radar_on, radar_dir_norm, fuel_norm], dtype=np.float32)
    

def compute_friendly_weapon_features(entity: Any) -> np.ndarray:
    """Compute weapon system features for a friendly entity.
    
    Encodes the entity's weapon capabilities and ammunition status.
    
    Features: has_air_weapons, has_surface_weapons, air_ammo_norm, surface_ammo_norm (4 features)
    
    Returns:
        Array of shape (4,) with weapon capability and ammo status
    """
    # Weapons (4 features)
    # TODO: What does weapon information look like for an entity? 
    # TODO: Is weapon capability dependent on the target group being engaged?
    # TODO: Should we encode weapon range information?
    # TODO: How do we access entity weapon systems - entity.weapons list?
    
    # Placeholder logic for testing:
    # Check if entity has weapons that can engage air targets
    # has_air_weapons = 1.0 if any(weapon.can_engage_air for weapon in entity.weapons) else 0.0
    
    # Check if entity has weapons that can engage surface targets  
    # has_surface_weapons = 1.0 if any(weapon.can_engage_surface for weapon in entity.weapons) else 0.0
    
    # Get ammunition status for air-to-air weapons
    # air_weapons = [w for w in entity.weapons if w.can_engage_air]
    # total_air_ammo = sum(w.current_ammo for w in air_weapons)
    # max_air_ammo = sum(w.max_ammo for w in air_weapons) 
    # air_ammo_norm = total_air_ammo / max_air_ammo if max_air_ammo > 0 else 0.0
    
    # Get ammunition status for air-to-surface weapons
    # surface_weapons = [w for w in entity.weapons if w.can_engage_surface]
    # total_surface_ammo = sum(w.current_ammo for w in surface_weapons)
    # max_surface_ammo = sum(w.max_ammo for w in surface_weapons)
    # surface_ammo_norm = total_surface_ammo / max_surface_ammo if max_surface_ammo > 0 else 0.0
    
    # Alternative: Maybe weapons are target-specific and we need to check against target groups?
    # available_weapons = entity.select_weapons(target_group, is_defensive=False)
    # has_weapons_for_targets = 1.0 if len(available_weapons) > 0 else 0.0
    
    # return np.array([has_air_weapons, has_surface_weapons, air_ammo_norm, surface_ammo_norm], dtype=np.float32)
    
    return np.zeros(4, dtype=np.float32)  # TODO: Implement when weapon system interface is clear


def compute_friendly_engagement_features(env: Any, entity: Any) -> np.ndarray:
    """Compute current engagement status features for a friendly entity.
    
    Encodes the entity's current combat engagement state and target information.
    
    Features: currently_engaging, target_group_id_norm, target_range_norm, 
             target_bearing_norm, target_closure_norm, time_since_shot_norm (6 features)
    
    Returns:
        Array of shape (6,) with engagement status and target information
    """
    # Engagement (6 features)
    # Check if entity is currently engaging a target group

    currently_engaging = 1.0 if hasattr(entity, 'target_group') and entity.target_group else 0.0# TODO: Fitler checks for entities that can commit
    
    # Get target group ID (matches action space target_group_id)
    # target_group_id_norm = entity.target_group.id / (env.config.max_target_groups - 1) if entity.target_group else -1.0
    
    # Get range to target group (center or nearest entity in group)
    # TODO: Do we have this information?
    # target_range_norm = entity.target_group_range / max_engagement_range if entity.target_group else 0.0
    
    # Get bearing to target group
    # TODO: Do we have this information?
    # target_bearing_norm = entity.target_group_bearing / 360.0 if entity.target_group else 0.0

    # Time since last weapon firing
    # time_since_shot_norm = entity.time_since_shot / max_reload_time if hasattr(entity, 'time_since_shot') else 1.0
    
    # Time on target information
    # TODO: We probably need to keep track of this?
    # time_on_target = entity.time_tracking_current_target if entity.target_group else 0.0
    # time_on_target_norm = np.clip(time_on_target / 60.0, 0.0, 1.0)  # Normalize to 1 minute max
    
    # Alternatively, time to firing range information
    # time_to_firing_range = entity.time_to_weapon_range if entity.target_group else 0.0
    # time_to_firing_range_norm = np.clip(time_to_firing_range / 300.0, 0.0, 1.0)  # 5 minutes max

    # Weapons usage mode (one-hot encoding: tight/selective/free)
    # weapons_mode = entity.weapons_usage_mode if hasattr(entity, 'weapons_usage_mode') else 'selective'
    # weapons_tight = 1.0 if weapons_mode == 'tight' else 0.0
    # weapons_selective = 1.0 if weapons_mode == 'selective' else 0.0  
    # weapons_free = 1.0 if weapons_mode == 'free' else 0.0
    
    # Shots fired this engagement/commit
    # shots_fired_this_commit = entity.shots_fired_current_engagement if hasattr(entity, 'shots_fired_current_engagement') else 0
    # shots_fired_norm = np.clip(shots_fired_this_commit / 10.0, 0.0, 1.0)  # Normalize to max 10 shots
    
    # Extended feature set (10 features total):
    # return np.array([
    #     currently_engaging, target_group_id_norm, target_range_norm, target_bearing_norm,  # Basic (4)
    #     time_since_shot_norm, time_on_target_norm,  # Timing (2)
    #     weapons_tight, weapons_selective, weapons_free,  # Weapons mode (3)
    #     shots_fired_norm  # Activity (1)
    # ], dtype=np.float32)
    
    return np.zeros(6, dtype=np.float32)  # TODO: Implement when target group is clear


def _compute_enemy_features(env: Any) -> np.ndarray:
    """Compute enemy entity features based on tiered sensing information.
    
    Implements a realistic intelligence gathering system where information quality
    depends on sensor coverage and capabilities. Higher sensing tiers provide
    more detailed and accurate information about enemy forces.
    
    Tiered Sensing Model:
    - Tier 1: Domain detection (air/surface/land) - enables weapon selection
    - Tier 2: Individual unit identification and positions  
    - Tier 3: Detailed weapon loadout information
    
    Features per enemy target group (16 features total):
    - Detection: is_detected, sensing_tier, detection_confidence, time_since_contact_norm
    - Identity: domain_one_hot (air/surface/land), estimated_count, unit_type_known
    - Position: last_known_x_norm, last_known_y_norm
    - Kinematics: heading_sin, heading_cos, speed_norm (if Tier 2+)
    - Weapons: has_air_weapons, has_surface_weapons, weapon_count_est (if Tier 3)
    - Threat: threat_level_norm, is_engaging_us
    
    Returns:
        Array of shape (max_target_groups * 16,) with zero padding for undetected groups
    """
    # NOTE: Sensing information should be updated in env.step() before calling this function
    # TODO: Add sensing update logic in trident_island_env.step():
    #   1. Update sensor coverage based on friendly unit positions/radar status
    #   2. Determine sensing tier for each enemy group based on range/sensor capability
    #   3. Update enemy_sensing_data with latest information per tier
    
    enemy_group_features = np.zeros((env.config.max_target_groups, 16), dtype=np.float32)
    
    # Populate rows by stable target_group_id; array size stays fixed
    for group_id, enemy_group in env.target_groups.items():
        # TODO: Implement sensing data
        sensing_data = env.enemy_sensing_data.get(group_id, None)
        
        if not sensing_data or not sensing_data.is_detected:
            # No detection - all zeros (already initialized)
            continue

        # Detection features (4)
        is_detected = 1.0 # If its in the target groups, it is detected
        sensing_tier = float(sensing_data.tier) / 3.0  # Normalize tier 1-3 to [0.33, 1.0]
        detection_confidence = sensing_data.confidence  # [0,1]
        time_since_contact = (env.time_elapsed - sensing_data.last_contact_time) / 300.0  # 5min max
        time_since_contact_norm = np.clip(time_since_contact, 0.0, 1.0)
        
        # Identity features (3) - Available at Tier 1+
        domain_air = 1.0 if sensing_data.domain == 'AIR' else 0.0
        domain_surface = 1.0 if sensing_data.domain == 'SURFACE' else 0.0
        domain_land = 1.0 if sensing_data.domain == 'LAND' else 0.0
        estimated_count_norm = np.clip(sensing_data.estimated_unit_count / 10.0, 0.0, 1.0)  # Max 10 units
        
        # Position features (3) - Improved accuracy at Tier 2+
        # TODO: Should we implement uncertainty?
        if sensing_data.tier >= 2:
            # Tier 2+: Individual unit positions known
            center_x, center_y = sensing_data.get_group_center_position()
        else:
            # Tier 1: Approximate group position only
            center_x, center_y = sensing_data.approximate_position
        
        # Convert to grid coordinates
        grid_pos = position_to_grid(center_x, center_y, env.config)
        last_known_x_norm = float(grid_pos % env.grid_size) / (env.grid_size - 1)
        last_known_y_norm = float(grid_pos // env.grid_size) / (env.grid_size - 1)
        
        # Kinematic features (3) - Available at Tier 2+
        if sensing_data.tier >= 2:
            heading_sin = np.sin(np.radians(sensing_data.average_heading))
            heading_cos = np.cos(np.radians(sensing_data.average_heading))
            speed_norm = sensing_data.average_speed / env.config.max_speed
        else:
            # Tier 1: No kinematic information
            heading_sin = 0.0
            heading_cos = 0.0
            speed_norm = 0.0
        
        # Weapon features (3) - Available at Tier 3
        # TODO: What do these functions look like?
        if sensing_data.tier >= 3:
            has_air_weapons = 1.0 if sensing_data.weapon_capabilities.can_engage_air else 0.0
            has_surface_weapons = 1.0 if sensing_data.weapon_capabilities.can_engage_surface else 0.0
            weapon_count_norm = np.clip(sensing_data.estimated_weapon_count / 20.0, 0.0, 1.0)
        else:
            # Tier 1-2: Weapon capabilities unknown or estimated from domain
            # TODOL Can we assume? Should it just be 0?
            if sensing_data.tier >= 1:
                # Basic weapon estimation from domain knowledge
                has_air_weapons = 1.0 if sensing_data.domain in ['AIR', 'SURFACE'] else 0.0
                has_surface_weapons = 1.0 if sensing_data.domain in ['AIR', 'SURFACE'] else 0.0
                weapon_count_norm = 0.5  # Unknown, assume moderate
            else:
                has_air_weapons = 0.0
                has_surface_weapons = 0.0
                weapon_count_norm = 0.0
        
        # Threat assessment (1) - Improves with sensing tier
        # TODO: Do any tiers reveal this information? Should we aggregate threat level?
        base_threat = sensing_data.estimated_threat_level / 10.0  # Base threat 0-10
        tier_confidence = sensing_data.tier / 3.0  # Higher tier = more confident assessment
        threat_level_norm = base_threat * tier_confidence
        
        # Engagement status (1)
        # TODO: Do any tiers reveal this information?
        is_engaging_us = 1.0 if sensing_data.is_engaging_friendly_forces else 0.0
        
        features = np.array([
            is_detected, sensing_tier, detection_confidence, time_since_contact_norm,  # Detection (4)
            domain_air, domain_surface, domain_land, estimated_count_norm,  # Identity (4)
            last_known_x_norm, last_known_y_norm,  # Position (2)
            heading_sin, heading_cos, speed_norm,  # Kinematics (3)
            has_air_weapons, has_surface_weapons, weapon_count_norm,  # Weapons (3)
            threat_level_norm, is_engaging_us  # Threat (2)
        ], dtype=np.float32)  # Total: 16 features
        
        # Assign into the stable ID-indexed row; keep array size fixed
        if 0 <= group_id < int(env.config.max_target_groups):
            enemy_group_features[group_id] = features
    
    return enemy_group_features.flatten()  # Shape: (max_target_groups * 15,)


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
    grid_size = int((config.map_size[0] / 1000) / config.grid_resolution_km)
    
    # Adjust for map center offset
    adjusted_x = x + (config.map_size[0] // 2)
    adjusted_y = y + (config.map_size[1] // 2)
    
    # Convert to grid indices
    # TODO: 1000s? Is this correct?
    grid_x = int(adjusted_x / (config.grid_resolution_km * 1000))
    grid_y = int(adjusted_y / (config.grid_resolution_km * 1000))
    
    # Clamp to valid range
    grid_x = max(0, min(grid_x, grid_size - 1))
    grid_y = max(0, min(grid_y, grid_size - 1))
    
    return grid_y * grid_size + grid_x


def distance_to_island(position: Any) -> float:
    """Calculate distance from entity position to the mission objective.
    
    Args:
        position: Entity position object with x, y coordinates
        
    Returns:
        Distance in meters to island center
    """
    # TODO: Get actual island center coordinates from scenario
    island_center_x = 0.0  # Assuming island at map center for now
    island_center_y = 0.0
    
    dx = position.x - island_center_x
    dy = position.y - island_center_y
    return np.sqrt(dx * dx + dy * dy)


def bearing_to_island(position: Any) -> float:
    """Calculate bearing from entity position to the mission objective.
    
    Args:
        position: Entity position object with x, y coordinates
        
    Returns:
        Bearing in degrees [0, 360)
    """
    # TODO: Get actual island center coordinates from scenario
    island_center_x = 0.0  # Assuming island at map center for now
    island_center_y = 0.0
    
    dx = island_center_x - position.x
    dy = island_center_y - position.y
    
    bearing_rad = np.arctan2(dy, dx)
    bearing_deg = np.degrees(bearing_rad)
    
    # Normalize to [0, 360)
    if bearing_deg < 0:
        bearing_deg += 360.0
        
    return bearing_deg


def _awacs_alive(env: Any) -> bool:
    """Check if friendly AWACS or equivalent radar platform is operational.

    Scans for any friendly, alive entity with AWACS identification.
    
    Returns:
        True if at least one AWACS platform is alive and operational
        
    TODO: Determine correct way to identify AWACS entities.
    """
    our_faction = env.config.our_faction
    for entity in env.entities.values():
        if (entity.is_alive and 
                entity.Entity.Identity == "AWACS" and 
                entity.faction.value == our_faction):
            return True
    return False

    
