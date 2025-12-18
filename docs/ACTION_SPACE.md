# Action Space Documentation

## Overview

The W4A environment uses a **hierarchical action space** where agents select an action type and provide parameters for that action. All actions are represented as a dictionary with multiple fields.

## Action Dictionary Structure

```python
action = {
    'action_type': int,              # Primary action selector (0-9)
    'entity_id': int,                # Which entity to command
    
    # Move action parameters (used when action_type=1)
    'move_center_grid': int,         # Grid position for patrol center
    'move_short_axis_km': int,       # Short axis of patrol ellipse
    'move_long_axis_km': int,        # Long axis of patrol ellipse
    'move_axis_angle': int,          # Rotation angle of patrol pattern
    
    # Engage action parameters (used when action_type=2)
    'target_group_id': int,          # Which detected target group to engage (represents enemy forces)
    'weapon_selection': int,         # Weapon combination to use
    'weapon_usage': int,             # How many shots per target
    'weapon_engagement': int,        # Engagement aggressiveness
    
    # Stealth action parameters (used when action_type=3)
    'stealth_enabled': int,          # 0=off, 1=on
    
    # Sensing action parameters (used when action_type=4)
    'sensing_position_grid': int,    # Grid position to focus sensors
    
    # Refuel action parameters (used when action_type=7)
    'refuel_target_id': int,         # Entity to refuel from/to
    
    # Jamming action parameters (used when action_type=8)
    'entity_to_protect_id': int,     # Friendly entity to protect with jamming
    'jam_target_grid': int,          # Grid position to jam (max_grid_positions = disable)
}
```

**Note**: All fields must be present in every action dict, even if not used by the selected action type.

## Action Types

### 0: No-Op
Do nothing this step.

**Parameters used**: None (but all fields must still be present)

### 1: Move / Combat Air Patrol (CAP)
Command an air unit to fly a patrol pattern.

**Parameters used**:
- `entity_id`: Which air unit to command
- `move_center_grid`: Center position of patrol (grid coordinate)
- `move_short_axis_km`: Short axis of patrol ellipse (discretized)
- `move_long_axis_km`: Long axis of patrol ellipse (discretized)
- `move_axis_angle`: Rotation of patrol pattern (discretized angles)

**Valid for**: Air units only

**Example**:
```python
action = {
    'action_type': 1,
    'entity_id': 5,
    'move_center_grid': 1250,
    'move_short_axis_km': 10,
    'move_long_axis_km': 20,
    'move_axis_angle': 8,
    'target_group_id': 0,
    'weapon_selection': 0,
    'weapon_usage': 0,
    'weapon_engagement': 0,
    'stealth_enabled': 0,
    'sensing_position_grid': 0,
    'refuel_target_id': 0,
    'entity_to_protect_id': 0,
    'jam_target_grid': 0
}
```

### 2: Engage
Command an entity to engage a detected enemy target group with weapons.

**Parameters used**:
- `entity_id`: Which entity fires
- `target_group_id`: Which detected target group to engage (from visible_targets mask)
- `weapon_selection`: Combination of weapons to use (platform-specific)
- `weapon_usage`: 0=1 shot per unit, 1=1 shot per adversary, 2=2 shots per adversary
- `weapon_engagement`: 0=defensive, 1=cautious, 2=assertive, 3=offensive

**Valid for**: Entities with weapons that can reach the target (check entity_target_matrix)

**Note**: Target groups represent detected enemy units visible to your sensors. You can only engage targets you've detected.

**Example**:
```python
action = {
    'action_type': 2,
    'entity_id': 3,
    'move_center_grid': 0,
    'move_short_axis_km': 0,
    'move_long_axis_km': 0,
    'move_axis_angle': 0,
    'target_group_id': 1,
    'weapon_selection': 5,
    'weapon_usage': 2,
    'weapon_engagement': 3,
    'stealth_enabled': 0,
    'sensing_position_grid': 0,
    'refuel_target_id': 0,
    'entity_to_protect_id': 0,
    'jam_target_grid': 0
}
```

### 3: Stealth Mode
Toggle radar emissions on/off for stealth.

**Parameters used**:
- `entity_id`: Which entity to configure
- `stealth_enabled`: 0=emit radar (detectable), 1=silent (stealthy)

**Valid for**: Entities with radar

### 4: Sensing Direction
Point sensors toward a specific area for enhanced detection.

**Parameters used**:
- `entity_id`: Which sensor platform to direct
- `sensing_position_grid`: Grid position to focus sensors toward

**Valid for**: Entities with radar/sensors

### 5: Capture Objective
Command ground unit to capture an objective flag.

**Parameters used**:
- `entity_id`: Which ground unit to command

**Valid for**: Ground units with capture capability

### 6: Return to Base (RTB)
Command aircraft to return to their carrier/airfield.

**Parameters used**:
- `entity_id`: Which aircraft to command

**Valid for**: Air units only

### 7: Refuel
Initiate refueling between tanker and receiver aircraft.

**Parameters used**:
- `entity_id`: Receiver aircraft
- `refuel_target_id`: Tanker aircraft entity ID

**Valid for**: Air units with refueling capability

### 8: Jam
Activate electronic jamming to protect a friendly entity by disrupting enemy sensors at a target location.

**Parameters used**:
- `entity_id`: Which jammer platform to command
- `entity_to_protect_id`: Friendly entity to protect with jamming
- `jam_target_grid`: Grid position to focus jamming on. Set to `max_grid_positions` to disable jamming.

**Valid for**: Entities with jamming capability (`has_jammer=True`)

**Example**:
```python
# Enable jamming to protect entity 2 at grid position 500
action = {
    'action_type': 8,
    'entity_id': 7,
    'move_center_grid': 0,
    'move_short_axis_km': 0,
    'move_long_axis_km': 0,
    'move_axis_angle': 0,
    'target_group_id': 0,
    'weapon_selection': 0,
    'weapon_usage': 0,
    'weapon_engagement': 0,
    'stealth_enabled': 0,
    'sensing_position_grid': 0,
    'refuel_target_id': 0,
    'entity_to_protect_id': 2,
    'jam_target_grid': 500
}

# Disable jamming (set jam_target_grid to max_grid_positions)
action = {
    'action_type': 8,
    'entity_id': 7,
    # ... other fields ...
    'entity_to_protect_id': 0,
    'jam_target_grid': 2500  # Assuming max_grid_positions = 2500
}
```

### 9: Spawn
Launch new units from a carrier or spawn-capable platform.

**Parameters used**:
- `entity_id`: Which carrier/platform to spawn from

**Valid for**: Entities with spawn capability (`can_spawn=True`)

**Example**:
```python
action = {
    'action_type': 9,
    'entity_id': 0,  # Carrier entity ID
    'move_center_grid': 0,
    'move_short_axis_km': 0,
    'move_long_axis_km': 0,
    'move_axis_angle': 0,
    'target_group_id': 0,
    'weapon_selection': 0,
    'weapon_usage': 0,
    'weapon_engagement': 0,
    'stealth_enabled': 0,
    'sensing_position_grid': 0,
    'refuel_target_id': 0,
    'entity_to_protect_id': 0,
    'jam_target_grid': 0
}
```

## Action Masking

The environment provides action masks in the `info` dict to indicate valid actions. These masks are useful for ensuring agents only select valid actions based on current game state.

```python
obs, reward, terminated, truncated, info = env.step(action)

info['valid_masks'] = {
    'action_types': {0, 1, 2, 5, 6, 8, 9},    # Valid action types this step
    'controllable_entities': {0, 3, 5, 7},     # Valid entity IDs (alive entities)
    'visible_targets': {1, 4},                 # Valid target group IDs (detected enemies)
    'entity_target_matrix': {                  # Entity-target engagement matrix
        0: {1, 4},    # Entity 0 can engage targets 1 and 4
        3: {4},       # Entity 3 can only engage target 4
    }
}
```

### Mask Usage

- **action_types**: Only action types in this set are valid this step
- **controllable_entities**: Only these entity IDs are alive and controllable
- **visible_targets**: Only these target group IDs are detected by your sensors
- **entity_target_matrix**: Maps which entities can engage which targets (weapon range check)

Invalid actions (not matching masks) will be rejected and treated as no-ops.

## Gymnasium Space Definition

The action space is defined as:

```python
spaces.Dict({
    "action_type": spaces.Discrete(10),
    "entity_id": spaces.Discrete(config.max_entities),
    "move_center_grid": spaces.Discrete(grid_size * grid_size),
    "move_short_axis_km": spaces.Discrete(patrol_steps),
    "move_long_axis_km": spaces.Discrete(patrol_steps),
    "move_axis_angle": spaces.Discrete(360 // angle_resolution),
    "target_group_id": spaces.Discrete(config.max_target_groups),
    "weapon_selection": spaces.Discrete(config.max_weapon_combinations),
    "weapon_usage": spaces.Discrete(3),
    "weapon_engagement": spaces.Discrete(4),
    "stealth_enabled": spaces.Discrete(2),
    "sensing_position_grid": spaces.Discrete(grid_size * grid_size + 1),
    "refuel_target_id": spaces.Discrete(config.max_entities),
    "entity_to_protect_id": spaces.Discrete(config.max_entities),
    "jam_target_grid": spaces.Discrete(grid_size * grid_size + 1),
})
```
