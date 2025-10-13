# Observation Space Documentation

## Overview

Each agent receives observations from their own perspective, filtered by fog-of-war. Agents only observe:
- Their own controllable entities
- Enemy units they have detected (via radar, visual contact, etc.)
- Shared objective flags

## Observation Format

Observations are returned as **normalized numpy arrays** with values in [0, 1].

**Current Status**: The observation system returns a flat vector of **12 global features**. The infrastructure for per-entity and per-target-group features is implemented but not yet active (returns zeros as placeholder).

## Current Features (v0.2)

```python
obs = np.array([
    time_remaining_norm,           # [0]  Mission time remaining (1.0 = full time, 0.0 = time up)
    my_casualties_norm,            # [1]  Your casualties / max_entities
    enemy_casualties_norm,         # [2]  Enemy casualties / max_entities
    kill_ratio_norm,               # [3]  Your kill ratio normalized by threshold
    awacs_alive_flag,              # [4]  1.0 if AWACS alive, 0.0 otherwise
    capture_progress_norm,         # [5]  Your capture progress / required_seconds
    enemy_capture_progress_norm,   # [6]  Enemy capture progress / required_seconds
    island_contested_flag,         # [7]  1.0 if objective contested, 0.0 otherwise
    capture_possible_flag,         # [8]  1.0 if you can still capture, 0.0 otherwise
    enemy_capture_possible_flag,   # [9]  1.0 if enemy can still capture, 0.0 otherwise
    island_center_x_norm,          # [10] Center island X coordinate normalized
    island_center_y_norm,          # [11] Center island Y coordinate normalized
])
```

**Note**: Currently returns zeros as a placeholder while the observation encoding pipeline is being finalized.

## Feature Descriptions

### time_remaining_norm
Remaining mission time normalized to [0, 1]
- `1.0`: Mission just started
- `0.5`: Half time remaining
- `0.0`: Time expired

### my_casualties_norm
Number of your units that have died, normalized by max entities
- `0.0`: No casualties
- `0.5`: Half your units destroyed
- `1.0`: All units destroyed (capped)

Note: Detailed casualties available in `info['mission']['my_casualties']`

### enemy_casualties_norm
Number of enemy units you've destroyed, normalized by max entities
- `0.0`: No enemy kills
- `1.0`: All enemy units destroyed (capped)

Note: Detailed enemy casualties available in `info['mission']['enemy_casualties']`

### kill_ratio_norm
Your kill ratio (enemy kills / your casualties) normalized by the win threshold
- `0.0`: Poor kill ratio
- `1.0`: At or above win threshold (typically 2.0x)
- Formula: `(enemy_casualties / my_casualties) / kill_ratio_threshold`

Note: Raw kill ratio available in `info['mission']['kill_ratio']`

### awacs_alive_flag
Whether your AWACS (airborne early warning) aircraft is alive
- `1.0`: AWACS operational (enhanced sensor coverage)
- `0.0`: AWACS destroyed (reduced sensor coverage)

### capture_progress_norm
Your progress toward capturing the objective (normalized 0-1)
- `0.0`: No progress
- `1.0`: Objective captured (win condition met)
- Formula: `my_capture_progress / capture_required_seconds`

Note: Detailed capture progress available in `info['mission']['my_capture_progress']`

### enemy_capture_progress_norm
Enemy's progress toward capturing the objective (normalized 0-1)
- `0.0`: Enemy has no progress
- `1.0`: Enemy has captured (loss condition met)
- Formula: `enemy_capture_progress / capture_required_seconds`

Note: Detailed enemy capture progress available in `info['mission']['enemy_capture_progress']`

### island_contested_flag
Whether both sides have units present at the capture objective
- `1.0`: Contested (both sides have capture-capable units present)
- `0.0`: Not contested (only one side present, or no one present)

Note: Also available in `info['mission']['island_contested']`

### capture_possible_flag
Whether you still have units capable of capturing
- `1.0`: You have capture-capable units alive
- `0.0`: No capture-capable units (cannot win via capture)

Note: Also available in `info['mission']['my_capture_possible']`

### enemy_capture_possible_flag
Whether the enemy still has units capable of capturing
- `1.0`: Enemy has capture-capable units alive
- `0.0`: Enemy has no capture-capable units (cannot win via capture)

Note: Also available in `info['mission']['enemy_capture_possible']`

### island_center_x_norm
X coordinate of the center island (capture objective) normalized to [0, 1]
- Normalized based on map size in kilometers
- Can be used to calculate distances and plan movements relative to objective

### island_center_y_norm
Y coordinate of the center island (capture objective) normalized to [0, 1]
- Normalized based on map size in kilometers
- Can be used to calculate distances and plan movements relative to objective

## Info Dict Structure

Each step returns an `info` dict with detailed mission metrics:

```python
obs, reward, terminated, truncated, info = env.step(action)

info = {
    'step': 42,                        # Current step number
    'time_elapsed': 120.5,             # Seconds elapsed
    'time_remaining': 879.5,           # Seconds remaining
    'faction': 'LEGACY',               # Your faction name
    
    # Entity counts
    'total_entities': 56,              # Total entities you control
    'my_entities_count': 48,           # Alive controllable entities
    'detected_targets_count': 12,      # Enemy groups detected
    
    # Action masks (for valid action selection)
    'valid_masks': {
        'action_types': {0, 1, 2, 5},           # Valid action types
        'controllable_entities': {0, 3, 5, 7},  # Valid entity IDs
        'visible_targets': {1, 4},              # Valid target IDs
        'entity_target_matrix': {...}           # Entity-target engagement matrix
    },
    
    # Refueling information
    'refuel': {
        'receivers': [5, 12, 18],      # Entities needing fuel
        'providers': [7, 9]            # Tanker entities
    },
    
    # Mission progress (faction-relative)
    'mission': {
        'my_casualties': 8,                    # Your losses
        'enemy_casualties': 12,                # Enemy losses
        'kill_ratio': 1.5,                     # Your kill ratio
        'my_capture_progress': 0.4,            # Your capture progress (0-1)
        'my_capture_possible': True,           # Can you still capture?
        'enemy_capture_progress': 0.1,         # Enemy capture progress
        'enemy_capture_possible': True,        # Can enemy still capture?
        'island_contested': False              # Is objective contested?
    }
}
```

## Querying Detailed State

For custom observations, you can query detailed state from your agent:

```python
class MyAgent(CompetitionAgent):
    def get_observation(self):
        # Get your entities
        my_entities = self.get_entities()
        
        # Get detected enemies
        enemy_groups = self.get_target_groups()
        
        # Access per-entity information
        for entity in my_entities:
            pos = entity.pos                    # Position (x, y, z)
            vel = entity.vel                    # Velocity
            fuel = entity.fuel                  # Fuel level
            is_alive = entity.is_alive          # Alive status
            domain = entity.domain              # AIR, SEA, or GROUND
            faction = entity.faction            # LEGACY or DYNASTY
            can_capture = entity.can_capture    # Capture capability
            # ... many more attributes available
        
        # Access per-target-group information
        for group in enemy_groups:
            faction = group.faction             # Enemy faction
            entity_count = group.get_entity_count()  # Number in group
            # ... more attributes
        
        # Build custom observation
        return self._encode_custom_observation(my_entities, enemy_groups)
```



## Fog of War

**Important**: Observations are limited by fog-of-war:
- You only see entities you control
- You only see enemy units your sensors detect
- Detection depends on:
  - Sensor range (radar, visual)
  - Enemy stealth status
  - Line of sight
  - Electronic warfare

**Implication**: Enemy observations may be incomplete or delayed. Design your agent to handle uncertainty!

## Customizing Observations

You can override `get_observation()` to customize:

```python
class MyAgent(CompetitionAgent):
    def get_observation(self):
        """Custom observation encoding."""
        my_entities = self.get_entities()
        enemy_groups = self.get_target_groups()
        
        # Example: Simple feature vector
        obs = np.zeros(10)
        obs[0] = len(my_entities) / 50  # Normalized entity count
        obs[1] = len(enemy_groups) / 20  # Normalized target count
        # ... your custom features
        
        return obs
```

## Gymnasium Space Definition

```python
spaces.Box(
    low=0.0,
    high=1.0,
    shape=(12,),  # Current: 12 global features
    dtype=np.float32
)
```

## Future Enhancements

The observation system has infrastructure prepared for expanded features:

### Planned Entity-Level Features (per friendly unit)
- **Identity & Intent** (7 features): can_engage, can_sense, can_refuel, can_capture, domain one-hot
- **Kinematics** (5 features): grid position, heading (sin/cos), speed
- **Egocentric** (2 features): distance/bearing to island
- **Status** (4 features): health, radar state, radar focus, fuel
- **Weapons** (4 features): air/surface weapon capability, ammo counts
- **Engagement** (6 features): engagement status, target info, target domain

Total: 28 features per entity (ID-indexed array)

### Planned Enemy Target Group Features
- **Detection** (1 feature): is_detected flag
- **Position** (2 features): normalized grid coordinates  
- **Domain** (3 features): air/surface/land one-hot
- **Count** (1 feature): number of units in group
- **Egocentric** (2 features): distance/bearing to island
- **Uncertainty** (1 feature): is_ghost (detection quality)

Total: 10 features per target group (ID-indexed array)

### ID-Indexed Design
Entity and target group features will use **stable ID indexing**:
- Row `i` corresponds to entity ID `i` (zero rows for unassigned IDs)
- Row `j` corresponds to target group ID `j` (zero rows for undetected groups)

This ensures observation indices remain aligned with action space IDs across timesteps.

