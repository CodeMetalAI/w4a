# Observation Space Documentation

## Overview

Each agent receives observations from their own perspective, filtered by fog-of-war. Agents only observe:
- Their own controllable entities
- Enemy units they have detected (via radar, visual contact, etc.), represented as "target groups"
- Shared objective flags

## Observation Format

Observations are returned as **normalized numpy arrays** with values in [0, 1].

**Total Size**: 11 global features + (max_entities × 36) + (max_target_groups × 12)

The observation space consists of three components:
1. **Global features** (11): Mission state, time, objectives, casualties, flag status
2. **Friendly entity features** (max_entities × 36): Per-entity capabilities, kinematics, status, engagement
3. **Enemy target group features** (max_target_groups × 12): Per-group position, velocity, domain, count

## Structure

```python
observation = np.concatenate([
    global_features,      # Shape: (11,)
    friendly_features,    # Shape: (max_entities * 36,)
    enemy_features        # Shape: (max_target_groups * 12,)
])
```

### ID-Indexed Layout
Entity and target group features use **stable ID indexing**:
- Entity features at index `i * 36` correspond to entity ID `i`
- Target group features correspond to target group ID `j`
- Unassigned/invisible IDs have zero-filled rows
- This keeps observation indices aligned with action space IDs across timesteps

## Global Features (11 total)

```python
global_features = [
    time_remaining_norm,           # [0]  Mission time remaining (1.0 = full, 0.0 = expired)
    my_casualties_norm,            # [1]  Your casualties / max_entities
    enemy_casualties_norm,         # [2]  Enemy casualties / max_entities
    kill_ratio_norm,               # [3]  Your kill ratio / threshold
    capture_progress_norm,         # [4]  Your capture progress / required_seconds
    capture_possible_flag,         # [5]  1.0 if you can capture, 0.0 otherwise
    enemy_capture_progress_norm,   # [6]  Enemy capture progress / required_seconds
    flag_faction,                  # [7]  0.0=neutral, 0.33=Legacy, 0.66=Dynasty
    enemy_capture_possible_flag,   # [8]  1.0 if enemy can capture, 0.0 otherwise
    island_center_x_norm,          # [9]  Center island X coordinate [0,1]
    island_center_y_norm,          # [10] Center island Y coordinate [0,1]
]
```

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
- `1.0`: At or above win threshold (default: 1.2×)
- Formula: `(enemy_casualties / my_casualties) / kill_ratio_threshold`

Note: Raw kill ratio available in `info['mission']['kill_ratio']`

### flag_faction
Which faction currently controls the center island flag
- `0.0`: Neutral (no faction control)
- `0.33`: Legacy faction controls flag
- `0.66`: Dynasty faction controls flag

**Capture Mechanics:**
- **During active capture**: Flag remains neutral (`0.0`) while `is_being_captured = True`
- **After capture completes**: Flag switches to capturing faction (`0.33` or `0.66`), `is_captured = True`

### capture_progress_norm
Your progress toward capturing the objective (normalized 0-1)
- `0.0`: No progress or capture complete
- `0.0 < x < 1.0`: Actively capturing (settler on flag)
- `1.0`: At threshold, capture about to complete
- Formula: `my_capture_progress / capture_required_seconds`

**Capture Lifecycle:**
1. **Pre-capture**: `capture_progress_norm = 0.0`, flag neutral
2. **During capture**: `capture_progress_norm` increments (0.0 → 1.0), `is_being_captured = True`, flag stays neutral
3. **After capture**: `capture_progress_norm = 0.0`, `is_captured = True`, flag_faction changes to your faction

Note: Detailed capture progress (in seconds) available in `info['mission']['my_capture_progress']`

### enemy_capture_progress_norm
Enemy's progress toward capturing the objective (normalized 0-1)
- `0.0`: Enemy has no progress or capture complete
- `0.0 < x < 1.0`: Enemy actively capturing
- `1.0`: Enemy at threshold, capture about to complete
- Formula: `enemy_capture_progress / capture_required_seconds`

Note: Detailed enemy capture progress available in `info['mission']['enemy_capture_progress']`


### capture_possible_flag
Whether you still have units capable of capturing
- `1.0`: You have capture-capable units alive
- `0.0`: No capture-capable units (cannot win via capture)

**Note**: If the flag is currently being captured by either side, this will be `0.0` (can't capture while someone else is capturing)

Available in `info['mission']['my_capture_possible']`

### enemy_capture_possible_flag
Whether the enemy still has units capable of capturing
- `1.0`: Enemy has capture-capable units alive
- `0.0`: Enemy has no capture-capable units (cannot win via capture)

Available in `info['mission']['enemy_capture_possible']`

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
        'my_capture_progress': 0.4,            # Your capture progress (seconds)
        'my_capture_possible': True,           # Can you still capture?
        'enemy_capture_progress': 0.1,         # Enemy capture progress (seconds)
        'enemy_capture_possible': True         # Can enemy still capture?
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

## Friendly Entity Features (36 per entity)

Each friendly entity (units you control) has 36 features organized as:

### Identity & Intent (7 features)
- `can_engage`: Can fire weapons (0.0 or 1.0)
- `can_sense`: Has radar/sensors (0.0 or 1.0)
- `can_refuel`: Can refuel others (0.0 or 1.0)
- `can_capture`: Can capture objectives (0.0 or 1.0)
- `domain_air`, `domain_sea`, `domain_ground`: One-hot domain encoding

### Kinematics (7 features)
- `grid_x`, `grid_y`: Normalized grid position [0,1]
- `heading_sin`, `heading_cos`: Heading direction encoding
- `speed_norm`: Speed normalized by max speed
- `altitude_norm`: Altitude normalized (air units only)
- `manoeuver`: Current maneuver type (patrol, engage, RTB, etc.)

### Egocentric (2 features)
- `distance_to_island_norm`: Distance to objective [0,1]
- `bearing_to_island_norm`: Bearing to objective [0,1]

### Status (6 features)
- `health_norm`: Health/hit points [0,1]
- `fuel_norm`: Fuel level [0,1]
- `radar_enabled`: Radar on/off (0.0 or 1.0)
- `radar_focus_x`, `radar_focus_y`: Radar focus position [0,1]
- `selected_flag`: Currently selected entity (0.0 or 1.0)

### Weapons (4 features)
- `has_air_weapons`: Can engage air targets (0.0 or 1.0)
- `has_surface_weapons`: Can engage surface targets (0.0 or 1.0)
- `air_ammo_norm`: Air weapon ammo [0,1]
- `surface_ammo_norm`: Surface weapon ammo [0,1]

### Engagement (10 features)
- `is_engaging`: Currently engaging target (0.0 or 1.0)
- `target_id_norm`: Current target ID normalized
- `target_grid_x`, `target_grid_y`: Target position [0,1]
- `target_distance_norm`: Distance to target [0,1]
- `target_bearing_norm`: Bearing to target [0,1]
- `target_domain_air`, `target_domain_sea`, `target_domain_ground`: Target domain
- `can_engage_target`: Weapon range check (0.0 or 1.0)

**Total**: 36 features per entity

## Enemy Target Group Features (12 per target group)

Each detected enemy target group has 12 features:

### Detection & Identity (4 features)
- `is_detected`: Target visible to sensors (0.0 or 1.0)
- `domain_air`, `domain_sea`, `domain_ground`: One-hot domain encoding

### Position & Velocity (6 features)
- `grid_x`, `grid_y`: Normalized grid position [0,1]
- `velocity_x`, `velocity_y`: Velocity components [0,1]
- `speed_norm`: Speed magnitude [0,1]
- `heading_norm`: Heading direction [0,1]

### Count & Egocentric (2 features)
- `entity_count_norm`: Number of units in group [0,1]
- `distance_to_island_norm`: Distance to objective [0,1]

**Total**: 12 features per target group

## Gymnasium Space Definition

```python
spaces.Box(
    low=0.0,
    high=1.0,
    shape=(11 + max_entities*36 + max_target_groups*12,),
    dtype=np.float32
)
```

Example with default config (max_entities=100, max_target_groups=50):
- Shape: `(3711,)` = 11 + (100×36) + (50×12)

