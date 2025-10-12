# Reward Structure Documentation

## Overview

The W4A environment uses a **sparse reward** structure by default, with large terminal rewards for winning/losing. Users can implement custom per-step reward shaping by overriding `calculate_reward()` in their agent.

## Default Rewards

### Per-Step Reward
**Default**: `0.0` (neutral)

The environment returns no per-step reward by default.

### Terminal Rewards
Automatically added when episode ends:

| Outcome | Reward |
|---------|--------|
| **Win** | `+100.0` |
| **Loss** | `-100.0` |
| **Draw/Timeout** | `0.0` |

**Zero-Sum**: In competitive mode, one agent's win is the other's loss:
- Winner gets `+100.0`
- Loser gets `-100.0`

## Win/Loss Conditions

### Win Conditions
An agent wins if:
1. **Capture Victory**: Maintain control of objective for required duration
2. **Kill Ratio Victory**: Achieve favorable kill ratio (default: destroy 3× more enemies than losses)

### Loss Conditions  
An agent loses if:
1. **Cannot Capture**: Lost all capture-capable units AND unfavorable kill ratio
2. **Inverse Kill Ratio**: Enemy has 3× better kill ratio

### Timeout
If neither side wins/loses before time limit:
- Episode truncates (not terminated)
- No terminal reward given

## Custom Per-Step Rewards

Override `calculate_reward()` to implement custom reward shaping:

```python
class MyAgent(CompetitionAgent):
    def calculate_reward(self, env) -> float:
        """Custom per-step reward shaping."""
        reward = 0.0
        
        # Force preservation
        my_entities = self.get_entities()
        reward += 0.1 * len(my_entities)
        
        # Attrition
        my_casualties = len([k for k in env.friendly_kills if k.faction == self.faction])
        enemy_casualties = len([k for k in env.enemy_kills if k.faction != self.faction])
        reward -= 0.5 * my_casualties
        reward += 0.3 * enemy_casualties
        
        # Capture progress
        capture_progress = env.capture_progress_by_faction[self.faction]
        if capture_progress > 0:
            reward += 0.5 * capture_progress
        
        # Detection bonus
        enemy_groups = self.get_target_groups()
        reward += 0.05 * len(enemy_groups)
        
        # Time penalty
        reward -= 0.01
        
        return reward
```

### Capture-Focused Example

```python
class MyAgent(CompetitionAgent):
    def calculate_reward(self, env) -> float:
        """Optimize for capture victory."""
        reward = 0.0
        
        # Strong capture progress incentive
        capture_progress = env.capture_progress_by_faction[self.faction] / env.config.capture_required_seconds
        reward += 10.0 * capture_progress
        
        # Penalty if objective contested
        if env.island_contested:
            reward -= 1.0
        
        # Bonus for protecting capture-capable units
        capturers = [e for e in self.get_entities() if e.can_capture]
        reward += 0.5 * len(capturers)
        
        return reward
```
