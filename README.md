# W4A - Wargaming for All

Reinforcement learning environment for tactical simulation. Gymnasium interface to BANE simulation engine.

## Setup

Requires Python 3.9 for SimulationInterface compatibility.

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Run setup script
./setup.sh
```

## What's Included

- **TridentIslandEnv** - Single-agent Gymnasium environment
- **ForceDesignEnv** - Force composition environment skeleton
- **EnvWrapper** - Hooks for custom reward/action/observation functions
- **Replay system** - Episode recording with deterministic replay
- **Scenario examples** - Force composition and laydown template JSONs

## Usage

```python
from w4a import TridentIslandEnv, EnvWrapper, Config

# Basic environment
env = TridentIslandEnv()

# With custom reward function
wrapped = EnvWrapper(
    env,
    reward_fn=lambda obs, action, reward, info: reward + 10
)

# Run episode
obs, info = wrapped.reset()
action = wrapped.action_space.sample()
obs, reward, done, truncated, info = wrapped.step(action)
```

## Customizing Scenarios

Modify JSON files to customize force composition and deployment:

```
scenarios/
├── force_composition/     # What units to spawn
│   ├── LegacyEntityList.json    # Blue forces
│   └── DynastyEntityList.json   # Red forces
└── laydown/               # Where units spawn
    ├── LegacyEntitySpawnData.json   # Blue deployment areas
    └── DynastyEntitySpawnData.json  # Red deployment areas
```

## Structure

```
src/w4a/
├── envs/                  # Environment implementations
├── wrappers/              # Environment wrappers
├── training/              # Training utilities (evaluation, replay)
├── scenarios/             # Force design and deployment examples
├── config.py              # User configuration
└── constants.py           # Game constants
```