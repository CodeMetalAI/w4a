# W4A - Wargaming for All

A simple RL environment for tactical simulation. Gymnasium interface to BANE simulation engine.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install
pip install -e .

# Run tests
pytest
```

## What's Included

1. **TridentIslandEnv** - Single-agent Gymnasium environment
2. **EnvWrapper** - Wrapper for custom reward/action/obs
3. **Replay system** - Record and visualize episodes  
4. **RandomAgent** - Baseline for evaluation

## Usage

```python
from w4a import Config
from w4a.trident_island_env import TridentIslandEnv
from w4a.wrapper import EnvWrapper

# Basic environment
config = Config()
env = TridentIslandEnv(config=config)

# With wrapper for customizing obs, action, reward, etc.
wrapped = EnvWrapper(
    env,
    reward_fn=lambda obs, action, reward, info: reward * 2
)

# Run episode
obs, info = wrapped.reset()
action = wrapped.action_space.sample()
obs, reward, done, truncated, info = wrapped.step(action)
```

## Structure

```
w4a/
├── src/w4a/
│   ├── trident_island_env.py    # Main environment
│   ├── wrapper.py               # Simple wrapper
│   ├── replay.py                # Replay system
│   ├── evaluation.py            # Random agent + evaluate()
│   ├── config.py                # Simple config
│   └── constants.py             # Basic constants
└── tests/                       # Tests
```