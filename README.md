# W4A - Wargaming for All

Multi-agent reinforcement learning environment for tactical military simulation. Provides a PettingZoo Parallel interface to the Trident Island simulation engine for competitive agent-to-agent scenarios.

## Overview

W4A enables reinforcement learning research on realistic tactical scenarios including air, surface, and ground combat. Agents command military forces in competitive engagements with objectives including capture-and-hold and force-on-force attrition.

## Requirements

- Python 3.9 (required for SimulationInterface compatibility)
- Simulation engine with SimulationInterface bindings

## Installation

```bash
# Create virtual environment (recommended)
python3.9 -m venv venv
source venv/bin/activate

# Run setup script
./setup.sh
```

The setup script will:
- Install required dependencies (pettingzoo, numpy)
- Build and install SimulationInterface with compiled binaries
- Install w4a
- Run basic tests to verify installation

## Quick Start

```python
from w4a import Config
from w4a.envs import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction

# Create environment
config = Config()
env = TridentIslandMultiAgentEnv(config=config)

# Create agents
agent_legacy = CompetitionAgent(Faction.LEGACY, config)
agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
env.set_agents(agent_legacy, agent_dynasty)

# Run episode
observations, infos = env.reset()

for step in range(1000):
    actions = {
        "legacy": agent_legacy.select_action(observations["legacy"]),
        "dynasty": agent_dynasty.select_action(observations["dynasty"])
    }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if terminations["legacy"] or truncations["legacy"]:
        observations, infos = env.reset()

env.close()
```

## What's Included

### Environments
- **TridentIslandMultiAgentEnv** - PettingZoo Parallel environment for two-player competitive scenarios
- Gymnasium-compatible observation and action spaces
- Action masking for valid action selection
- Fog-of-war filtered observations
- Observation space contains global mission state, per-entity features, and per-target-group features

### Agents
- **CompetitionAgent** - Base class for custom RL agents with clean API
  - Implement `select_action(observation)` with your AI policy (neural network, random policy, etc.)
  - Customizable reward shaping via `calculate_reward()`
  - Customizable observations via `get_observation()`
- **SimpleAgent** - Rule-based agent for testing and baselines
  - Automatically engages all detected enemies with available weapons
  - Useful as opponent during training or for baseline comparisons
- Per-agent entity and target group tracking with fog-of-war

### Features
- Multi-domain combat (air, surface)
- Realistic sensor and weapon systems
- Capture-and-hold objectives
- Deterministic episode replay support

## Documentation

Detailed documentation is available in the `docs/` directory:

- **ACTION_SPACE.md** - Action types, parameters, and masking
- **OBSERVATION_SPACE.md** - Observation structure and fog-of-war
- **REWARDS.md** - Default rewards and custom shaping

## Project Structure

```
w4a/
├── src/w4a/
│   ├── agents/              # Agent implementations
│   │   ├── competition_agent.py
│   │   ├── simple_agent.py
│   │   └── _simulation_agent.py
│   ├── envs/                # Environment implementations
│   │   ├── trident_multiagent_env.py
│   │   ├── actions.py
│   │   ├── observations.py
│   │   └── simulation_utils.py
│   ├── entities/            # Entity type definitions
│   ├── scenarios/           # Scenario configurations
│   │   └── trident_island/
│   ├── config.py            # Configuration parameters
│   └── constants.py         # Game constants
├── tests/                   # Test suite
├── docs/                    # Documentation
└── SimulationInterface/     # Simulation Engine Python bindings
```

## Development and Testing

The `tests/` directory contains comprehensive test suites that validate the simulation's behavior during development. These tests are designed to ensure simulation fidelity and API correctness.

**Note for Users**: Tests are development tools for validating the simulation engine. You do not need to worry about tests passing as you modify agents or experiment with the environment. The tests help maintain simulation integrity, but your custom agents and experiments are independent of the test suite.

```bash

# Run specific test file (optional, for development validation)
pytest tests/test_basic_multiagent.py -v -s

```

## Learn More

For game-specific information and gameplay details, visit [war.game](https://war.game).

## Support

If you encounter bugs or have questions, please file an issue on our [GitHub repository](https://github.com/CodeMetalAI/w4a/issues).

## Author

Sanjna Ravichandar - [sanjna@codemetal.ai](mailto:sanjna@codemetal.ai)
