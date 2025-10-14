# W4A - Wargaming for All

Multi-agent reinforcement learning environment for tactical military simulation. Provides a PettingZoo Parallel interface to the Trident Island simulation engine for competitive agent-to-agent scenarios.

## Overview

W4A enables reinforcement learning research on realistic tactical scenarios including air, surface, and ground combat. Agents command military forces in competitive engagements with objectives including capture-and-hold and force-on-force attrition.

## Requirements

- Python 3.9 (required for SimulationInterface compatibility)
- Simulation engine with SimulationInterface bindings
- MacOS (ARM64), Linux (x64), Windows (x64)

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
- **TridentIslandMultiAgentEnv** - PettingZoo Parallel environment for multi agent (two-player) compeitive scenarios
- Gymnasium-compatible observation and action spaces
- Action masking for valid action selection
- Fog-of-war filtered observations

### Agents
- **CompetitionAgent** - Base class for custom RL agents with clean API
- **SimpleAgent** - Rule-based agent for testing and baselines
- Per-agent entity and target group tracking
- Customizable reward shaping, observation space implementation, action space implementation

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

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_actions.py -v

# Run with output
pytest tests/ -v -s
```

## Learn More

For game-specific information and gameplay details, visit [war.game](https://war.game).

## Support

If you encounter bugs or have questions, please file an issue on our [GitHub repository](https://github.com/CodeMetalAI/w4a/issues).

## Author

Sanjna Ravichandar - [sanjna@codemetal.ai](mailto:sanjna@codemetal.ai)
