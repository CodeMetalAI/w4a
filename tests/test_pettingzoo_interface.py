"""
Test PettingZoo Parallel interface compliance.

Uses PettingZoo's official parallel_api_test to verify the environment
properly implements the Parallel API.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from pettingzoo.test import parallel_api_test

from w4a.config import Config
from SimulationInterface import Faction


class MockCompetitionAgent:
    """Mock agent for testing PettingZoo interface without full simulation."""
    
    def __init__(self, faction: Faction, config: Config):
        self.faction = faction
        self.config = config
        self._env = None
        
        # Mock internal sim agent
        self._sim_agent = Mock()
        self._sim_agent.faction = faction
        self._sim_agent.controllable_entities = {}
        self._sim_agent.target_groups = {}
        self._sim_agent.flags = {}
    
    def _set_env(self, env):
        """Link agent to environment."""
        self._env = env
    
    def _get_sim_agent(self):
        """Return mock sim agent."""
        return self._sim_agent
    
    def get_observation(self):
        """Return mock observation."""
        return np.zeros(7, dtype=np.float32)
    
    def select_action(self, observation):
        """Return mock action (noop)."""
        return {
            'action_type': 0,
            'entity_id': 0,
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
        }
    
    def get_entities(self):
        """Return empty entity list."""
        return []
    
    def get_target_groups(self):
        """Return empty target group list."""
        return []
    
    def calculate_reward(self, env):
        """Default reward is 0.0."""
        return 0.0


def test_pettingzoo_interface_basic():
    """Test basic PettingZoo Parallel interface structure."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Check PettingZoo required attributes
    assert hasattr(env, 'possible_agents')
    assert hasattr(env, 'agents')
    assert env.possible_agents == ["legacy", "dynasty"]
    assert env.agents == ["legacy", "dynasty"]


def test_agent_registration():
    """Test agent registration via set_agents()."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create mock agents
    agent_legacy = MockCompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    
    # Register agents
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Verify agents are set
    assert env.agent_legacy is agent_legacy
    assert env.agent_dynasty is agent_dynasty
    
    # Verify agents are linked to environment
    assert agent_legacy._env is env
    assert agent_dynasty._env is env


def test_observation_action_spaces():
    """Test that observation and action spaces are set up correctly."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    from gymnasium import spaces
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create and register agents
    agent_legacy = MockCompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Check observation spaces
    obs_spaces = env.observation_spaces
    assert isinstance(obs_spaces, dict)
    assert "legacy" in obs_spaces
    assert "dynasty" in obs_spaces
    assert isinstance(obs_spaces["legacy"], spaces.Space)
    assert isinstance(obs_spaces["dynasty"], spaces.Space)
    
    # Check action spaces
    action_spaces = env.action_spaces
    assert isinstance(action_spaces, dict)
    assert "legacy" in action_spaces
    assert "dynasty" in action_spaces
    assert isinstance(action_spaces["legacy"], spaces.Dict)
    assert isinstance(action_spaces["dynasty"], spaces.Dict)


def test_spaces_before_agents_raises_error():
    """Test that accessing spaces before set_agents() raises error."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Should raise RuntimeError if agents not set
    with pytest.raises(RuntimeError, match="Agents must be set"):
        _ = env.observation_spaces
    
    with pytest.raises(RuntimeError, match="Agents must be set"):
        _ = env.action_spaces


def test_noop_action_structure():
    """Test that noop action has correct structure."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create and register agents
    agent_legacy = MockCompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Get noop action
    noop_action = env._get_noop_action()
    assert isinstance(noop_action, dict)
    assert 'action_type' in noop_action
    assert noop_action['action_type'] == 0
    
    # Verify it's valid for action space
    assert env.action_spaces["legacy"].contains(noop_action)
    assert env.action_spaces["dynasty"].contains(noop_action)


def test_reward_calculation():
    """Test that rewards are calculated per agent."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create and register agents
    agent_legacy = MockCompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Test default reward calculation (should return 0.0)
    reward_legacy = env._calculate_reward_for_agent(agent_legacy)
    reward_dynasty = env._calculate_reward_for_agent(agent_dynasty)
    
    assert isinstance(reward_legacy, float)
    assert isinstance(reward_dynasty, float)
    assert reward_legacy == 0.0
    assert reward_dynasty == 0.0


def test_custom_agent_reward():
    """Test that agents can implement custom reward functions."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    class CustomRewardAgent(MockCompetitionAgent):
        """Agent with custom reward function."""
        
        def calculate_reward(self, env):
            """Custom reward based on time elapsed."""
            return -0.1 * env.time_elapsed
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create custom agent
    agent_legacy = CustomRewardAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Set mock time
    env.time_elapsed = 10.0
    
    # Calculate rewards
    reward_legacy = env._calculate_reward_for_agent(agent_legacy)
    reward_dynasty = env._calculate_reward_for_agent(agent_dynasty)
    
    assert reward_legacy == -1.0
    assert reward_dynasty == 0.0


def test_info_dict_structure():
    """Test that info dict has correct structure for each agent."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create and register agents
    agent_legacy = MockCompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Build info dict
    info = env._build_info_for_agent(agent_legacy)
    
    # Check required fields
    assert 'step' in info
    assert 'time_elapsed' in info
    assert 'time_remaining' in info
    assert 'faction' in info
    assert 'total_entities' in info
    assert 'my_entities_count' in info
    assert 'detected_targets_count' in info
    assert 'last_events_count' in info
    assert 'valid_masks' in info
    assert 'controllable_entities' in info
    assert 'refuel' in info
    assert 'last_action_intent_by_entity' in info
    assert 'last_action_applied_by_entity' in info
    assert 'mission' in info
    
    # Check valid_masks structure
    masks = info['valid_masks']
    assert 'action_types' in masks
    assert 'controllable_entities' in masks
    assert 'visible_targets' in masks
    assert 'entity_target_matrix' in masks
    
    # Check refuel structure
    refuel = info['refuel']
    assert 'receivers' in refuel
    assert 'providers' in refuel
    
    # Check mission info structure
    mission = info['mission']
    assert 'my_casualties' in mission
    assert 'enemy_casualties' in mission
    assert 'kill_ratio' in mission
    assert 'my_capture_progress' in mission
    assert 'my_capture_possible' in mission
    assert 'enemy_capture_progress' in mission
    assert 'enemy_capture_possible' in mission
    assert 'island_contested' in mission
    
    # Note: termination_cause is added during reset/step, not in _build_info_for_agent


def test_both_agents_get_separate_info():
    """Test that both agents get their own info dicts with correct faction."""
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create and register agents
    agent_legacy = MockCompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = MockCompetitionAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Build info dicts
    info_legacy = env._build_info_for_agent(agent_legacy)
    info_dynasty = env._build_info_for_agent(agent_dynasty)
    
    # Verify factions are correct
    assert info_legacy['faction'] == 'LEGACY'
    assert info_dynasty['faction'] == 'DYNASTY'
    
    # Verify they're separate dicts (not same reference)
    assert info_legacy is not info_dynasty


def test_pettingzoo_parallel_api_official():
    """
    Official PettingZoo Parallel API test.
    
    This uses PettingZoo's built-in parallel_api_test to verify
    the environment correctly implements the Parallel API.
    """
    from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
    from w4a.agents import SimpleAgent
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Set up agents (use SimpleAgent for realistic test)
    agent_legacy = SimpleAgent(Faction.LEGACY, config)
    agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Run official PettingZoo API test
    parallel_api_test(env, num_cycles=10)


if __name__ == "__main__":
    test_pettingzoo_interface_basic()
    test_agent_registration()
    test_observation_action_spaces()
    test_spaces_before_agents_raises_error()
    test_noop_action_structure()
    test_reward_calculation()
    test_custom_agent_reward()
    test_info_dict_structure()
    test_both_agents_get_separate_info()
    test_pettingzoo_parallel_api_official()

