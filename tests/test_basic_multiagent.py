"""
Basic test for multi-agent environment setup.

Tests that the environment can be created with agents and basic operations work.
"""

from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from w4a.config import Config
from SimulationInterface import Faction


def test_multiagent_basic_setup():
    """Test that environment can be set up with CompetitionAgent and SimpleAgent."""
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create agents
    agent_legacy = CompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
    
    # Set agents
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Check spaces are defined
    assert "legacy" in env.observation_spaces
    assert "dynasty" in env.observation_spaces
    assert "legacy" in env.action_spaces
    assert "dynasty" in env.action_spaces


def test_multiagent_reset_and_step():
    """Test reset and step operations for multiagent environment."""
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    # Create and set agents
    agent_legacy = CompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Reset
    observations, infos = env.reset(seed=42)
    
    # Check observations structure
    assert "legacy" in observations
    assert "dynasty" in observations
    assert observations["legacy"] is not None
    assert observations["dynasty"] is not None
    
    # Check infos structure
    assert "legacy" in infos
    assert "dynasty" in infos
    assert "faction" in infos["legacy"]
    assert "faction" in infos["dynasty"]
    assert infos["legacy"]["faction"] == "LEGACY"
    assert infos["dynasty"]["faction"] == "DYNASTY"
    
    # Step with actions from both agents
    actions = {
        "legacy": agent_legacy.select_action(observations["legacy"]),
        "dynasty": agent_dynasty.select_action(observations["dynasty"])
    }
    
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    # Check step returns correct structure
    assert "legacy" in obs and "dynasty" in obs
    assert "legacy" in rewards and "dynasty" in rewards
    assert "legacy" in terms and "dynasty" in terms
    assert "legacy" in truncs and "dynasty" in truncs
    assert "legacy" in infos and "dynasty" in infos
    
    # Check return types
    assert isinstance(rewards["legacy"], (int, float))
    assert isinstance(rewards["dynasty"], (int, float))
    assert isinstance(terms["legacy"], bool)
    assert isinstance(terms["dynasty"], bool)
    assert isinstance(truncs["legacy"], bool)
    assert isinstance(truncs["dynasty"], bool)
    
    env.close()


def test_multiagent_episode_completion():
    """Test that episodes can complete through termination or truncation."""
    config = Config()
    config.max_game_time = 50.0  # Short time for quick test
    env = TridentIslandMultiAgentEnv(config=config)
    
    agent_legacy = CompetitionAgent(Faction.LEGACY, config)
    agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
    env.set_agents(agent_legacy, agent_dynasty)
    
    observations, infos = env.reset(seed=42)
    
    max_steps = 100
    episode_ended = False
    
    for step in range(max_steps):
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        
        observations, rewards, terms, truncs, infos = env.step(actions)
        
        if terms["legacy"] or truncs["legacy"]:
            episode_ended = True
            # Both agents should terminate together
            assert terms["legacy"] == terms["dynasty"]
            assert truncs["legacy"] == truncs["dynasty"]
            break
    
    assert episode_ended, "Episode should end within time/step limit"
    env.close()


if __name__ == "__main__":
    test_multiagent_basic_setup()
    test_multiagent_reset_and_step()
    test_multiagent_episode_completion()

