"""
Test that agents can access their action and observation spaces.
"""

from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.agents import SimpleAgent
from w4a.config import Config
from SimulationInterface import Faction


def test_agent_space_access():
    """Test that agents can access action_space and observation_space."""
    print("\n=== Testing Agent Space Access ===")
    
    config = Config()
    env = TridentIslandMultiAgentEnv(config=config)
    
    agent_legacy = SimpleAgent(Faction.LEGACY, config)
    agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
    
    env.set_agents(agent_legacy, agent_dynasty)
    
    # Test action_space access
    action_space_legacy = agent_legacy.action_space
    action_space_dynasty = agent_dynasty.action_space
    
    print(f"✓ Legacy agent can access action_space: {action_space_legacy}")
    print(f"✓ Dynasty agent can access action_space: {action_space_dynasty}")
    
    # Test observation_space access
    obs_space_legacy = agent_legacy.observation_space
    obs_space_dynasty = agent_dynasty.observation_space
    
    print(f"✓ Legacy agent can access observation_space: {obs_space_legacy}")
    print(f"✓ Dynasty agent can access observation_space: {obs_space_dynasty}")
    
    # Test sampling from action space
    action = agent_legacy.action_space.sample()
    assert 'action_type' in action
    assert 'entity_id' in action
    print(f"✓ Can sample action from agent's action_space: {action}")
    
    # Test default select_action uses action_space
    obs, info = env.reset()
    action = agent_legacy.select_action(obs['legacy'])
    assert 'action_type' in action
    print(f"✓ Default select_action() samples from action_space")
    
    print("\n✅ All agent space access tests passed!")


if __name__ == "__main__":
    test_agent_space_access()

