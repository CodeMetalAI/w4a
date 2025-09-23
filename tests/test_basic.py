"""
Basic tests to make sure things work
"""

from w4a import Config
from w4a.trident_island_env import TridentIslandEnv
from w4a.wrapper import EnvWrapper
from w4a.evaluation import RandomAgent, evaluate


def test_config():
    """Test config creation"""
    config = Config()
    assert config.max_episode_steps == 1000 # TODO: Change this test


def test_env_creation():
    """Test environment can be created"""
    env = TridentIslandEnv()
    assert env is not None


def test_env_reset_step():
    """Test basic environment usage"""
    env = TridentIslandEnv()
    obs, info = env.reset()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs is not None
    assert isinstance(reward, (int, float))


def test_wrapper():
    """Test wrapper works"""
    env = TridentIslandEnv()
    wrapped = EnvWrapper(env, reward_fn=lambda obs, action, reward, info: reward * 2)
    
    obs, info = wrapped.reset()
    action = wrapped.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped.step(action)
    
    assert obs is not None


def test_random_agent():
    """Test random agent"""
    env = TridentIslandEnv()
    agent = RandomAgent()
    
    result = evaluate(agent, env, episodes=2)
    assert "mean_reward" in result
