"""
Basic tests to validate w4a environment works
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import RLEnvWrapper
from w4a.training.evaluation import RandomAgent, evaluate
from w4a.training.replay import ReplayRecorder
from SimulationInterface import (EntityList, EntitySpawnData)


def test_config():
    """Test config creation"""
    config = Config()
    
    # Config should be created successfully
    assert config is not None
    
    assert hasattr(config, 'max_episode_steps')
    assert isinstance(config.max_episode_steps, int)
    assert config.max_episode_steps > 0


def test_env_creation():
    """Create wrapped env, verify spaces and basic reward on one step."""
    from gymnasium import spaces
    wrapped = RLEnvWrapper(TridentIslandEnv())
    # Spaces exist and are Gymnasium spaces
    assert hasattr(wrapped, 'action_space')
    assert hasattr(wrapped, 'observation_space')
    assert isinstance(wrapped.action_space, spaces.Space)
    assert isinstance(wrapped.observation_space, spaces.Space)
    # Do a minimal reset/step to confirm reward is numeric
    obs, info = wrapped.reset()
    action = wrapped.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped.step(action)
    assert isinstance(reward, (int, float))
    wrapped.close()

def test_env_reset():
    """Test environment creation"""
    wrapped = RLEnvWrapper(TridentIslandEnv(enable_replay=True))

    obs, info = wrapped.reset()
    assert obs is not None
    assert isinstance(info, dict)

    # Agents should be installed by the wrapper and simulation created
    assert hasattr(wrapped.env, "legacy_agent")
    assert hasattr(wrapped.env, "dynasty_agent")
    assert wrapped.env.simulation is not None

def test_simulation_interface_integration():
    """Test that SimulationInterface integration works or fails gracefully"""
    wrapped = RLEnvWrapper(TridentIslandEnv())
    wrapped.reset()
    # Require SimulationInterface and successful simulation creation
    assert wrapped.env.simulation is not None


def test_env_reset_step_cycle():
    """Test complete environment usage cycle"""
    wrapped = RLEnvWrapper(TridentIslandEnv())
    
    # Test reset
    obs, info = wrapped.reset()
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == wrapped.observation_space.shape
    assert isinstance(info, dict)
    
    # Test multiple steps
    for _ in range(5):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        
        # Validate return types
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == wrapped.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        if terminated or truncated:
            break
    
    # Test cleanup
    wrapped.close()


def test_wrapper_hooks_applied():
    """Custom obs/action transforms via overriding wrapper methods"""
    class HookWrapper(RLEnvWrapper):
        def obs_fn(self, obs):
            import numpy as _np
            return _np.ones_like(obs)

        def action_fn(self, action):
            action = dict(action)
            action["action_type"] = 0  # force no-op
            return action

    wrapped = HookWrapper(TridentIslandEnv())
    obs, info = wrapped.reset()
    assert isinstance(obs, np.ndarray)
    assert np.allclose(obs, 1.0)

    action = wrapped.action_space.sample()
    # Should not raise with transformed action/obs and obs remains ones after step
    obs2, _, _, _, _ = wrapped.step(action)
    assert np.allclose(obs2, 1.0)
    wrapped.close()


# TODO: Add replay testing


def test_scenario_loading():
    """Test loading scenario JSON files into simulation"""
    import json
    from pathlib import Path
    from SimulationInterface import (EntityList, EntitySpawnData)
    
    try:
        # Test that scenario files exist and are valid JSON
        scenarios_path = Path(__file__).parent.parent / "src" / "w4a" / "scenarios"
        
        # Test force composition files
        legacy_composition = scenarios_path / "force_composition" / "LegacyEntityList.json"
        dynasty_composition = scenarios_path / "force_composition" / "DynastyEntityList.json"
        
        assert legacy_composition.exists(), "LegacyEntityList.json not found"
        assert dynasty_composition.exists(), "DynastyEntityList.json not found"
        
        # Test that they're valid JSON
        with open(legacy_composition) as f:
            legacy_data = EntityList().load_json(f.read())
            assert len(legacy_data.entities) > 0
            
        with open(dynasty_composition) as f:
            dynasty_data = EntityList().load_json(f.read())
            assert len(dynasty_data.entities) > 0
            
        # Test laydown files
        legacy_laydown = scenarios_path / "laydown" / "LegacyEntitySpawnData.json"
        dynasty_laydown = scenarios_path / "laydown" / "DynastyEntitySpawnData.json"
        
        assert legacy_laydown.exists(), "LegacyEntitySpawnData.json not found"
        assert dynasty_laydown.exists(), "DynastyEntitySpawnData.json not found"
        
        # Test that they're valid JSON
        with open(legacy_laydown) as f:
            legacy_spawn = EntitySpawnData.import_json(f.read())

            assert len(legacy_spawn.ground_forces_spawn_areas) > 0
            assert len(legacy_spawn.sea_forces_spawn_areas) > 0
            assert len(legacy_spawn.air_forces_spawn_locations) > 0
            assert len(legacy_spawn.caps) > 0

        with open(dynasty_laydown) as f:
            dynasty_spawn = EntitySpawnData.import_json(f.read())

            assert len(dynasty_spawn.ground_forces_spawn_areas) > 0
            assert len(dynasty_spawn.sea_forces_spawn_areas) > 0
            assert len(dynasty_spawn.air_forces_spawn_locations) > 0
            assert len(dynasty_spawn.caps) > 0
            
        # Test integration with SimulationInterface
        from SimulationInterface import EntitySpawnData
        
        # Test that EntitySpawnData can import the JSON - should raise exception if invalid
        try:
            legacy_spawn_data = EntitySpawnData.import_json(legacy_laydown.read_text())
            dynasty_spawn_data = EntitySpawnData.import_json(dynasty_laydown.read_text())
            
            assert legacy_spawn_data is not None
            assert dynasty_spawn_data is not None
            
            print("Scenario JSON files loaded successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to parse scenario JSON with SimulationInterface: {e}")
        
    except ImportError:
        pytest.skip("SimulationInterface not available - skipping scenario loading test")

# TODO: Add test for overriding with different JSON


# TODO: Add random agent testing
@pytest.mark.skip(reason="Test is disabled because it's not ready yet")
def test_random_agent_evaluation():
    """Test agent evaluation system"""
    env = TridentIslandEnv()
    agent = RandomAgent(seed=42)
    
    # Test short evaluation
    result = evaluate(agent, env, episodes=2)
    
    # Validate result structure
    assert isinstance(result, dict)
    assert "mean_reward" in result
    assert "std_reward" in result
    assert "mean_steps" in result
    assert "episodes" in result
    assert result["episodes"] == 2
    
    env.close()

def test_env_properties():
    """Gymnasium compliance for wrapped env with Dict action space."""
    from gymnasium import spaces
    env = RLEnvWrapper(TridentIslandEnv())
    # Required attributes
    assert hasattr(env, 'action_space')
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'close')
    # Spaces should be Gymnasium spaces
    assert isinstance(env.action_space, spaces.Dict)
    assert isinstance(env.observation_space, spaces.Space)
    # Reset/step cycle sanity
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, (int, float))
    env.close()


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic smoke tests...")
    test_config()
    test_env_creation()
    test_env_reset_step_cycle()
    print("Basic tests passed")
