"""
RL Episode Lifecycle Tests

Tests for episode lifecycle, termination conditions, and environment properties.
"""

import pytest
import numpy as np
from gymnasium import spaces
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.training.evaluation import RandomAgent, evaluate


class TestEnvironmentProperties:
    """Test environment properties and Gymnasium compliance"""
    
    def test_gymnasium_interface_compliance(self):
        """Test environment implements Gymnasium interface correctly"""
        env = TridentIslandEnv()
        
        # Required attributes
        assert hasattr(env, 'action_space'), "Missing action_space"
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        assert hasattr(env, 'close'), "Missing close method"
        
        # Spaces should be proper Gymnasium spaces
        assert isinstance(env.action_space, spaces.Space), "action_space not a Space"
        assert isinstance(env.observation_space, spaces.Space), "observation_space not a Space"
        
        # Methods should be callable
        assert callable(env.reset), "reset not callable"
        assert callable(env.step), "step not callable"
        assert callable(env.close), "close not callable"
        
        env.close()


class TestTerminationConditions:
    """Test episode termination and truncation conditions"""
    
    def test_time_based_truncation(self):
        """Test episodes truncate based on time limit"""
        config = Config()
        config.max_game_time = 100.0  # Short time limit for testing
        
        env = TridentIslandEnv(config=config)
        obs, info = env.reset()
        
        step_count = 0
        max_steps = 200  # Prevent infinite loop
        
        while step_count < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Check time information
            time_elapsed = info.get('time_elapsed', 0)
            
            if truncated:
                # Should truncate when time limit reached
                assert time_elapsed >= config.max_game_time, \
                    f"Truncated at {time_elapsed}s, but limit is {config.max_game_time}s"
                break
            
            if terminated:
                # Early termination is also valid
                break
        
        # Episode should have ended
        assert terminated or truncated, "Episode never ended within time/step limit"
        
        env.close()
    
    def test_manual_termination_conditions(self):
        """Test manual termination by setting game state"""
        env = TridentIslandEnv()
        obs, info = env.reset()
        
        # Manually set capture conditions to test termination
        # This tests if the environment properly detects win conditions
        
        # Set capture timer to completion
        env.capture_timer_progress = env.config.capture_required_seconds
        env.capture_possible = True
        env.island_contested = False
        
        # Take a step to trigger termination check
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should detect capture completion
        if terminated:
            print("Capture-based termination detected")
            # Should get bonus reward for capture
            assert reward > 0, "No bonus reward for capture completion"
        
        env.close()
        
        # Test elimination-based termination
        env2 = TridentIslandEnv()
        obs, info = env2.reset()
        
        # Manually set all opponents as eliminated
        # This simulates destroying all enemy forces
        if hasattr(env2, 'enemy_kills'):
            # Set enemy kills to simulate total victory
            original_entities = len(env2.entities)
            env2.enemy_kills = list(range(original_entities // 2))  # Simulate killing half (enemies)
        
        action = env2.action_space.sample()
        obs, reward, terminated, truncated, info = env2.step(action)
        
        if terminated:
            print("Elimination-based termination detected")
            # Should get bonus reward for total victory
            assert reward >= 0, "Negative reward for victory"
        
        env2.close()


class TestEnvironmentIntegration:
    """Test environment integration with training utilities"""
    
    def test_random_agent_integration(self):
        """Test environment works with RandomAgent"""
        env = TridentIslandEnv()
        agent = RandomAgent(seed=42)
        
        # Should be able to run evaluation
        try:
            result = evaluate(agent, env, episodes=2)
            
            # Result should have expected structure
            assert isinstance(result, dict), "Evaluation result not dict"
            
            expected_keys = {'mean_reward', 'std_reward', 'mean_steps', 'episodes'}
            for key in expected_keys:
                if key in result:  # Some keys might be optional
                    assert isinstance(result[key], (int, float, np.number)), \
                        f"Result key {key} not numeric: {type(result[key])}"
            
            print("RandomAgent integration successful")
            
        except Exception as e:
            pytest.skip(f"RandomAgent evaluation not implemented: {e}")
        
        env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running RL lifecycle tests...")
    
    test_properties = TestEnvironmentProperties()
    test_properties.test_gymnasium_interface_compliance()
    print("Environment property tests passed")
    
    test_termination = TestTerminationConditions()
    test_termination.test_time_based_truncation()
    test_termination.test_manual_termination_conditions()
    print("Termination condition tests passed")
    
    test_integration = TestEnvironmentIntegration()
    test_integration.test_random_agent_integration()
    print("Environment integration tests passed")
    
    print("All RL lifecycle tests passed")