"""
Episode Lifecycle Tests

Tests for episode lifecycle, termination conditions, and environment properties
for multiagent environment.
"""

import pytest
import numpy as np
from gymnasium import spaces
from w4a import Config
from w4a.envs.trident_multiagent_env import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction


class TestEnvironmentProperties:
    """Test environment properties and PettingZoo compliance"""
    
    def test_pettingzoo_interface_compliance(self):
        """Test environment implements PettingZoo Parallel interface correctly"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        # Required PettingZoo attributes
        assert hasattr(env, 'possible_agents'), "Missing possible_agents"
        assert hasattr(env, 'agents'), "Missing agents"
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        assert hasattr(env, 'close'), "Missing close method"
        
        # Agent list should be correct
        assert env.possible_agents == ["legacy", "dynasty"]
        assert env.agents == ["legacy", "dynasty"]
        
        # Set up agents to test spaces
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        # Spaces should be dicts mapping agent names to spaces
        assert hasattr(env, 'observation_spaces')
        assert hasattr(env, 'action_spaces')
        assert isinstance(env.observation_spaces, dict)
        assert isinstance(env.action_spaces, dict)
        assert "legacy" in env.observation_spaces
        assert "dynasty" in env.observation_spaces
        assert "legacy" in env.action_spaces
        assert "dynasty" in env.action_spaces
        
        # Methods should be callable
        assert callable(env.reset)
        assert callable(env.step)
        assert callable(env.close)
        
        env.close()


class TestTerminationConditions:
    """Test episode termination and truncation conditions for multiagent environment"""
    
    def test_time_based_truncation(self):
        """Test episodes truncate based on time limit for both agents"""
        config = Config()
        config.max_game_time = 100.0
        
        env = TridentIslandMultiAgentEnv(config=config)
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        step_count = 0
        max_steps = 200
        
        while step_count < max_steps:
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            
            observations, rewards, terms, truncs, infos = env.step(actions)
            step_count += 1
            
            # Check time information
            time_elapsed = infos["legacy"].get('time_elapsed', 0)
            
            if truncs["legacy"]:
                # Should truncate when time limit reached
                assert time_elapsed >= config.max_game_time, \
                    f"Truncated at {time_elapsed}s, but limit is {config.max_game_time}s"
                # Both agents should truncate together
                assert truncs["dynasty"]
                break
            
            if terms["legacy"]:
                # Early termination is also valid
                # Both agents should terminate together
                assert terms["dynasty"]
                break
        
        # Episode should have ended
        assert terms["legacy"] or truncs["legacy"], "Episode never ended within time/step limit"
        
        env.close()
    
    def test_synchronized_termination(self):
        """Test that both agents terminate and truncate together"""
        config = Config()
        config.max_game_time = 50.0
        
        env = TridentIslandMultiAgentEnv(config=config)
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        for step in range(100):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            
            observations, rewards, terms, truncs, infos = env.step(actions)
            
            # Both agents should always have same termination/truncation status
            assert terms["legacy"] == terms["dynasty"], "Agents have different termination status"
            assert truncs["legacy"] == truncs["dynasty"], "Agents have different truncation status"
            
            if terms["legacy"] or truncs["legacy"]:
                break
        
        env.close()
    
    def test_cleanup_and_multiple_cycles(self):
        """Test that simulation cleanup works correctly across multiple episodes"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        # First cycle
        env.reset()
        assert env.simulation is not None
        env.close()
        assert env.simulation is None
        
        # Second cycle
        env = TridentIslandMultiAgentEnv(config=config)
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        env.reset()
        assert env.simulation is not None
        env.close()
        assert env.simulation is None


class TestRewardStructure:
    """Test reward structure for multiagent environment"""
    
    def test_reward_types(self):
        """Test that rewards are returned correctly for both agents"""
        config = Config()
        env = TridentIslandMultiAgentEnv(config=config)
        
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        
        observations, rewards, terms, truncs, infos = env.step(actions)
        
        # Rewards should be returned for both agents
        assert "legacy" in rewards
        assert "dynasty" in rewards
        
        # Rewards should be numeric
        assert isinstance(rewards["legacy"], (int, float))
        assert isinstance(rewards["dynasty"], (int, float))
        
        env.close()
    
    def test_termination_cause_in_info(self):
        """Test that termination_cause is included in info dict"""
        config = Config()
        config.max_game_time = 30.0
        
        env = TridentIslandMultiAgentEnv(config=config)
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        # During episode, termination_cause should be None
        for step in range(5):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            
            observations, rewards, terms, truncs, infos = env.step(actions)
            
            if not terms["legacy"] and not truncs["legacy"]:
                # Still running - should be None
                assert infos["legacy"]["termination_cause"] is None
                assert infos["dynasty"]["termination_cause"] is None
            else:
                # Episode ended - should have a termination cause
                assert infos["legacy"]["termination_cause"] is not None
                assert infos["dynasty"]["termination_cause"] is not None
                # Both agents should have same termination cause
                assert infos["legacy"]["termination_cause"] == infos["dynasty"]["termination_cause"]
                # Should be one of the valid causes
                valid_causes = ["legacy_win", "dynasty_win", "time_limit", "terminated"]
                assert infos["legacy"]["termination_cause"] in valid_causes
                break
        
        env.close()
    
    def test_terminal_rewards_on_time_limit(self):
        """Test terminal rewards when episode ends by time limit"""
        config = Config()
        config.max_game_time = 30.0
        
        env = TridentIslandMultiAgentEnv(config=config)
        agent_legacy = CompetitionAgent(Faction.LEGACY, config)
        agent_dynasty = SimpleAgent(Faction.DYNASTY, config)
        env.set_agents(agent_legacy, agent_dynasty)
        
        observations, infos = env.reset()
        
        final_rewards = None
        final_infos = None
        for step in range(100):
            actions = {
                "legacy": agent_legacy.select_action(observations["legacy"]),
                "dynasty": agent_dynasty.select_action(observations["dynasty"])
            }
            
            observations, rewards, terms, truncs, infos = env.step(actions)
            
            if terms["legacy"] or truncs["legacy"]:
                final_rewards = rewards
                final_infos = infos
                break
        
        # Should have reached episode end
        assert final_rewards is not None
        assert final_infos is not None
        
        # Rewards should exist for both agents
        assert "legacy" in final_rewards
        assert "dynasty" in final_rewards
        
        # Check termination cause
        assert "termination_cause" in final_infos["legacy"]
        assert "termination_cause" in final_infos["dynasty"]
        
        # Rewards can be zero (draw), positive (win), or negative (loss)
        # Both should be numeric
        assert isinstance(final_rewards["legacy"], (int, float))
        assert isinstance(final_rewards["dynasty"], (int, float))
        
        env.close()


if __name__ == "__main__":
    test_properties = TestEnvironmentProperties()
    test_properties.test_pettingzoo_interface_compliance()
    
    test_termination = TestTerminationConditions()
    test_termination.test_time_based_truncation()
    test_termination.test_synchronized_termination()
    test_termination.test_cleanup_and_multiple_cycles()
    
    test_rewards = TestRewardStructure()
    test_rewards.test_reward_types()
    test_rewards.test_termination_cause_in_info()
    test_rewards.test_terminal_rewards_on_time_limit()