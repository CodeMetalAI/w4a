"""
W4A Replay System

Leverages Bane's deterministic replay capabilities.
Records both RL agent state/action pairs and complete simulation state.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class W4AReplay:
    """Complete replay data for multiagent episodes.
    
    Structure:
        observations: Dict[agent_name, List[np.ndarray]]  # e.g., {"legacy": [...], "dynasty": [...]}
        actions: Dict[agent_name, List[Dict]]
        rewards: Dict[agent_name, List[float]]
    """
    
    # Per-agent data (dict keyed by agent name: "legacy", "dynasty")
    observations: Dict[str, List[np.ndarray]]
    actions: Dict[str, List[Dict]]
    rewards: Dict[str, List[float]]
    
    # Complete simulation state (for deterministic replay)
    simulation_json: str  # FFSimulation's ExportJSON output
    
    # Metadata
    episode_info: Dict[str, Any]
    timestamp: str
    config: Dict[str, Any]


class ReplayRecorder:
    """Records both RL agent data and complete simulation state for multiagent episodes."""
    
    def __init__(self, save_dir: str = "./replays"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Current episode data (dict of lists per agent)
        self.observations = {}
        self.actions = {}
        self.rewards = {}
        self.recording = False
        self.env = None  # Reference to environment
    
    def start_recording(self, env=None):
        """Start recording an episode.
        
        Args:
            env: TridentIslandMultiAgentEnv instance
        """
        self.env = env
        
        # Initialize dict of lists for each agent
        self.observations = {agent: [] for agent in env.possible_agents}
        self.actions = {agent: [] for agent in env.possible_agents}
        self.rewards = {agent: [] for agent in env.possible_agents}
        
        self.recording = True
    
    def record_step(
        self, 
        obs: Dict[str, np.ndarray], 
        action: Dict[str, Dict], 
        reward: Dict[str, float]
    ):
        """Record a single step from RL perspective.
        
        Args:
            obs: Observations dict keyed by agent name
            action: Actions dict keyed by agent name
            reward: Rewards dict keyed by agent name
        """
        if not self.recording:
            return
        
        # Record for each agent
        for agent_name in self.observations.keys():
            if agent_name in obs:
                self.observations[agent_name].append(
                    obs[agent_name].copy() if isinstance(obs[agent_name], np.ndarray) 
                    else obs[agent_name]
                )
            if agent_name in action:
                self.actions[agent_name].append(action[agent_name])
            if agent_name in reward:
                self.rewards[agent_name].append(reward[agent_name])
    
    def save_replay(self, episode_name: str, info: Dict[str, Any] = None) -> str:
        """Save complete replay (RL data + simulation state).
        
        Args:
            episode_name: Name for the replay file
            info: Additional episode information to store
            
        Returns:
            Path to saved replay file
        """
        if not self.recording:
            return ""
        
        # Get deterministic simulation replay from FFSimulation
        simulation_json = ""
        if self.env and hasattr(self.env, 'get_simulation_handle'):
            sim_handle = self.env.get_simulation_handle()
            if sim_handle:
                # This uses FFSimulation's built-in ExportJSON
                simulation_json = sim_handle.export_json()
        
        replay = W4AReplay(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            simulation_json=simulation_json,
            episode_info=info or {},
            timestamp=datetime.now().isoformat(),
            config=getattr(self.env, 'config', {})
        )
        
        # Save with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.save_dir / f"{episode_name}_{timestamp}.w4a"
        
        with open(filepath, 'wb') as f:
            pickle.dump(replay, f)
        
        print(f"Saved W4A replay: {filepath}")
        self.recording = False
        return str(filepath)
    
    def load_replay(self, replay_path: str) -> W4AReplay:
        """Load a complete replay"""
        with open(replay_path, 'rb') as f:
            return pickle.load(f)
    
    def export_simulation_json(self, replay_path: str, output_path: str = None):
        """Extract just the simulation JSON for FFSimulation replay"""
        replay = self.load_replay(replay_path)
        
        if not output_path:
            output_path = replay_path.replace('.w4a', '_simulation.json')
        
        with open(output_path, 'w') as f:
            f.write(replay.simulation_json)
        
        print(f"Exported simulation replay: {output_path}")
        return output_path


def visualize_replay(replay: W4AReplay, save_path: str = None):
    """
    Episode visualization for multiagent replays.
    
    Shows reward curves and basic stats for all agents.
    
    Args:
        replay: W4AReplay object to visualize
        save_path: Optional path to save visualization image
    """
    import matplotlib.pyplot as plt
    
    agent_names = list(replay.rewards.keys())
    num_agents = len(agent_names)
    
    # Create figure with subplots for each agent
    fig, axes = plt.subplots(num_agents, 2, figsize=(12, 4 * num_agents))
    if num_agents == 1:
        axes = axes.reshape(1, -1)
    
    for idx, agent_name in enumerate(agent_names):
        agent_rewards = replay.rewards[agent_name]
        
        # Per-step rewards
        axes[idx, 0].plot(agent_rewards)
        axes[idx, 0].set_title(f"{agent_name.title()} - Rewards per Step")
        axes[idx, 0].set_xlabel("Step")
        axes[idx, 0].set_ylabel("Reward")
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Cumulative rewards
        cumulative_rewards = np.cumsum(agent_rewards)
        axes[idx, 1].plot(cumulative_rewards)
        axes[idx, 1].set_title(f"{agent_name.title()} - Cumulative Reward")
        axes[idx, 1].set_xlabel("Step")
        axes[idx, 1].set_ylabel("Cumulative Reward")
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()
    
    # Print episode stats
    print(f"\n=== Replay Stats ===")
    print(f"Timestamp: {replay.timestamp}")
    
    for agent_name in agent_names:
        agent_rewards = replay.rewards[agent_name]
        agent_obs = replay.observations[agent_name]
        
        print(f"\n{agent_name.upper()}:")
        print(f"  Total steps: {len(agent_rewards)}")
        print(f"  Total reward: {sum(agent_rewards):.2f}")
        print(f"  Mean reward: {np.mean(agent_rewards):.4f}")
        print(f"  Final observation shape: {agent_obs[-1].shape if agent_obs else 'N/A'}")
    
    print(f"\nHas simulation data: {bool(replay.simulation_json)}")
    print(f"Episode info: {replay.episode_info}")


def record_multiagent_episode(env, agent_legacy, agent_dynasty, recorder: ReplayRecorder, episode_name: str = "episode"):
    """
    Convenience function to record a full multiagent episode.
    
    Args:
        env: TridentIslandMultiAgentEnv instance
        agent_legacy: CompetitionAgent for Legacy faction
        agent_dynasty: CompetitionAgent for Dynasty faction
        recorder: ReplayRecorder instance
        episode_name: Name for the saved replay file
        
    Returns:
        Tuple of (replay_path, episode_rewards, episode_info)
    """
    recorder.start_recording(env)
    
    observations, infos = env.reset()
    episode_rewards = {"legacy": 0.0, "dynasty": 0.0}
    done = False
    steps = 0
    
    while not done:
        # Get actions from both agents
        actions = {
            "legacy": agent_legacy.select_action(observations["legacy"]),
            "dynasty": agent_dynasty.select_action(observations["dynasty"])
        }
        
        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Record step
        recorder.record_step(observations, actions, rewards)
        
        # Update totals
        episode_rewards["legacy"] += rewards["legacy"]
        episode_rewards["dynasty"] += rewards["dynasty"]
        steps += 1
        
        # Check if episode is done
        done = terminations["legacy"] or truncations["legacy"]
        
        observations = next_observations
    
    # Save replay
    episode_info = {
        "total_steps": steps,
        "total_rewards": episode_rewards,
        "winner": infos["legacy"].get("termination_cause", "unknown")
    }
    
    replay_path = recorder.save_replay(episode_name, episode_info)
    
    return replay_path, episode_rewards, episode_info


if __name__ == "__main__":
    """Simple replay viewer script"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python replay.py <replay_file.w4a>")
        sys.exit(1)
    
    replay_file = sys.argv[1]
    recorder = ReplayRecorder()
    
    try:
        replay = recorder.load_replay(replay_file)
        visualize_replay(replay)
        
        # Optionally export simulation JSON
        if replay.simulation_json:
            print("\nExporting simulation JSON for FFSimulation replay...")
            json_file = recorder.export_simulation_json(replay_file)
            print(f"Use this file to replay in FFSimulation: {json_file}")
            
    except FileNotFoundError:
        print(f"Replay file '{replay_file}' not found")
        sys.exit(1)
