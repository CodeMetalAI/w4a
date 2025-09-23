"""
W4A Replay System

Leverages Bane's deterministic replay capabilities.
Records both RL agent state/action pairs and complete simulation state.
"""

# TODO: Test this replay functionality
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class W4AReplay:
    """Complete replay data combining RL and simulation views"""
    
    # RL Agent perspective (for analysis)
    observations: List[np.ndarray]
    actions: List[int] 
    rewards: List[float]
    
    # Complete simulation state (for deterministic replay)
    simulation_json: str  # FFSimulation's ExportJSON output
    
    # Metadata
    episode_info: Dict[str, Any]
    timestamp: str
    config: Dict[str, Any]


class ReplayRecorder:
    """Records both RL agent data and complete simulation state"""
    
    def __init__(self, save_dir: str = "./replays"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Current episode data
        self.observations = []
        self.actions = []
        self.rewards = []
        self.recording = False
        self.env = None  # Reference to environment
    
    def start_recording(self, env=None):
        """Start recording an episode"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.env = env
        self.recording = True
    
    def record_step(self, obs: np.ndarray, action: int, reward: float):
        """Record a single step from RL perspective"""
        if self.recording:
            self.observations.append(obs.copy())
            self.actions.append(action)
            self.rewards.append(reward)
    
    def save_replay(self, episode_name: str, info: Dict[str, Any] = None) -> str:
        """Save complete replay (RL data + simulation state)"""
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
    Simple episode visualization.
    
    Shows reward curve and basic stats.
    """
    import matplotlib.pyplot as plt
    
    # Plot reward curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(replay.rewards)
    plt.title("Rewards per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    
    plt.subplot(1, 2, 2)
    cumulative_rewards = np.cumsum(replay.rewards)
    plt.plot(cumulative_rewards)
    plt.title("Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()
    
    # Print episode stats
    print(f"\nReplay Stats:")
    print(f"Timestamp: {replay.timestamp}")
    print(f"Total steps: {len(replay.rewards)}")
    print(f"Total reward: {sum(replay.rewards):.2f}")
    print(f"Mean reward: {np.mean(replay.rewards):.2f}")
    print(f"Final observation shape: {replay.observations[-1].shape}")
    print(f"Has simulation data: {bool(replay.simulation_json)}")
    print(f"Episode info: {replay.episode_info}")


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
