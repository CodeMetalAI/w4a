"""
Evaluation Module

Provides evaluation functions for multiagent environments with CompetitionAgent interface.
"""

import numpy as np
from typing import Dict, Any


def evaluate(
    agent_legacy, 
    agent_dynasty, 
    env, 
    episodes: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate two agents competing in the multiagent environment.
    
    Args:
        agent_legacy: CompetitionAgent for Legacy faction
        agent_dynasty: CompetitionAgent for Dynasty faction
        env: TridentIslandMultiAgentEnv instance
        episodes: Number of episodes to run
        verbose: If True, print episode results
        
    Returns:
        dict with statistics for both agents:
        {
            "legacy": {"mean_reward": ..., "std_reward": ..., "wins": ..., ...},
            "dynasty": {"mean_reward": ..., "std_reward": ..., "wins": ..., ...},
            "draws": ...,
            "mean_steps": ...,
            "episodes": ...
        }
    """
    # Track per-agent stats
    legacy_rewards = []
    dynasty_rewards = []
    legacy_wins = 0
    dynasty_wins = 0
    draws = 0
    total_steps = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        
        episode_reward_legacy = 0.0
        episode_reward_dynasty = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # Get actions from both agents
            actions = {
                "legacy": agent_legacy.select_action(obs["legacy"]),
                "dynasty": agent_dynasty.select_action(obs["dynasty"])
            }
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Accumulate rewards
            episode_reward_legacy += rewards["legacy"]
            episode_reward_dynasty += rewards["dynasty"]
            episode_steps += 1
            
            # Check termination
            done = terminations["legacy"] or truncations["legacy"]
        
        # Record episode results
        legacy_rewards.append(episode_reward_legacy)
        dynasty_rewards.append(episode_reward_dynasty)
        total_steps.append(episode_steps)
        
        # Determine winner
        termination_cause = infos["legacy"].get("termination_cause", None)
        if termination_cause == "legacy_win":
            legacy_wins += 1
            winner = "Legacy"
        elif termination_cause == "dynasty_win":
            dynasty_wins += 1
            winner = "Dynasty"
        else:
            draws += 1
            winner = "Draw"
        
        if verbose:
            print(f"Episode {episode + 1}/{episodes}: Winner={winner}, "
                  f"Legacy={episode_reward_legacy:.2f}, Dynasty={episode_reward_dynasty:.2f}, "
                  f"Steps={episode_steps}")
    
    # Compile statistics
    results = {
        "legacy": {
            "mean_reward": float(np.mean(legacy_rewards)),
            "std_reward": float(np.std(legacy_rewards)),
            "total_reward": float(np.sum(legacy_rewards)),
            "wins": legacy_wins,
            "win_rate": legacy_wins / episodes
        },
        "dynasty": {
            "mean_reward": float(np.mean(dynasty_rewards)),
            "std_reward": float(np.std(dynasty_rewards)),
            "total_reward": float(np.sum(dynasty_rewards)),
            "wins": dynasty_wins,
            "win_rate": dynasty_wins / episodes
        },
        "draws": draws,
        "mean_steps": float(np.mean(total_steps)),
        "std_steps": float(np.std(total_steps)),
        "episodes": episodes
    }
    
    return results


def print_evaluation_results(results: Dict[str, Any]):
    """
    Pretty print evaluation results from evaluate_multiagent.
    
    Args:
        results: Results dict from evaluate_multiagent
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {results['episodes']}")
    print(f"Mean Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print()
    
    print("LEGACY FACTION:")
    print(f"  Mean Reward: {results['legacy']['mean_reward']:.2f} ± {results['legacy']['std_reward']:.2f}")
    print(f"  Total Reward: {results['legacy']['total_reward']:.2f}")
    print(f"  Wins: {results['legacy']['wins']} ({results['legacy']['win_rate']*100:.1f}%)")
    print()
    
    print("DYNASTY FACTION:")
    print(f"  Mean Reward: {results['dynasty']['mean_reward']:.2f} ± {results['dynasty']['std_reward']:.2f}")
    print(f"  Total Reward: {results['dynasty']['total_reward']:.2f}")
    print(f"  Wins: {results['dynasty']['wins']} ({results['dynasty']['win_rate']*100:.1f}%)")
    print()
    
    print(f"DRAWS: {results['draws']}")
    print("="*60)