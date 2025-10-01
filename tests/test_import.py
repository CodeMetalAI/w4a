import numpy as np
from SimulationInterface import SetRadarEnabled
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import RLEnvWrapper


def test_environment_100_steps():
    """Test that steps through the environment 100 times with detailed logging"""
    print("=" * 60)
    print("Starting 100-step environment test")
    print("=" * 60)
    
    # Create wrapped environment
    env = RLEnvWrapper(TridentIslandEnv())
    
    try:
        # Reset environment
        print("\n[RESET] Resetting environment...")
        obs, info = env.reset()
        print(f"[RESET] Initial observation shape: {obs.shape}")
        print(f"[RESET] Initial info keys: {list(info.keys())}")
        print(f"[RESET] Environment ready for stepping")
        
        # Step through environment 100 times
        for step_num in range(1, 101):
            print(f"\n--- Step {step_num:3d} ---")
            
            # Sample action
            action = env.action_space.sample()
            print(f"[ACTION] Sampled action: {action}")
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print step results
            print(f"[RESULT] Reward: {reward:.4f}")
            print(f"[RESULT] Terminated: {terminated}")
            print(f"[RESULT] Truncated: {truncated}")
            print(f"[RESULT] Observation shape: {obs.shape}")
            
            # Print additional info if available
            if 'time_elapsed' in info:
                print(f"[INFO] Time elapsed: {info['time_elapsed']:.2f}s")
            if 'current_step' in info:
                print(f"[INFO] Current step: {info['current_step']}")
            if 'entities_count' in info:
                print(f"[INFO] Entities count: {info['entities_count']}")
            
            # Check for episode end
            if terminated or truncated:
                end_reason = "TERMINATED" if terminated else "TRUNCATED"
                print(f"\n[EPISODE END] Episode ended at step {step_num} - {end_reason}")
                print(f"[EPISODE END] Final reward: {reward:.4f}")
                print(f"[EPISODE END] Final info: {info}")
                break
        else:
            print(f"\n[COMPLETE] Completed all 100 steps without episode ending")
            
    except Exception as e:
        print(f"\n[ERROR] Exception occurred during testing: {e}")
        raise
    finally:
        # Clean up
        print(f"\n[CLEANUP] Closing environment...")
        env.close()
        print(f"[CLEANUP] Environment closed successfully")
    
    print("\n" + "=" * 60)
    print("100-step environment test completed")
    print("=" * 60)


if __name__ == "__main__":
    test_environment_100_steps()