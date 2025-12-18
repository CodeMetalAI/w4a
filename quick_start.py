from w4a import Config
from w4a.envs import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent, SimpleAgent
from SimulationInterface import Faction

# Create environment
config = Config()
env = TridentIslandMultiAgentEnv(config=config)

# Register agent classes
env.set_agent_classes(
    lambda: CompetitionAgent(Faction.LEGACY, config),
    lambda: SimpleAgent(Faction.DYNASTY, config)
)

# Run episode (using default force laydowns from config)
observations, infos = env.reset()

# Custom force laydowns:
# observations, infos = env.reset(options={
#     "legacy_force_laydown": "path/to/custom_legacy.json",
#     "dynasty_force_laydown": "path/to/custom_dynasty.json"
# })

for step in range(1000):
    actions = {
        "legacy": env.agent_legacy.select_action(observations["legacy"]),
        "dynasty": env.agent_dynasty.select_action(observations["dynasty"])
    }

    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if terminations["legacy"] or truncations["legacy"]:
        print(f"Episode ended at step {step}: {infos['legacy']['termination_cause']}")
        observations, infos = env.reset()

env.close()
print("Test completed!")