from argparse import ArgumentParser
from importlib import import_module

from w4a import Config
from w4a.envs import TridentIslandMultiAgentEnv
from w4a.agents import CompetitionAgent#, SimpleAgent
from SimulationInterface import Faction

def import_agent_class(class_name):
    module_name = class_name
    package = class_name
    class_name = class_name

    module = import_module(module_name, package = package)

    assert module

    agent_class = getattr(module, class_name)

    assert agent_class

    return agent_class

def adjudicate(adjudication_config):
    # Create environment
    config = Config()

    if hasattr(adjudication_config, "random_seed"):
        config.seed = adjudication_config.random_seed

    env = TridentIslandMultiAgentEnv(config=config)

    # Register agent classes
    env.set_agent_classes(
        lambda: adjudication_config.legacy_agent_class(Faction.LEGACY, config),
        lambda: adjudication_config.dynasty_agent_class(Faction.DYNASTY, config)
    )

    # Run episode
    observations, infos = env.reset()

    for step in range(10):
        actions = {
            "legacy": env.agent_legacy.select_action(observations["legacy"]),
            "dynasty": env.agent_dynasty.select_action(observations["dynasty"])
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if terminations["legacy"] or truncations["legacy"]:
            print(f"Episode ended at step {step}: {infos['legacy']['termination_cause']}")
            observations, infos = env.reset()

    env.close()

    print("Adjudication completed!")

class AdjudicationConfig:
    def __init__(self):
        self.random_seed = None
        self.legacy_agent_class = None
        self.legacy_entity_force_laydown_path = None
        self.dynasty_agent_class = None
        self.dynasty_entity_force_laydown_path = None

        self.replay_filename = None
        self.log_filename = None

def create_config(args):
    # Set up adjudication
    adjudication_config = AdjudicationConfig()
    adjudication_config.random_seed = args.random_seed
    adjudication_config.legacy_agent_class = import_agent_class(args.legacy_agent_module)
    adjudication_config.legacy_entity_force_laydown_path = args.legacy_entity_force_laydown_path
    adjudication_config.dynasty_agent_class = import_agent_class(args.dynasty_agent_module)
    adjudication_config.legacy_entity_force_laydown_path = args.legacy_entity_force_laydown_path
    adjudication_config.replay_filename = args.replay_filename
    adjudication_config.log_filename = args.log_filename

    return adjudication_config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--legacy_agent_module", type=str, required=True)
    parser.add_argument("--legacy_entity_force_laydown_path", type=str)
    parser.add_argument("--dynasty_agent_module", type=str, required=True)
    parser.add_argument("--dynasty_entity_force_laydown_path", type=str)
    parser.add_argument("--replay_filename", type=str, required=True)
    parser.add_argument("--log_filename", type=str) # Not currently in use
    parser.add_argument("--random_seed", type=int)

    args = parser.parse_args()

    adjudication_config = create_config(args)
    
    adjudicate(adjudication_config)