from argparse import ArgumentParser
from importlib import import_module

from pathlib import Path

from w4a import Config
from w4a.envs import TridentIslandMultiAgentEnv
from SimulationInterface import Faction

def try_adjudicate(adjudication_config):
    if adjudication_config.legacy_agent_class == None and adjudication_config.dynasty_agent_class == None:
        output_outcome(adjudication_config, Faction.NEUTRAL)
        return
        
    elif adjudication_config.legacy_agent_class == None:
        output_outcome(adjudication_config, Faction.DYNASTY)
        return

    elif adjudication_config.dynasty_agent_class == None:
        output_outcome(adjudication_config, Faction.LEGACY)
        return
    
    adjudicate(adjudication_config)

def adjudicate(adjudication_config):
    # Create environment
    config = Config()
    config.legacy_force_laydown_path = adjudication_config.legacy_force_laydown_path
    config.dynasty_force_laydown_path = adjudication_config.dynasty_force_laydown_path

    if adjudication_config.random_seed:
        config.seed = adjudication_config.random_seed

    env = TridentIslandMultiAgentEnv(config=config, enable_replay=True)

    # Register agent classes
    env.set_agent_classes(
        lambda: adjudication_config.legacy_agent_class(Faction.LEGACY, config),
        lambda: adjudication_config.dynasty_agent_class(Faction.DYNASTY, config)
    )

    # Run episode
    observations, infos = env.reset()

    for step in range(1000):
        actions = {
            "legacy": env.agent_legacy.select_action(observations["legacy"]),
            "dynasty": env.agent_dynasty.select_action(observations["dynasty"])
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if terminations["legacy"] or truncations["legacy"]:
            print(f"Episode ended at step {step}: {infos['legacy']['termination_cause']}")
            
            break

    replay = env.export_replay()
    replay_path = Path("adjudication_config.replay_path")
    replay_path.parent.mkdir(exist_ok=True)

    simulation_json = env.export_replay()

    with open(adjudication_config.replay_path, 'w') as f:
        f.write(simulation_json)

    output_outcome(adjudication_config, env.winning_faction)

    env.close()

def output_outcome(adjudication_config, winning_faction):
    print(f"Match was won by {winning_faction}")

    with open(adjudication_config.outcome_path, 'w') as f:
        f.write(winning_faction.name)

class AdjudicationConfig:
    def __init__(self):
        self.random_seed = None
        self.legacy_agent_class = None
        self.legacy_force_laydown_path = None
        self.dynasty_agent_class = None
        self.dynasty_force_laydown_path = None

        self.replay_path = None
        self.outcome_path = None
        self.log_path = None

def import_agent_class(package_name, module_name, class_name):
    try:
        module = import_module(module_name, package = package_name)
    except:
        print(F"Error trying to import module {module_name} from package {package_name}")

        return None

    agent_class = getattr(module, class_name)

    assert agent_class

    return agent_class

def create_config(args):
    # Set up adjudication
    adjudication_config = AdjudicationConfig()
    adjudication_config.random_seed = args.random_seed
    adjudication_config.legacy_agent_class = import_agent_class(args.legacy_agent_package, args.legacy_agent_module, args.legacy_agent_class)
    adjudication_config.legacy_force_laydown_path = Path(args.legacy_force_laydown_path)
    adjudication_config.dynasty_agent_class = import_agent_class(args.dynasty_agent_package, args.dynasty_agent_module, args.dynasty_agent_class)
    adjudication_config.dynasty_force_laydown_path = Path(args.dynasty_force_laydown_path)
    adjudication_config.replay_path = args.replay_path
    adjudication_config.outcome_path = args.outcome_path
    adjudication_config.log_path = args.log_path

    return adjudication_config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--legacy_agent_package", type=str, required=True)
    parser.add_argument("--legacy_agent_module", type=str, required=True)
    parser.add_argument("--legacy_agent_class", type=str, required=True)
    parser.add_argument("--legacy_force_laydown_path", type=str, required=True)
    parser.add_argument("--dynasty_agent_package", type=str, required=True)
    parser.add_argument("--dynasty_agent_module", type=str, required=True)
    parser.add_argument("--dynasty_agent_class", type=str, required=True)
    parser.add_argument("--dynasty_force_laydown_path", type=str, required=True)
    parser.add_argument("--replay_path", type=str, required=True)
    parser.add_argument("--outcome_path", type=str, required=True)
    parser.add_argument("--log_path", type=str) # Not currently in use
    parser.add_argument("--random_seed", type=int)

    args = parser.parse_args()

    adjudication_config = create_config(args)

    try_adjudicate(adjudication_config)