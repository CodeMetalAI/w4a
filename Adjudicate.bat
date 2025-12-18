@echo off
set legacy_dependency_location=git+https://github.com/NEORONIN-INTERACTIVE/TestingAgent
set legacy_agent_package=TestingAgent
set legacy_agent_module=testing_agent
set legacy_agent_class=TestingAgent
set legacy_force_laydown_path=../W4A_ForceLaydown_Legacy.json

set dynasty_dependency_location=SimpleAgent
set dynasty_agent_package=w4a
set dynasty_agent_module=w4a.agents.simple_agent
set dynasty_agent_class=SimpleAgent
set dynasty_force_laydown_path=../W4A_ForceLaydown_Dynasty.json

call python3 -m pip install %legacy_dependency_location%
call python3 -m pip install %dynasty_dependency_location%

call python3 adjudicate.py --legacy_agent_package=%legacy_agent_package% --legacy_agent_module=%legacy_agent_module% --legacy_agent_class=%legacy_agent_class% --legacy_force_laydown_path=%legacy_force_laydown_path% --dynasty_agent_package=%dynasty_agent_package% --dynasty_agent_module=%dynasty_agent_module% --dynasty_agent_class=%dynasty_agent_class% --dynasty_force_laydown_path=%dynasty_force_laydown_path% --replay_path=../Replay.json  --outcome_path=../Outcome.txt