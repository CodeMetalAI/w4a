@echo off
set legacy_dependency_location=git+https://github.com/NEORONIN-INTERACTIVE/TestingAgent
set legacy_agent_package=TestingAgent
set legacy_agent_module=testing_agent
set legacy_agent_class=TestingAgent

set dynasty_dependency_location=./SimpleAgent
set dynasty_agent_package=SimpleAgent
set dynasty_agent_module=SimpleAgent
set dynasty_agent_class=SimpleAgent

call python3 -m pip install %legacy_dependency_location%
call python3 -m pip install %dynasty_dependency_location%

call python3 adjudicate.py --legacy_agent_package=%legacy_agent_package% --legacy_agent_module=%legacy_agent_module% --legacy_agent_class=%legacy_agent_class% --dynasty_agent_package=%dynasty_agent_package% --dynasty_agent_module=%dynasty_agent_module% --dynasty_agent_class=%dynasty_agent_class% --replay_path=../Replay.json  --outcome_path=../Outcome.txt