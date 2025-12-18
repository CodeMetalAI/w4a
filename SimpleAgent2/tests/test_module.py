import pytest

from simple_agent2 import (SimpleAgent2, Faction, ForceLaydown, SimulationData)

def test_create_agent():
    agent1 = SimpleAgent2(Faction.LEGACY)
    agent2 = SimpleAgent2(Faction.DYNASTY)
    assert(agent1.faction != agent2.faction)

def test_force_laydown():
    agent = SimpleAgent2(Faction.LEGACY)

    agent.start_force_laydown(ForceLaydown())
    agent.finalize_force_laydown()

    agent.pre_simulation_tick(SimulationData())

def test_pre_simulation_tick():
    agent = SimpleAgent2(Faction.LEGACY)

    agent.pre_simulation_tick(SimulationData())

def test_tick():
    agent = SimpleAgent2(Faction.LEGACY)
    
    agent.tick(SimulationData())