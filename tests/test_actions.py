"""
Basic tests to validate w4a environment works
"""

import pytest
import numpy as np
from w4a import Config
from w4a.envs.trident_island_env import TridentIslandEnv
from w4a.wrappers.wrapper import EnvWrapper
from w4a.training.evaluation import RandomAgent, evaluate
from w4a.training.replay import ReplayRecorder

from SimulationInterface import Vector3
from w4a import execute_move_action


class MockEntity:
    def __init__(self):
        self.pos = Vector3(0, 0, 0)


def test_move_action():
    """Test move action"""

    entity = MockEntity()

    entities = { 1: entity}

    action = {"move_center_grid": 10, "move_short_axis_km": 10, "move_long_axis_km": 100, "move_axis_angle": 0}

    cap = execute_move_action(1, action, entities, Config())

    assert cap
    assert len(cap.events) == 2
    assert cap.events[0].spline_points[0] == entity.pos
    assert cap.events[0].spline_points[1] == cap.events[1].spline_points[0]

    race_track = cap.events[1].spline_points

    avg = sum(race_track, Vector3(0, 0, 0)) / len(race_track)

    print(f"Average position {avg}")

    # Todo: not sure yet how to validate this any further
