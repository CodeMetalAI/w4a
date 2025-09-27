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

from SimulationInterface import (Vector3, Simulation, create_mock_entity)

from w4a import(is_valid_action)
from w4a import (execute_move_action, execute_rtb_action, execute_set_radar_focus_action, execute_stealth_action, execute_capture_action, execute_refuel_action)

from w4a import w4a_entities

def create_test_entity():
    return w4a_entities.create_mock_entity("F-35C (Air Superiority)")

def create_refueling_entity():
    return w4a_entities.create_mock_entity("KC-46")

def create_capture_entity():
    return w4a_entities.create_mock_entity("C-130")

def create_player_flag():
    return w4a_entities.create_mock_entity("LegacyFlag")

def create_neutral_flag():
    return w4a_entities.create_mock_entity("NeutralFlag")

def test_move_action():
    """Test move action"""

    entity = create_test_entity()

    entities = { 1: entity}

    action = {"entity_id": 1, "action_type": 1, "move_center_grid": 10, "move_short_axis_km": 10, "move_long_axis_km": 100, "move_axis_angle": 0}

    assert is_valid_action(action, entities, {}, Config())

    cap = execute_move_action(1, action, entities, Config())

    assert cap
    assert len(cap.events) == 2
    assert cap.events[0].spline_points[0] == entity.pos
    assert cap.events[0].spline_points[1] == cap.events[1].spline_points[0]

    race_track = cap.events[1].spline_points

    avg = sum(race_track, Vector3(0, 0, 0)) / len(race_track)

    #print(f"Average position {avg}")

    # Todo: not sure yet how to validate this any further

def test_rtb_action():
    """Test rtb action"""

    entity = create_test_entity()
    flag = create_player_flag()

    entities = { 1: entity, 2: flag}

    action = {"entity_id": 1, "action_type": 6, "flag_id": 2 }

    assert is_valid_action(action, entities, {}, Config())

    event = execute_rtb_action(1, action, entities)

    assert event
    assert event.flag == flag

def test_set_radar_focus_action():
    """Test set radar focus action"""

    entity = create_test_entity()

    entities = { 1: entity}

    action = {"entity_id": 1, "action_type": 4, "sensing_position_grid": 10 }  
    assert is_valid_action(action, entities, {}, Config())


    event = execute_set_radar_focus_action(1, action, entities, Config())

    assert event

def test_stealth_action():
    """Test set radar strength action"""

    entity = create_test_entity()

    entities = { 1: entity}

    action = {"entity_id": 1, "action_type": 3, "stealth_enabled": True }

    assert is_valid_action(action, entities, {}, Config())

    event = execute_stealth_action(1, action, entities, Config())

    assert event
    assert event.strength == 0

def test_capture_action():
    """Test capture action"""

    entity = create_capture_entity()
    flag = create_neutral_flag()

    entities = { 1: entity, 2: flag}

    action = {"entity_id": 1, "action_type": 5, "flag_id": 2 }

    assert is_valid_action(action, entities, {}, Config())

    event = execute_capture_action(1, action, entities)

    assert event
    assert event.flag == flag

def test_refuel_action():
    """Test refuel action"""

    entity = create_test_entity()
    refueling_entity = create_refueling_entity()

    entities = { 1: entity, 2: refueling_entity}

    action = {"entity_id": 1, "action_type": 7, "refuel_target_id": 2 }

    assert is_valid_action(action, entities, {}, Config())

    event = execute_refuel_action(1, action, entities)

    assert event
    assert event.component
    assert event.component.entity == entity
    assert event.refueling_entity == refueling_entity


