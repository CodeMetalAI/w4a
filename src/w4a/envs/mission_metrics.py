"""
Mission Metrics Management

This module provides functions to track and update mission progress metrics including
kill counts, capture progress, contestation status, and capture capability. These
metrics represent the current tactical situation and progress toward mission objectives.

The metrics are updated each simulation step and used for reward calculation,
termination conditions, and agent observations.
"""

from typing import Dict, List, Set, Any

from SimulationInterface import Entity, EntityDomain, Faction

from .utils import *


def update_dead_entities(env: Any) -> None:
    """Track dead entities per faction.

    Scans all entities from both agents and adds newly dead units to the 
    appropriate faction-specific set. Uses dead_entities_by_faction as the
    single source of truth for kill tracking.
    
    Args:
        env: Environment instance with agents and kill counters
    """
    # Check if entity is already tracked in any faction set
    def is_already_tracked(entity, env):
        for faction_set in env.dead_entities_by_faction.values():
            if entity in faction_set:
                return True
        return False
    
    # Iterate over entities from both agents
    for entity in env.agent_legacy._sim_agent.controllable_entities.values():
        if not entity.is_alive and not is_already_tracked(entity, env):
            # Add directly to per-faction set (single source of truth)
            if entity.faction not in env.dead_entities_by_faction:
                env.dead_entities_by_faction[entity.faction] = set()
            env.dead_entities_by_faction[entity.faction].add(entity)
    
    for entity in env.agent_dynasty._sim_agent.controllable_entities.values():
        if not entity.is_alive and not is_already_tracked(entity, env):
            # Add directly to per-faction set (single source of truth)
            if entity.faction not in env.dead_entities_by_faction:
                env.dead_entities_by_faction[entity.faction] = set()
            env.dead_entities_by_faction[entity.faction].add(entity)


def update_capture_progress(env: Any) -> None:
    """Update capture timer progress based on current capture conditions.
    
    Advances capture progress per-faction when that faction has settlers actively
    capturing. Progress resets if faction loses all settlers. Multiple factions can
    progress simultaneously. Tracks the step when each faction
    first completes capture to determine winner in ties.
    
    Args:
        env: Environment instance with per-faction capture state
    """
    # Calculate time delta from simulation parameters
    time_delta = env.frame_rate / 60.0  # frame_rate frames per step / 60 frames per second
    required_time = env.config.capture_required_seconds
    
    # Update progress for each faction independently
    for faction in [Faction.LEGACY, Faction.DYNASTY]:
        if env.capture_possible_by_faction[faction]:
            # Faction has settlers - advance their capture progress
            old_progress = env.capture_progress_by_faction[faction]
            new_progress = old_progress + time_delta
            env.capture_progress_by_faction[faction] = min(new_progress, required_time)

            # @Sanjna: on the flag: can_be_captured, capturing_faction, is_captured, is_being_captured, capture_progress

            
            # Track when faction first completes capture (crosses threshold)
            if old_progress < required_time and new_progress >= required_time:
                # First time crossing threshold - record the step
                if env.capture_completed_at_step[faction] is None:
                    env.capture_completed_at_step[faction] = env.current_step
        else:
            # Faction has no settlers - reset EVERYTHING
            # This handles settlers leaving/dying: they must start over from scratch
            env.capture_progress_by_faction[faction] = 0.0
            env.capture_completed_at_step[faction] = None  # Clear completion timestamp

def update_island_contested(env: Any) -> None:
    """Check if the capture area is contested by enemy forces.
    
    The area is considered contested when both factions have settlers
    present and capable of capturing the objective.
    
    Args:
        env: Environment instance with contestation flag
    """
    # Island is contested if BOTH factions have settlers capable of capturing
    legacy_has_settlers = env.capture_possible_by_faction[Faction.LEGACY]
    dynasty_has_settlers = env.capture_possible_by_faction[Faction.DYNASTY]
    
    env.island_contested = legacy_has_settlers and dynasty_has_settlers




def update_capture_possible(env: Any) -> None:
    """Check if capture is currently possible for each faction.

    Capture is possible per faction when at least one settler unit from that
    faction is alive and capable of capturing the objective.
    
    Args:
        env: Environment instance with per-faction capture_possible flags
    """
    # Check Legacy faction for settler units
    legacy_can_capture = False
    for entity in env.agent_legacy._sim_agent.controllable_entities.values():
        if entity.is_alive and entity.can_capture:
            legacy_can_capture = True
            break
    env.capture_possible_by_faction[Faction.LEGACY] = legacy_can_capture
    
    # Check Dynasty faction for settler units
    dynasty_can_capture = False
    for entity in env.agent_dynasty._sim_agent.controllable_entities.values():
        if entity.is_alive and entity.can_capture:
            dynasty_can_capture = True
            break
    env.capture_possible_by_faction[Faction.DYNASTY] = dynasty_can_capture


def update_kill_ratios(env: Any) -> None:
    """Update kill ratios for both factions.
    
    Calculates and caches kill ratios (enemy kills / own casualties) for each
    faction. These are used for win/loss conditions and observations.
    
    Kill ratio = (enemy units killed) / (own units lost)
    
    Args:
        env: Environment instance with kill tracking
    """
    # Count casualties per faction directly from per-faction sets (no filtering!)
    legacy_casualties = len(env.dead_entities_by_faction.get(Faction.LEGACY, set()))
    dynasty_casualties = len(env.dead_entities_by_faction.get(Faction.DYNASTY, set()))
    
    # Kills are the enemy's casualties
    legacy_kills = dynasty_casualties  # Legacy killed Dynasty units
    dynasty_kills = legacy_casualties  # Dynasty killed Legacy units
    
    # Calculate kill ratios (avoid division by zero)
    env.kill_ratio_by_faction[Faction.LEGACY] = float(legacy_kills) / max(1, legacy_casualties)
    env.kill_ratio_by_faction[Faction.DYNASTY] = float(dynasty_kills) / max(1, dynasty_casualties)
    
    # Cache the individual counts for observations
    env.casualties_by_faction[Faction.LEGACY] = legacy_casualties
    env.casualties_by_faction[Faction.DYNASTY] = dynasty_casualties
    env.kills_by_faction[Faction.LEGACY] = legacy_kills
    env.kills_by_faction[Faction.DYNASTY] = dynasty_kills



def update_all_mission_metrics(env: Any) -> None:
    """Update all mission metrics in the correct order.
    
    This function should be called once per simulation step to maintain
    consistent mission progress tracking across all metrics.
    
    Args:
        env: Environment instance
    """
    # Order matters: some functions depend on others
    update_dead_entities(env)
    update_kill_ratios(env)           # Depends on: dead_entities_by_faction
    update_capture_possible(env)       # Depends on: alive entities
    update_island_contested(env)       # Depends on: capture_possible_by_faction
    update_capture_progress(env)       # Depends on: capture_possible_by_faction, island_contested


def reset_mission_metrics(env: Any) -> None:
    """Reset all mission metrics to initial values.
    
    This function should be called during environment reset to initialize
    fresh mission tracking for a new episode.
    
    Args:
        env: Environment instance
    """
    # Reset per-faction dead entity tracking (single source of truth)
    env.dead_entities_by_faction = {
        Faction.LEGACY: set(),
        Faction.DYNASTY: set()
    }
    
    # Reset per-faction kill tracking (derived from dead_entities_by_faction)
    env.kill_ratio_by_faction = {
        Faction.LEGACY: 0.0,
        Faction.DYNASTY: 0.0
    }
    env.casualties_by_faction = {
        Faction.LEGACY: 0,
        Faction.DYNASTY: 0
    }
    env.kills_by_faction = {
        Faction.LEGACY: 0,
        Faction.DYNASTY: 0
    }
    
    # Reset per-faction capture tracking
    env.capture_progress_by_faction = {
        Faction.LEGACY: 0.0,
        Faction.DYNASTY: 0.0
    }
    env.capture_possible_by_faction = {
        Faction.LEGACY: True,
        Faction.DYNASTY: True
    }
    env.capture_completed_at_step = {
        Faction.LEGACY: None,
        Faction.DYNASTY: None
    }
    env.island_contested = False
