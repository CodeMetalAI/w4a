"""
Mission Metrics Management

This module provides functions to track and update mission progress metrics including
kill counts, capture progress, and capture capability. These
metrics represent the current tactical situation and progress toward mission objectives.

The metrics are updated each simulation step and used for reward calculation,
termination conditions, and agent observations.
"""

from typing import Dict, List, Set, Any

from SimulationInterface import Entity, PlatformDomain, ProjectileDomain, Faction

from .utils import *
from .constants import CENTER_ISLAND_FLAG_ID


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
    
    Uses the flag's capture_progress property (0..1) and capturing_faction to track
    which faction is currently making progress. Only one faction can capture at a time.
    Tracks the step when each faction first completes capture.
    
    Args:
        env: Environment instance with per-faction capture state
    """
    flag = env.flags[CENTER_ISLAND_FLAG_ID]
    required_time = env.config.capture_required_seconds
    
    # Get current capture state from flag
    # flag.capture_progress is normalized 0..1 (0 = not started, 1 = complete)
    current_progress_normalized = flag.capture_progress  # 0..1
    current_progress_seconds = current_progress_normalized * required_time  # Convert to seconds
    
    # Only the faction currently capturing the flag has progress
    # All other factions have 0 progress
    capturing_faction = flag.capturing_faction

    for faction in [Faction.LEGACY, Faction.DYNASTY]:
        old_progress = env.capture_progress_by_faction[faction]
        
        if capturing_faction == faction:
            # This faction is actively capturing
            new_progress = current_progress_seconds
        else:
            # This faction is not capturing - no progress
            new_progress = 0.0
        
        env.capture_progress_by_faction[faction] = new_progress
        
        # Track when faction first completes capture (crosses threshold)
        if old_progress < required_time and new_progress >= required_time:
            # First time crossing threshold - record the step
            if env.capture_completed_at_step[faction] is None:
                env.capture_completed_at_step[faction] = env.current_step

def update_capture_possible(env: Any) -> None:
    """Check if capture is currently possible for each faction.

    Capture is possible for a faction when:
    1. The faction has at least one settler unit alive and capable of capturing
    2. The flag itself can be captured (neutral, not already captured)
    
    Args:
        env: Environment instance with per-faction capture_possible flags
    """
    flag = env.flags[CENTER_ISLAND_FLAG_ID]
    flag_can_be_captured = flag.can_be_captured
    
    # Check Legacy faction for settler units
    legacy_has_settlers = False
    for entity in env.agent_legacy._sim_agent.controllable_entities.values():
        if entity.is_alive and entity.can_capture:
            legacy_has_settlers = True
            break
    env.capture_possible_by_faction[Faction.LEGACY] = legacy_has_settlers and flag_can_be_captured
    
    # Check Dynasty faction for settler units
    dynasty_has_settlers = False
    for entity in env.agent_dynasty._sim_agent.controllable_entities.values():
        if entity.is_alive and entity.can_capture:
            dynasty_has_settlers = True
            break
    env.capture_possible_by_faction[Faction.DYNASTY] = dynasty_has_settlers and flag_can_be_captured


def update_casualty_counts(env: Any) -> None:
    """Update casualty and kill counts for both factions.
    
    Calculates and caches casualties and kills for each faction.
    These are used for observations.
    
    Args:
        env: Environment instance with kill tracking
    """
    # Count casualties per faction directly from per-faction sets (no filtering!)
    legacy_casualties = len(env.dead_entities_by_faction.get(Faction.LEGACY, set()))
    dynasty_casualties = len(env.dead_entities_by_faction.get(Faction.DYNASTY, set()))

    # Kills are the enemy's casualties
    legacy_kills = dynasty_casualties  # Legacy killed Dynasty units
    dynasty_kills = legacy_casualties  # Dynasty killed Legacy units
    
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
    update_casualty_counts(env)        # Depends on: dead_entities_by_faction
    update_capture_possible(env)       # Depends on: alive entities
    update_capture_progress(env)       # Depends on: capture_possible_by_faction


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
    
    # Reset per-faction casualty tracking (derived from dead_entities_by_faction)
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
