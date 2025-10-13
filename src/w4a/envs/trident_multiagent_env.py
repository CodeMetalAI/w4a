"""
Trident Island Multi-Agent Environment

PettingZoo Parallel environment for two-player competitive tactical simulation.
Each agent controls one faction (Legacy or Dynasty) and competes to achieve
mission objectives.

This uses PettingZoo's Parallel API where both agents act simultaneously each step.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from pettingzoo import ParallelEnv
from gymnasium import spaces

from ..config import Config
from . import actions as actions_module
from . import observations
from . import simulation_utils
from . import mission_metrics
from .utils import get_time_elapsed
from .constants import FACTION_FLAG_IDS, CENTER_ISLAND_FLAG_ID

import SimulationInterface
from SimulationInterface import (
    Simulation, SimulationConfig, SimulationData, Faction, Flag,
    EntitySpawned, Victory, AdversaryContact, TargetGroup,
    Entity, ControllableEntity, Unit, EntityDomain,
    EntitySpawnData, EntityList, ForceLaydown, FactionConfiguration
)


class TridentIslandMultiAgentEnv(ParallelEnv):
    """
    PettingZoo Parallel environment for competitive tactical simulation.
    
    Two agents (Legacy and Dynasty factions) compete simultaneously to achieve
    mission objectives including capture-and-hold and force-on-force engagements.
    
    PettingZoo Parallel API:
        - Both agents act simultaneously each step
        - Returns dicts keyed by agent name: {"legacy": ..., "dynasty": ...}
        - Supports independent observation/action spaces per agent
    """
    
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "name": "trident_island_v0",
    }
    
    def __init__(
        self,
        config: Optional[Config] = None,
        render_mode: Optional[str] = None,
        enable_replay: bool = False,
        scenario_name: str = "TridentIsland"
    ):
        """
        Initialize the multi-agent tactical simulation environment.
        
        Args:
            config: Environment configuration parameters
            render_mode: Rendering mode for visualization
            enable_replay: Whether to enable replay recording
            scenario_name: Name of the tactical scenario to load
        """
        super().__init__()
        
        self.config = config or Config()
        self.render_mode = render_mode
        self.enable_replay = enable_replay
        self.scenario_name = scenario_name
        
        # PettingZoo required attributes
        self.possible_agents = ["legacy", "dynasty"]
        self.agents = self.possible_agents[:]  # Active agents (copy)
        
        self.max_ammo = 50.0  # TODO: Erwin, is this correct?

        # Set up paths
        self.scenario_path = Path(__file__).parent.parent / "scenarios"
        
        # Initialize simulation interface
        SimulationInterface.initialize()
        
        self.simulation = None
        
        # Initialize state tracking
        self.current_step = 0
        self.frame_rate = 600
        self.FrameIndex = 0
        self.time_elapsed = 0.0
        
        self.simulation_events = []
        
        # Competition agents (set via set_agents())
        self.agent_legacy = None
        self.agent_dynasty = None
        
        # Spaces (set after agents are registered)
        self._observation_spaces = None
        self._action_spaces = None
        
        # Calculate grid dimensions for action space
        map_size_km = self.config.map_size_km[0]
        self.grid_size = map_size_km // self.config.grid_resolution_km
        self.max_grid_positions = self.grid_size * self.grid_size
        
        # Calculate action space parameters
        angle_steps = 360 // self.config.angle_resolution_degrees
        patrol_steps = (
            (self.config.max_patrol_axis_km - self.config.min_patrol_axis_km) 
            // self.config.patrol_axis_increment_km + 1
        )
        
        # Build action space template (same for both agents)
        self._action_space_template = spaces.Dict({
            "action_type": spaces.Discrete(8),
            "entity_id": spaces.Discrete(self.config.max_entities),
            "move_center_grid": spaces.Discrete(self.max_grid_positions),
            "move_short_axis_km": spaces.Discrete(patrol_steps),
            "move_long_axis_km": spaces.Discrete(patrol_steps),
            "move_axis_angle": spaces.Discrete(angle_steps),
            "target_group_id": spaces.Discrete(self.config.max_target_groups),
            "weapon_selection": spaces.Discrete(self.config.max_weapon_combinations),
            "weapon_usage": spaces.Discrete(3),
            "weapon_engagement": spaces.Discrete(4),
            "stealth_enabled": spaces.Discrete(2),
            "sensing_position_grid": spaces.Discrete(self.max_grid_positions + 1),
            "refuel_target_id": spaces.Discrete(self.config.max_entities),
        })
        
        # Flags (shared resource, both agents can see)
        self.flags = {}
        
        # Set up environment event handlers
        # EntitySpawned: for shared resources like flags
        # AdversaryContact: for target detection (routed to agents)
        # Victory: for mission completion
        self.simulation_event_handlers = {
            EntitySpawned: self._on_entity_spawned,
            AdversaryContact: self._on_adversary_contact,
            Victory: self._on_victory
        }
        
        # Mission metrics (for termination and rewards)
        # Per-faction dead entity tracking (single source of truth)
        self.dead_entities_by_faction = {
            Faction.LEGACY: set(),
            Faction.DYNASTY: set()
        }
        
        # Per-faction kill tracking (cached in mission_metrics)
        self.kill_ratio_by_faction = {
            Faction.LEGACY: 0.0,
            Faction.DYNASTY: 0.0
        }
        self.casualties_by_faction = {
            Faction.LEGACY: 0,
            Faction.DYNASTY: 0
        }
        self.kills_by_faction = {
            Faction.LEGACY: 0,
            Faction.DYNASTY: 0
        }
        
        # Per-faction capture tracking (agent-agnostic multiagent design)
        self.capture_progress_by_faction = {
            Faction.LEGACY: 0.0,
            Faction.DYNASTY: 0.0
        }
        self.capture_possible_by_faction = {
            Faction.LEGACY: True,
            Faction.DYNASTY: True
        }
        self.capture_completed_at_step = {
            Faction.LEGACY: None,  # Step number when capture completed (None = not completed)
            Faction.DYNASTY: None
        }
        self.island_contested = False
        
        # Action tracking for debugging and analysis
        self.last_action_by_entity = {}  # Intent: what agent tried to do
        self.last_action_applied_by_entity = {}  # Reality: what actually happened
    
    def set_agents(self, legacy_agent, dynasty_agent):
        """
        Register the competition agents for both factions.
        
        Args:
            legacy_agent: CompetitionAgent controlling Legacy faction
            dynasty_agent: CompetitionAgent controlling Dynasty faction
        """
        self.agent_legacy = legacy_agent
        self.agent_dynasty = dynasty_agent
        
        # Link agents to environment (for observation building)
        legacy_agent._set_env(self)
        dynasty_agent._set_env(self)
        
        # Build observation and action spaces
        self._observation_spaces = {
            "legacy": observations.build_observation_space(self.config),
            "dynasty": observations.build_observation_space(self.config)
        }
        self._action_spaces = {
            "legacy": self._action_space_template,
            "dynasty": self._action_space_template
        }
    
    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """PettingZoo required: observation spaces for each agent."""
        if self._observation_spaces is None:
            raise RuntimeError("Agents must be set via set_agents() before accessing spaces")
        return self._observation_spaces
    
    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """PettingZoo required: action spaces for each agent."""
        if self._action_spaces is None:
            raise RuntimeError("Agents must be set via set_agents() before accessing spaces")
        return self._action_spaces
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducible episodes
            options: Additional reset options
            
        Returns:
            Tuple of (observations, infos) where each is a dict keyed by agent name
        """
        # Validate agents are set
        if self.agent_legacy is None or self.agent_dynasty is None:
            raise RuntimeError("Agents must be set via set_agents() before reset()")
        
        # Reset simulation
        if self.simulation:
            SimulationInterface.destroy_simulation(self.simulation)
            self.simulation = None
        
        # Reset state tracking
        self.current_step = 0
        self.FrameIndex = 0
        self.time_elapsed = 0.0
        self.simulation_events = []
        
        # Reset agents list (both active)
        self.agents = self.possible_agents[:]
        
        # Reset flags
        self.flags.clear()
        
        # Reset mission metrics
        mission_metrics.reset_mission_metrics(self)
        
        # Setup simulation with both agents
        legacy_entity_list = self.scenario_path / "force_composition" / "LegacyEntityList.json"
        dynasty_entity_list = self.scenario_path / "force_composition" / "DynastyEntityList.json"
        legacy_spawn_data = self.scenario_path / "laydown" / "LegacyEntitySpawnData.json"
        dynasty_spawn_data = self.scenario_path / "laydown" / "DynastyEntitySpawnData.json"
        
        simulation_utils.setup_simulation_from_json(
            self,
            str(legacy_entity_list),
            str(dynasty_entity_list),
            str(legacy_spawn_data),
            str(dynasty_spawn_data),
            legacy_agent=self.agent_legacy._get_sim_agent(),
            dynasty_agent=self.agent_dynasty._get_sim_agent(),
            seed=seed
        )
        
        # Get observations for both agents
        observations = {
            "legacy": self.agent_legacy.get_observation(),
            "dynasty": self.agent_dynasty.get_observation()
        }
        
        # Get info for both agents
        infos = {
            "legacy": self._build_info_for_agent(self.agent_legacy),
            "dynasty": self._build_info_for_agent(self.agent_dynasty)
        }
        
        # Set termination_cause to None on reset
        infos["legacy"]["termination_cause"] = None
        infos["dynasty"]["termination_cause"] = None
        
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, Dict]
    ) -> Tuple[
        Dict[str, Any],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminations
        Dict[str, bool],  # truncations
        Dict[str, Dict]  # infos
    ]:
        """
        Execute one environment step with actions from both agents.
        
        Args:
            actions: Dict of actions keyed by agent name
                     {"legacy": action_dict, "dynasty": action_dict}
        
        Returns:
            observations: {"legacy": obs, "dynasty": obs}
            rewards: {"legacy": reward, "dynasty": reward}
            terminations: {"legacy": terminated, "dynasty": terminated}
            truncations: {"legacy": truncated, "dynasty": truncated}
            infos: {"legacy": info_dict, "dynasty": info_dict}
        """
        self.current_step += 1
        self.FrameIndex += self.frame_rate
        self.time_elapsed = get_time_elapsed(self.FrameIndex)
        
        self.simulation_events = []
        
        # Execute actions from both agents
        action_legacy = actions.get("legacy", self._get_noop_action())
        action_dynasty = actions.get("dynasty", self._get_noop_action())
        
        # Record action intent for debugging
        self._record_last_intended_action(action_legacy, "legacy")
        self._record_last_intended_action(action_dynasty, "dynasty")
        
        player_events_legacy = actions_module.execute_action(
            action_legacy,
            self.agent_legacy._sim_agent.controllable_entities,
            self.agent_legacy._sim_agent.target_groups,
            self.flags,
            self.config
        )
        
        player_events_dynasty = actions_module.execute_action(
            action_dynasty,
            self.agent_dynasty._sim_agent.controllable_entities,
            self.agent_dynasty._sim_agent.target_groups,
            self.flags,
            self.config
        )
        
        # Combine player events and tick simulation
        sim_data = SimulationData()
        sim_data.player_events = player_events_legacy + player_events_dynasty
        
        simulation_utils.tick_simulation(self)
        
        # Get observations for both agents
        observations = {
            "legacy": self.agent_legacy.get_observation(),
            "dynasty": self.agent_dynasty.get_observation()
        }
        
        # Calculate rewards for both agents (independent but zero-sum)
        reward_legacy = self._calculate_reward_for_agent(self.agent_legacy)
        reward_dynasty = self._calculate_reward_for_agent(self.agent_dynasty)
        
        # Update mission metrics
        mission_metrics.update_all_mission_metrics(self)
        
        # Check termination (same for both agents)
        terminated = self._check_termination()
        truncated = self.time_elapsed >= self.config.max_game_time
        
        # Terminal rewards (zero-sum)
        if terminated or truncated:
            winner = self._get_winner()
            if winner == "legacy":
                reward_legacy += 100.0
                reward_dynasty -= 100.0
            elif winner == "dynasty":
                reward_legacy -= 100.0
                reward_dynasty += 100.0
            # If winner is None (draw), no terminal reward
        
        rewards = {
            "legacy": reward_legacy,
            "dynasty": reward_dynasty
        }
        
        terminations = {
            "legacy": terminated,
            "dynasty": terminated
        }
        
        truncations = {
            "legacy": truncated,
            "dynasty": truncated
        }
        
        # Determine termination cause for info
        if terminated:
            outcome = self._evaluate_outcome()
            if outcome == "legacy_win":
                termination_cause = "legacy_win"
            elif outcome == "dynasty_win":
                termination_cause = "dynasty_win"
            elif outcome == "draw":
                termination_cause = "draw"
            else:
                termination_cause = "terminated"  # Fallback (shouldn't happen)
        elif truncated:
            termination_cause = "time_limit"
        else:
            termination_cause = None
        
        # Get info for both agents
        infos = {
            "legacy": self._build_info_for_agent(self.agent_legacy),
            "dynasty": self._build_info_for_agent(self.agent_dynasty)
        }
        
        # Add termination cause to both agents' info
        infos["legacy"]["termination_cause"] = termination_cause
        infos["dynasty"]["termination_cause"] = termination_cause
        
        return observations, rewards, terminations, truncations, infos
    
    def _calculate_reward_for_agent(self, agent) -> float:
        """
        Calculate per-step reward for a specific agent.
        
        Calls agent's calculate_reward() method. Default returns 0.0.
        Users can override to implement custom reward shaping.
        
        Args:
            agent: CompetitionAgent to calculate reward for
            
        Returns:
            Per-step reward (float)
        """
        return agent.calculate_reward(self)
    
    def _check_termination(self) -> bool:
        """
        Check if the mission should terminate early.
        
        Returns:
            True if episode should end due to win/loss conditions (including draws)
        """
        outcome = self._evaluate_outcome()
        return outcome in ("legacy_win", "dynasty_win", "draw")
    
    def _evaluate_outcome(self) -> str:
        """
        Evaluate current mission outcome.
        
        This method checks multiple win/loss conditions for both factions:
        1. Capture win: Holding the objective for required time (per-faction)
        2. Kill ratio win: Achieving favorable kill:death ratio
        3. Kill ratio loss: Suffering unfavorable kill:death ratio
        
        Returns:
            "legacy_win", "dynasty_win", "draw", or "ongoing"
        """
        # Capture win condition (per-faction, agent-agnostic)
        legacy_capture_progress = self.capture_progress_by_faction[Faction.LEGACY]
        dynasty_capture_progress = self.capture_progress_by_faction[Faction.DYNASTY]
        legacy_can_capture = self.capture_possible_by_faction[Faction.LEGACY]
        dynasty_can_capture = self.capture_possible_by_faction[Faction.DYNASTY]
        
        # Check if factions have completed capture
        legacy_captured = legacy_capture_progress >= self.config.capture_required_seconds and legacy_can_capture
        dynasty_captured = dynasty_capture_progress >= self.config.capture_required_seconds and dynasty_can_capture
        
        # Determine winner based on who completed FIRST (tracked by step number)
        if legacy_captured and dynasty_captured:
            # Both completed - check who finished first
            legacy_step = self.capture_completed_at_step[Faction.LEGACY]
            dynasty_step = self.capture_completed_at_step[Faction.DYNASTY]
            
            if legacy_step is not None and dynasty_step is not None:
                if legacy_step < dynasty_step:
                    return "legacy_win"  # Legacy completed first
                elif dynasty_step < legacy_step:
                    return "dynasty_win"  # Dynasty completed first
                else:
                    return "draw"  # Completed at exact same step (rare)
            # Fallback if completion steps not tracked properly
            return "draw"
        elif legacy_captured:
            return "legacy_win"
        elif dynasty_captured:
            return "dynasty_win"
        
        # Get cached kill ratios (computed in mission_metrics.update_kill_ratios)
        legacy_kill_ratio = self.kill_ratio_by_faction[Faction.LEGACY]
        dynasty_kill_ratio = self.kill_ratio_by_faction[Faction.DYNASTY]
        
        threshold = self.config.kill_ratio_threshold
        
        # Check win conditions (kill ratio threshold met)
        if legacy_kill_ratio >= threshold:
            return "legacy_win"
        if dynasty_kill_ratio >= threshold:
            return "dynasty_win"
        
        # Check loss conditions (inverse kill ratio + no capture possible)
        inverse_threshold = 1.0 / max(1e-6, threshold)
        if legacy_kill_ratio <= inverse_threshold and not legacy_can_capture:
            return "dynasty_win"
        if dynasty_kill_ratio <= inverse_threshold and not dynasty_can_capture:
            return "legacy_win"
        
        return "ongoing"
    
    def _get_winner(self) -> Optional[str]:
        """
        Determine the winner of the mission.
        
        Returns:
            "legacy", "dynasty", or None (draw/ongoing)
        """
        outcome = self._evaluate_outcome()
        if outcome == "legacy_win":
            return "legacy"
        elif outcome == "dynasty_win":
            return "dynasty"
        return None
    
    def _build_info_for_agent(self, agent) -> Dict:
        """
        Build info dict for a specific agent.
        
        Args:
            agent: CompetitionAgent to build info for
            
        Returns:
            Info dict with agent-specific and shared information
        """
        # Determine which faction this agent controls
        faction_name = agent.faction.name
        agent_name = faction_name.lower()
        
        # Get all entities controlled by this agent
        controllable = set(agent._sim_agent.controllable_entities.keys())
        detected = set(agent._sim_agent.target_groups.keys())
        
        # Count total entities for this agent (includes non-controllable like carriers)
        total_entities = len(agent._sim_agent.controllable_entities)
        
        return {
            'step': self.current_step,
            'time_elapsed': self.time_elapsed,
            'time_remaining': max(0.0, self.config.max_game_time - self.time_elapsed),
            'faction': faction_name,
            
            # Entity counts
            'total_entities': total_entities,
            'my_entities_count': len(agent.get_entities()),  # Alive and controllable
            'detected_targets_count': len(detected),
            'last_events_count': len(self.simulation_events),
            
            # Masks for THIS agent only
            'valid_masks': {
                'action_types': self._get_valid_action_types(agent),
                'controllable_entities': controllable,
                'visible_targets': detected,
                'entity_target_matrix': self._get_entity_target_engagement_matrix(agent)
            },
            
            'controllable_entities': list(controllable),
            
            # Refuel info for THIS agent
            'refuel': {
                'receivers': list(self._get_refuel_receivers_set(agent)),
                'providers': list(self._get_refuel_providers_set(agent))
            },
            
            # Action tracking for debugging (filtered to this agent's actions)
            'last_action_intent_by_entity': {
                k: v for k, v in self.last_action_by_entity.items() 
                if v.get("agent") == agent_name
            },
            'last_action_applied_by_entity': {
                k: v for k, v in self.last_action_applied_by_entity.items() 
                if k.startswith(f"{agent_name}_")
            },
            
            # Mission progress (faction-relative)
            'mission': {
                'my_casualties': self._get_casualties_count_for_faction(agent.faction),
                'enemy_casualties': self._get_casualties_count_for_enemy(agent.faction),
                'kill_ratio': self._compute_kill_ratio_for_faction(agent.faction),
                'my_capture_progress': float(self.capture_progress_by_faction[agent.faction]),
                'my_capture_possible': bool(self.capture_possible_by_faction[agent.faction]),
                'enemy_capture_progress': float(self._get_enemy_capture_progress(agent.faction)),
                'enemy_capture_possible': bool(self._get_enemy_capture_possible(agent.faction)),
                'island_contested': bool(self.island_contested)
            }
        }
    
    def _get_valid_action_types(self, agent) -> set:
        """Get set of valid action types based on agent's entity capabilities.
        
        Args:
            agent: CompetitionAgent instance
        
        Returns:
            Set of valid action type indices
        """
        valid_actions = {0}  # noop always valid
        
        for entity in agent._sim_agent.controllable_entities.values():
            if not self._is_controllable_entity_for_agent(entity, agent):
                continue
            
            if entity.domain == EntityDomain.AIR:
                valid_actions.update({1, 6})  # move, rtb
            if self._entity_can_engage_for_agent(entity, agent):
                valid_actions.add(2)  # engage
            if self._entity_can_capture(entity):
                valid_actions.add(5)  # capture
            if entity.has_radar:
                valid_actions.update({3, 4})  # stealth, sense
            if entity.can_refuel:
                valid_actions.add(7)  # refuel
        
        return valid_actions
    
    def _get_entity_target_engagement_matrix(self, agent) -> dict:
        """Get matrix of which entities can engage which target groups for an agent."""
        engagement_matrix = {}
        
        for entity_id, entity in agent._sim_agent.controllable_entities.items():
            if not entity.is_alive:
                continue
            
            valid_targets = set()
            for tg_id, target_group in agent._sim_agent.target_groups.items():
                if target_group.faction == agent.faction:
                    continue  # Skip friendly targets
                
                available_weapons = entity.select_weapons(target_group, False)
                if len(available_weapons) > 0:
                    valid_targets.add(tg_id)
            
            engagement_matrix[entity_id] = valid_targets
        
        return engagement_matrix
    
    def _get_refuel_receivers_set(self, agent) -> set:
        """Get entities that can receive fuel for an agent."""
        receivers = set()
        for entity_id, entity in agent._sim_agent.controllable_entities.items():
            if entity.is_alive and entity.can_refuel:
                receivers.add(entity_id)
        return receivers
    
    def _get_refuel_providers_set(self, agent) -> set:
        """Get entities that can provide fuel for an agent."""
        providers = set()
        for entity_id, entity in agent._sim_agent.controllable_entities.items():
            if entity.is_alive and entity.can_refuel_others:
                providers.add(entity_id)
        return providers
    
    def _get_casualties_count_for_faction(self, faction: Faction) -> int:
        """Get casualty count for a specific faction (cached in mission_metrics)."""
        return self.casualties_by_faction[faction]
    
    def _get_casualties_count_for_enemy(self, faction: Faction) -> int:
        """Get enemy casualty count from a faction's perspective (cached in mission_metrics)."""
        return self.kills_by_faction[faction]
    
    def _compute_kill_ratio_for_faction(self, faction: Faction) -> float:
        """Get kill ratio for a specific faction (cached in mission_metrics)."""
        return self.kill_ratio_by_faction[faction]
    
    def _get_noop_action(self) -> Dict:
        """Get a no-op action."""
        return {
            'action_type': 0,
            'entity_id': 0,
            'move_center_grid': 0,
            'move_short_axis_km': 0,
            'move_long_axis_km': 0,
            'move_axis_angle': 0,
            'target_group_id': 0,
            'weapon_selection': 0,
            'weapon_usage': 0,
            'weapon_engagement': 0,
            'stealth_enabled': 0,
            'sensing_position_grid': 0,
            'refuel_target_id': 0,
        }
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}, Time: {self.time_elapsed:.1f}s")
    
    def get_simulation_handle(self):
        """Provide access to simulation handle for replay recording.
        
        Returns:
            Simulation handle if replay is enabled, None otherwise
        """
        return self.simulation if self.enable_replay else None
    
    def _on_entity_spawned(self, event):
        """
        Handle EntitySpawned events.
        
        Routes to both agents for entity tracking, and handles shared resources like flags.
        """
        entity = event.entity
        
        # Track flags as shared resource
        if isinstance(entity, Flag):
            self.flags[FACTION_FLAG_IDS[entity.faction]] = entity
        
        # # Route to both agents for entity tracking
        # if self.agent_legacy:
        #     self.agent_legacy._sim_agent._on_entity_spawned(event)
        # if self.agent_dynasty:
        #     self.agent_dynasty._sim_agent._on_entity_spawned(event)
    
    def _on_adversary_contact(self, event):
        """
        Handle AdversaryContact events for target detection.
        
        Routes to both agents so each can track enemy target groups.
        """
        # Route to both agents for target group tracking
        # if self.agent_legacy:
        #     self.agent_legacy._sim_agent._on_adversary_contact(event)
        # if self.agent_dynasty:
        #     self.agent_dynasty._sim_agent._on_adversary_contact(event)
    
    def _on_victory(self, event):
        """
        Handle Victory events.
        
        Routes to both agents and environment for mission completion tracking.
        """
        # Route to both agents
        # if self.agent_legacy:
        #     self.agent_legacy._sim_agent._on_victory(event)
        # if self.agent_dynasty:
        #     self.agent_dynasty._sim_agent._on_victory(event)
    
    def _record_last_intended_action(self, action: Dict, agent_name: str) -> None:
        """Record last action issued (pre-validation) per entity for debugging and analysis.
        
        Args:
            action: Action dictionary that was intended to be executed
            agent_name: Name of agent issuing action ("legacy" or "dynasty")
        """
        entity_id = action["entity_id"]
        # Key by both faction and entity_id for clarity
        action_key = f"{agent_name}_entity_{entity_id}"
        self.last_action_by_entity[action_key] = {
            "agent": agent_name,
            "action_type": int(action["action_type"]),
            "entity_id": int(entity_id),
            "time": float(self.time_elapsed)
        }
    
    def _is_controllable_entity_for_agent(self, entity, agent) -> bool:
        """Check if entity is controllable by given agent.
        
        Args:
            entity: Entity to check
            agent: CompetitionAgent instance
        
        Returns:
            True if entity can be controlled by the agent
        """
        if not isinstance(entity, ControllableEntity):
            return False
        
        return entity.is_alive and entity.faction == agent.faction
    
    def _entity_can_engage_for_agent(self, entity, agent) -> bool:
        """Check if entity can engage targets for given agent.
        
        Args:
            entity: Entity to check
            agent: CompetitionAgent instance
        
        Returns:
            True if entity has weapons and valid targets exist
        """
        # Check if entity has weapons
        has_weapons = entity.target_domains != 0
        if not has_weapons:
            return False
        
        # Check if any detected targets are valid for this entity
        for target_group in agent._sim_agent.target_groups.values():
            is_enemy = target_group.faction != agent.faction
            if is_enemy:
                available_weapons = entity.select_weapons(target_group, False)
                if len(available_weapons) > 0:
                    return True
        return False
    
    def _entity_can_capture(self, entity) -> bool:
        """Check if entity can capture objectives.
        
        Args:
            entity: Entity to check
        
        Returns:
            True if entity is capable of capturing objectives
        """
        return entity.can_capture
    
    def _get_enemy_capture_progress(self, my_faction: Faction) -> float:
        """Get enemy's capture progress.
        
        Args:
            my_faction: Faction of the agent requesting info
        
        Returns:
            Enemy faction's capture progress
        """
        enemy_faction = Faction.DYNASTY if my_faction == Faction.LEGACY else Faction.LEGACY
        return self.capture_progress_by_faction[enemy_faction]
    
    def _get_enemy_capture_possible(self, my_faction: Faction) -> bool:
        """Get whether enemy can still capture.
        
        Args:
            my_faction: Faction of the agent requesting info
        
        Returns:
            True if enemy faction can still capture objectives
        """
        enemy_faction = Faction.DYNASTY if my_faction == Faction.LEGACY else Faction.LEGACY
        return self.capture_possible_by_faction[enemy_faction]
    
    def close(self):
        """Clean up environment resources."""
        if self.simulation:
            SimulationInterface.destroy_simulation(self.simulation)
            self.simulation = None

