"""
TridentIslandEnv

Single-agent Gymnasium environment for tactical simulation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from .simple_agent import SimpleAgent

from ..config import Config
from . import actions
from ..entities import w4a_entities
from . import simulation_utils
from . import observations
from . import mission_metrics
from .utils import get_time_elapsed

from pathlib import Path

import SimulationInterface
from SimulationInterface import (
    Simulation, SimulationConfig, SimulationData, Faction,
    EntitySpawned, Victory, AdversaryContact,
    Entity, ControllableEntity, Unit, EntityDomain,
    EntitySpawnData, EntityList, ForceLaydown, FactionConfiguration
)


class TridentIslandEnv(gym.Env):
    """
    Single-agent tactical environment using BANE simulation engine.
    
    """
    
    metadata = {"render_modes": ["rgb_array", "human"]}
    
    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None, 
                 enable_replay: bool = False, scenario_name: str = "TridentIsland",
                 force_config_json: Optional[str] = None, spawn_config_json: Optional[str] = None):
        """Initialize environment"""
        
        super().__init__()

        self.config = config or Config()
        self.render_mode = render_mode
        self.enable_replay = enable_replay
        self.scenario_name = scenario_name
        
        # Set up paths
        self.scenario_path = Path(__file__).parent.parent / "scenarios"  
        # This is fixed for a single scenario, can be configured for running multiple mission types
        self.mission_events_path = Path(__file__).parent.parent.parent.parent / "FFSimulation" / "python" / "Bane" # TODO: Path to fixed MissionEvents.json?

        # Initialize simulation interface once
        SimulationInterface.initialize()
        
        # Simulation will be created in reset() method for proper Gymnasium compliance
        self.simulation = None
        
        # Load scenario data 
        scenario_path = Path(__file__).parent.parent / "scenarios" / "trident_island"

        # Initialize state tracking variables
        self.current_step = 0
        self.frame_rate = 600
        self.FrameIndex = 0
        self.time_elapsed = 0.0  # Mission time in seconds

        self.simulation_events = []
        
        # Parse data from the config and constants file as needed
        # Load CONSTANT mission rules (objectives, victory conditions, map)
        with open(scenario_path / "MissionEvents.json") as f:
            self.mission_events = Simulation.create_mission_events(f.read())

        # Set up simulation event handlers
        self.simulation_event_handlers = {
            EntitySpawned: self._entity_spawned,
            Victory: self._victory,
            AdversaryContact: self._adversary_contact
        }

        # Entity tracking - stores ALL entities from ALL factions
        self.entities = {}  # Dict[entity_id -> entity] for all entities in game
        self.target_groups = {}  # Dict[target_group_id -> target_group] for all target groups 
        
        # Calculate grid dimensions for discretized positioning
        map_size_km = self.config.map_size_km[0]
        self.grid_size = map_size_km // self.config.grid_resolution_km
        self.max_grid_positions = self.grid_size * self.grid_size
        
        # Calculate action space parameters from config
        angle_steps = 360 // self.config.angle_resolution_degrees
        patrol_steps = (self.config.max_patrol_axis_km - self.config.min_patrol_axis_km) // self.config.patrol_axis_increment_km + 1
        
        # Hierarchical action space
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(8),  # 0=noop, 1=move, 2=engage, 3=stealth, 4=sense_direction, 5=capture, 6=rtb, 7=refuel
            "entity_id": spaces.Discrete(self.config.max_entities),  # Which controllable entity to command
            
            # Move action parameters
            "move_center_grid": spaces.Discrete(self.max_grid_positions), # CAP route center position
            "move_short_angle": spaces.Discrete(angle_steps),  # Based on angle resolution
            "move_long_axis_km": spaces.Discrete(patrol_steps),  # 100-1000km in 25km increments
            "move_axis_angle": spaces.Discrete(angle_steps),   # Based on angle resolution
            
            # Engage action parameters
            "target_group_id": spaces.Discrete(self.config.max_target_groups),
            "weapon_selection": spaces.Discrete(self.config.max_weapon_combinations), # Combinatorial weapon selection
            "weapon_usage": spaces.Discrete(3),       # 0=1 shot/unit, 1=1 shot/adversary, 2=2 shots/adversary
            "weapon_engagement": spaces.Discrete(4),  # 0=defensive, 1=cautious, 2=assertive, 3=offensive
            
            # Stealth action parameters
            "stealth_enabled": spaces.Discrete(2),    # 0=off, 1=on
            
            # Sensing direction action parameters
            "sensing_position_grid": spaces.Discrete(self.max_grid_positions + 1), # +1 for default sensing (forward)
            
            # Refuel action parameters
            "refuel_target_id": spaces.Discrete(self.config.max_entities),
        })
        
        # Observation space (globals-only for now)
        self.observation_space = observations.build_observation_space(self.config)
        
        # Initialize force configuration paths (will be set by wrapper)
        self.force_config_paths = None
        
    def set_force_config_paths(self, legacy_entity_list, dynasty_entity_list, 
                              legacy_spawn_data, dynasty_spawn_data):
        """Set the JSON file paths for force configuration"""
        self.force_config_paths = {
            'legacy_entity_list': legacy_entity_list,
            'dynasty_entity_list': dynasty_entity_list,
            'legacy_spawn_data': legacy_spawn_data,
            'dynasty_spawn_data': dynasty_spawn_data
        }
        
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)

        # Destroy existing simulation if it exists
        # TODO: Do this at every reset?
        if self.simulation:
            SimulationInterface.destroy_simulation(self.simulation)
            self.simulation = None

        # Set up simulation using JSON files if available
        if self.force_config_paths:
            self.simulation = simulation_utils.setup_simulation_from_json(
                self, 
                self.force_config_paths['legacy_entity_list'],
                self.force_config_paths['dynasty_entity_list'],
                self.force_config_paths['legacy_spawn_data'],
                self.force_config_paths['dynasty_spawn_data'],
                seed=seed
            )
        else:
            # Use default paths from scenario directory
            legacy_entity_list = self.scenario_path / "force_composition" / "LegacyEntityList.json"
            dynasty_entity_list = self.scenario_path / "force_composition" / "DynastyEntityList.json"
            legacy_spawn_data = self.scenario_path / "laydown" / "LegacyEntitySpawnData.json"
            dynasty_spawn_data = self.scenario_path / "laydown" / "DynastyEntitySpawnData.json"
            
            self.simulation = simulation_utils.setup_simulation_from_json(
                self, 
                str(legacy_entity_list),
                str(dynasty_entity_list),
                str(legacy_spawn_data),
                str(dynasty_spawn_data),
                seed=seed
            )
        
        # Reset state tracking
        self.current_step = 0
        self.FrameIndex = 0
        self.time_elapsed = 0.0

        # For now just clear events
        self.simulation_events = []
        self.entities.clear()
        self.target_groups.clear()
        
        # Initialize enemy sensing data tracking
        self.enemy_sensing_data = {}  # Dict[int, EnemySensingData]
        
        # Reset mission metrics to track fresh mission progress
        mission_metrics.reset_mission_metrics(self)

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "valid_masks": {
                "action_types": self._get_valid_action_types(),           
                "controllable_entities": self._get_controllable_entities_set(),  
                "detected_targets": self._get_detected_targets_set(),
                "entity_target_matrix": self._get_entity_target_engagement_matrix()
            }
        }
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute hierarchical action"""
        self.current_step += 1
        self.FrameIndex += self.frame_rate  # Advance simulation time
        self.time_elapsed = get_time_elapsed(self.FrameIndex)  # Update mission time
        
        self.simulation_events = []  # Clear previous step's events
        
        # TODO: Reset frame-specific state if needed
        # self.frame_index += self.frame_rate  # Advance simulation time
        
        # Execute action
        player_events = actions.execute_action(action, self.entities, self.target_groups, self.config)
        
        # Prepare simulation step data
        sim_data = SimulationData()
        sim_data.player_events = player_events
        
        # Execute simulation step
        # TODO: Implement event processing
        simulation_utils.tick_simulation(self)
        
        # Update sensing information for enemy target groups
        self._update_enemy_sensing_data()

        # Get observation based on current sensing capabilities
        observation = self._get_observation()

        # Compute step reward 
        reward = self._calculate_reward()
        

        # Update all mission progress metrics
        mission_metrics.update_all_mission_metrics(self)

        terminated = self._check_termination()
        
        # Truncate on time limit 
        truncated = self.time_elapsed >= self.config.max_game_time

        # Update terminal reward
        
        info = {
            "step": self.current_step,
            "time_elapsed": self.time_elapsed,
            "total_entities": len(self.entities),
            "controllable_entities_count": len(self._get_controllable_entities_set()),
            "detected_targets_count": len(self._get_detected_targets_set()),
            "last_events_count": len(self.simulation_events),
            "valid_masks": {
                "action_types": self._get_valid_action_types(),           # What action types are possible 
                "controllable_entities": self._get_controllable_entities_set(),  # Which entity IDs can be commanded
                "detected_targets": self._get_detected_targets_set(),     # Which target group IDs are available
                "entity_target_matrix": self._get_entity_target_engagement_matrix()  # Which entities can engage which targets
        }
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Extract observation from simulation state (globals-only for now)."""
        # Update globals that depend on per-step conditions
        #self._update_global_flags()
        return observations.compute_observation(self)
    
    def _calculate_reward(self) -> float:
        """Calculate reward from simulation events"""
        # TODO: Implement reward calculation based on:
        # - Mission objectives (flag capture, target destruction)
        # - Unit preservation (casualties vs enemy losses) 
        # - Tactical performance (effective engagement ranges, formations)
        # - Step penalties (encourage efficient missions)
        return 0.0
    

    def _check_termination(self) -> bool:
        """Check if mission/episode should end"""
        # TODO: Check victory conditions from simulation
        # - Mission objectives achieved (victory)
        # - Critical units destroyed (defeat)
        # - Time limits exceeded
        # - Use constants.SIMULATION_VICTORY_THRESHOLD
        return False
    
    
    
    def render(self) -> Optional[np.ndarray]:
        # TODO: Should we implement rendering?
        """Render environment"""
        if self.render_mode == "rgb_array":
            # TODO: Generate visualization from simulation
            return np.zeros((400, 600, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            # TODO: Display visualization
            print(f"Step {self.current_step}")
    
    def get_simulation_handle(self):
        """Provide access to simulation for replay recording"""
        return self.simulation if self.enable_replay else None
    
    def load_force_config(self, force_config_path: str, spawn_areas_path: str = None):
        """Load force configuration and spawn areas from JSON files"""
        # TODO: Implement force configuration loading
        # This should load JSON and set up force laydown similar to bane_environment.py
        # - Load MissionEvents.json from FFSim scenarios
        # - Load EntitySpawnData.json for spawn areas from spawn_areas_path
        # - Load user-specified entity composition from force_config_path
        # - Set up simulation with agents and force laydown
        pass
    
    def load_spawn_areas(self, spawn_areas_path: str):
        """Load spawn areas configuration from JSON file"""
        # TODO: Implement spawn areas loading
        # This should load the EntitySpawnData.json format files
        # - Parse spawn area polygons for ground/sea/air forces
        # - Set up spawn area constraints for unit placement
        pass
    
    def _is_controllable_entity(self, entity) -> bool:
        """Check if entity is controllable by our faction"""
        return (entity.is_controllable and entity.is_alive and 
                entity.faction.value == self.config.our_faction)
    
    def _entity_can_engage(self, entity) -> bool:
        """Check if entity can engage targets (has weapons and valid targets exist)"""
        # Check if entity has weapons # TODO: Is this correct check?
        has_weapons = entity.has_weapons and len(entity.weapons) > 0
        if not has_weapons:
            return False
    
        # Check if any detected targets are valid for this entity
        # TODO: Is this correct check?
        for target_group in self.target_groups.values():
            is_enemy = target_group.faction.value != self.config.our_faction
            if is_enemy:
                available_weapons = entity.select_weapons(target_group, False)
                if len(available_weapons) > 0:
                    return True
        return False
    
    def _entity_can_capture(self, entity) -> bool:
        """Check if entity can capture objectives (ground units near objectives)"""
        # TODO: Check if its settler 
        return entity.is_settler
    
    def _get_valid_action_types(self) -> set:
        """Get set of valid action types based on current entities"""
        valid_actions = {0}  # noop always valid
    
        for entity in self.entities.values():
            if not self._is_controllable_entity(entity):
                continue
    
            if entity.domain == EntityDomain.AIR:
                valid_actions.update({1, 6})  # move, rtb
            if self._entity_can_engage(entity):
                valid_actions.add(2)  # engage
            if self._entity_can_capture(entity):
                valid_actions.add(5)  # capture
            if entity.has_radar:
                valid_actions.update({3, 4})  # stealth, sense
            if entity.has_fuel: # TODO: Can all air units refuel?
                valid_actions.add(7)  # refuel
    
        return valid_actions
    
    def _get_controllable_entities_set(self) -> set:
        """Get set of controllable entity IDs"""
        return {
            entity_id for entity_id, entity in self.entities.items()
            if self._is_controllable_entity(entity)
        }
    
    def _get_detected_targets_set(self) -> set:
        """Get set of detected target group IDs"""
        return {
            tg_id for tg_id, target_group in self.target_groups.items()
            if target_group.faction.value != self.config.our_faction
        }
        
    def _get_entity_target_engagement_matrix(self) -> dict:
        """Get matrix of which entities can engage which target groups"""
        engagement_matrix = {}
        
        for entity_id, entity in self.entities.items():
            if not self._is_controllable_entity(entity):
                continue
        
            # Get valid targets this entity can engage
            valid_targets = set()
            for tg_id, target_group in self.target_groups.items():
                # Check if enemy faction
                is_enemy = target_group.faction.value != self.config.our_faction
                if not is_enemy:
                    continue
    
                # Check weapon compatibility
                available_weapons = entity.select_weapons(target_group, False)
                if len(available_weapons) > 0:
                    valid_targets.add(tg_id)
        
            # Returns set of target group IDs that this entity can engage, empty set if none
            engagement_matrix[entity_id] = valid_targets
        
        return engagement_matrix
    
        
    def _entity_spawned(self, event):
        """Handle entity spawned events"""
        # TODO: Track spawned entities
        pass
        
    def _victory(self, event):
        """Handle victory events"""
        # TODO: Handle victory conditions
        pass
    
    def _adversary_contact(self, event):
        """Handle adversary contact events"""
        # TODO: Handle adversary detection
        pass
        
    def _update_enemy_sensing_data(self):
        """Update sensing information for all enemy target groups.
        
        This method should be called every step to update what we know about enemy forces
        based on our current sensor coverage and capabilities.
        """
        # TODO: Implement sensing update logic
        # 
        # Do something like: 
        # 1. Get all friendly sensor platforms (radars, AWACS, etc.)
        # friendly_sensors = [entity for entity in self.entities.values() 
        #                    if entity.faction.value == self.config.our_faction 
        #                    and entity.is_alive and entity.has_sensor_capability]
        # 
        # 2. For each enemy target group:
        # for group_id, target_group in self.target_groups.items():
        #     if target_group.faction.value == self.config.our_faction:
        #         continue  # Skip friendly groups
        #     
        #     # Calculate sensing capability for this group
        #     sensing_tier, confidence = self._calculate_sensing_tier(target_group, friendly_sensors)
        #     
        #     # Update or create sensing data entry
        #     if group_id not in self.enemy_sensing_data:
        #         self.enemy_sensing_data[group_id] = EnemySensingData()
        #     
        #     sensing_data = self.enemy_sensing_data[group_id]
        #     
        #     if sensing_tier > 0:
        #         # We can detect this group
        #         sensing_data.is_detected = True
        #         sensing_data.tier = sensing_tier
        #         sensing_data.confidence = confidence
        #         sensing_data.last_contact_time = self.time_elapsed
        #         
        #         # Update information based on sensing tier
        #         if sensing_tier >= 1:
        #             # Tier 1: Domain detection
        #             sensing_data.domain = target_group.get_primary_domain()
        #             sensing_data.estimated_unit_count = target_group.get_estimated_count()
        #             sensing_data.approximate_position = target_group.get_center_position()
        #         
        #         if sensing_tier >= 2:
        #             # Tier 2: Individual unit identification
        #             sensing_data.individual_positions = target_group.get_unit_positions()
        #             sensing_data.unit_types = target_group.get_unit_types()
        #             sensing_data.average_heading = target_group.get_average_heading()
        #             sensing_data.average_speed = target_group.get_average_speed()
        #         
        #         if sensing_tier >= 3:
        #             # Tier 3: Detailed weapon intelligence
        #             sensing_data.weapon_capabilities = target_group.get_weapon_capabilities()
        #             sensing_data.estimated_weapon_count = target_group.get_weapon_count()
        #             sensing_data.ammunition_status = target_group.get_ammo_status()
        #     
        #     else:
        #         # Lost contact or never detected
        #         sensing_data.is_detected = False
        #         # Keep historical data but mark as stale
        pass
    
    def _calculate_sensing_tier(self, target_group, friendly_sensors):
        """Calculate sensing tier and confidence for a target group.
    
        Args:
            target_group: Enemy target group to assess
            friendly_sensors: List of friendly sensor platforms
            
        Returns:
            tuple: (sensing_tier, confidence) where tier is 0-3 and confidence is 0.0-1.0
        """
        # TODO: Implement sensing tier calculation
        # 
        # Factors to consider:
        # - Range to target group from nearest sensor
        # - Sensor type and capability (radar, visual, ESM)
        # - Environmental conditions (weather, terrain masking)
        # - Target group stealth/ECM capabilities
        # - Sensor platform status (damaged, jammed, etc.)
        # 
        # Example logic:
        # max_tier = 0
        # best_confidence = 0.0
        # 
        # for sensor in friendly_sensors:
        #     range_to_target = calculate_distance(sensor.position, target_group.center)
        #     
        #     # Determine sensing capability based on range and sensor type
        #     if range_to_target <= sensor.tier3_range and sensor.has_detailed_capability:
        #         tier = 3
        #         confidence = 0.9 * sensor.reliability
        #     elif range_to_target <= sensor.tier2_range and sensor.has_identification_capability:
        #         tier = 2  
        #         confidence = 0.7 * sensor.reliability
        #     elif range_to_target <= sensor.tier1_range:
        #         tier = 1
        #         confidence = 0.5 * sensor.reliability
        #     else:
        #         tier = 0
        #         confidence = 0.0
        #     
        #     # Apply environmental and stealth modifiers
        #     confidence *= self._get_environmental_modifier(sensor, target_group)
        #     confidence *= (1.0 - target_group.stealth_factor)
        #     
        #     if tier > max_tier or (tier == max_tier and confidence > best_confidence):
        #         max_tier = tier
        #         best_confidence = confidence
        # 
        # return max_tier, best_confidence
        return 0, 0.0  # Placeholder

    def close(self):
        # TODO: To implement
        """Clean up"""
        if self.simulation:
            SimulationInterface.destroy_simulation(self.simulation)
            self.simulation = None