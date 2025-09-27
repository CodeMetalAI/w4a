"""
TridentIslandEnv

Single-agent Gymnasium environment for tactical simulation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from .simple_agent import SimpleAgent

from ..config import Config
from . import actions
from ..entities import w4a_entities

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
                 enable_replay: bool = False):
        """Initialize environment"""
        self.config = config or Config()
        self.render_mode = render_mode
        self.enable_replay = enable_replay
        
        # Initialize Bane simulation engine
        SimulationInterface.initialize()
        
        # Load scenario data 
        scenario_path = Path(__file__).parent.parent / "scenarios" / "trident_island"

        # Parse data from the config and constants file as needed
        # Load CONSTANT mission rules (objectives, victory conditions, map)
        with open(scenario_path / "MissionEvents.json") as f:
            self.mission_events = Simulation.create_mission_events(f.read())

        # Load the spawn data
        self.faction_entity_spawn_data = {}

        with open(scenario_path / "LegacyEntitySpawnData.json") as f:
            self.faction_entity_spawn_data[Faction.LEGACY] = EntitySpawnData.import_json(f.read())

        with open(scenario_path / "DynastyEntitySpawnData.json") as f:
            self.faction_entity_spawn_data[Faction.DYNASTY] = EntitySpawnData.import_json(f.read())

        # Load the entity data. This is the outcome of the auction that takes place prior to the simulation
        entity_lists_path = Path(__file__).parent.parent / "entity_lists"

        self.faction_entity_data = {}

        with open(entity_lists_path / "LegacyEntityList.json") as f:
            self.faction_entity_data[Faction.LEGACY] = EntityList().load_json(f.read())

        with open(entity_lists_path / "DynastyEntityList.json") as f:
            self.faction_entity_data[Faction.DYNASTY] = EntityList().load_json(f.read())
        
        # Calculate grid dimensions for discretized positioning
        map_size_km = self.config.map_size_km[0]
        self.grid_size = map_size_km // self.config.grid_resolution_km
        self.max_grid_positions = self.grid_size * self.grid_size
        
        # Calculate action space parameters from config
        angle_steps = 360 // self.config.angle_resolution_degrees
        patrol_steps = (self.config.max_patrol_axis_km - self.config.min_patrol_axis_km) // self.config.patrol_axis_increment_km + 1
        
        # Hierarchical action space
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(8),  # 0=noop, 1=move, 2=engage, 3=stealth, 4=sense_direction, 5=land, 6=rtb, 7=refuel
            "entity_id": spaces.Discrete(self.config.max_entities),
            
            # Move action parameters
            "move_center_grid": spaces.Discrete(self.max_grid_positions),
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
        
        # Observation space: TODO: to implement, placeholder for entity positions, health, etc.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(100,), dtype=np.float32
        )
        # TODO: Weapons: Range
        
        self.current_step = 0
        self.simulation_events = []
        
        # Entity tracking - stores ALL entities from ALL factions
        self.entities = {}  # Dict[entity_id -> entity] for all entities in game
        self.target_groups = {}  # Dict[target_group_id -> target_group] for all target groups
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)

        # Create simulation with replay capability
        sim_config = SimulationConfig()
        sim_config.log_json = self.enable_replay  # Enable replay recording
        sim_config.random_seed = self.config.seed or 42
        sim_config.name = "TridentIsland"
        
        # Create simulation from scenario data
        # self.simulation = SimulationInterface.create_simulation_from_data(scenario_json, True)
        
        # For now: basic simulation creation without scenario
        self.simulation = SimulationInterface.create_simulation(sim_config)

        # TODO: @Sanjna initialize the actual RL agents here
        self.simulation.add_agent(SimpleAgent(Faction.LEGACY))
        self.simulation.add_agent(SimpleAgent(Faction.DYNASTY))

        sim_data = SimulationData()
        sim_data.add_mission_events(self.mission_events)    # Don't think this holds up the second time. We might need to recrete them from json every time

        force_laydowns = {}
        for faction in [Faction.LEGACY, Faction.DYNASTY]:
            force_laydown = ForceLaydown()
            force_laydown.entity_spawn_data = self.faction_entity_spawn_data[faction]
            force_laydown.entity_data = FactionConfiguration().create_entities(self.faction_entity_data[faction], lambda type: self.simulation.create_mission_event(w4a_entities.get_entity(type)))

            force_laydowns[faction] = force_laydown

        # Process the mission setup
        self.simulation.start_force_laydown(force_laydowns)
        self.simulation.finalize_force_laydown(sim_data)

        # Process all events coming out of this
        self._process_simulation_events(sim_data.simulation_events)

        # For now just clear events
        self.simulation_events = []
        
        self.current_step = 0
        observation = self._get_observation()
        info = {"step": self.current_step}
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute hierarchical action"""
        self.current_step += 1
        
        # Execute action based on type
        player_events = actions.execute_action(action, self.entities, self.target_groups, self.config)
        
        # Prepare simulation step data
        sim_data = SimulationData()
        sim_data.player_events = player_events
        
        # Execute simulation step
        # TODO: Parameters for simulation step
        # self.simulation_events = SimulationInterface.tick_simulation(
        #     self.simulation, sim_data, 1  # 1 simulation step
        # )
        
        # Process simulation events (adjudication results)
        # TODO: Implement event processing
        # self._process_simulation_events(self.simulation_events)
        
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_episode_steps
        
        info = {
            "step": self.current_step,
            "action_mask": self.get_action_mask()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Extract observation from simulation state"""
        # TODO: Implement observation extraction
        # - Get all entities from simulation
        # - Extract positions, health, weapon status, sensor data
        # - Apply fog of war / sensor limitations
        # - Convert to fixed-size observation vector
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    
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
    
    def _process_simulation_events(self, events):
        """Process events from simulation step"""
        # TODO: Handle simulation events for observations and rewards
        # - EntitySpawned, EntityKilled events
        # - AdversaryContact (radar detections)
        # - CombatShooting, ProjectileExploded events
        # - Victory/defeat conditions
        pass
    
    
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
    
    
    
    
    
    
    
    
    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """Get action validity masks for dynamic parameters"""
        return {
            "entity_id": self._get_entity_mask(),
            "refuel_target_id": self._get_refuel_target_mask(),
            # TODO: Add engagement masks when sensor model is clear
            # "target_group_id": self._get_target_group_mask(),
            # "weapon_selection": self._get_weapon_selection_mask(),
        }
    
    def _get_entity_mask(self) -> np.ndarray:
        """Mask for entities that can perform actions"""
        mask = np.zeros(self.config.max_entities, dtype=bool)
        
        for entity_id, entity in self.entities.items():
            if entity_id < self.config.max_entities:
                # Our faction + controllable + alive
                mask[entity_id] = (
                    isinstance(entity, ControllableEntity) and
                    entity.is_alive and
                    entity.faction.value == self.config.our_faction
                )
        
        return mask
    
    def _get_move_entity_mask(self) -> np.ndarray:
        """Mask for entities that can perform move actions (air units with fuel)"""
        mask = np.zeros(self.config.max_entities, dtype=bool)
        
        for entity_id, entity in self.entities.items():
            if entity_id < self.config.max_entities:
                # Our faction + controllable + alive + air unit + has fuel
                mask[entity_id] = (
                    isinstance(entity, ControllableEntity) and
                    entity.is_alive and
                    entity.faction.value == self.config.our_faction and
                    entity.domain == EntityDomain.AIR and
                    entity.has_fuel  # TODO: Check if this property exists in sim
                )
        
        return mask
    
    def _get_refuel_target_mask(self) -> np.ndarray:
        """Mask for entities that can provide refueling"""
        mask = np.zeros(self.config.max_entities, dtype=bool)
        
        for entity_id, entity in self.entities.items():
            if entity_id < self.config.max_entities:
                # Our faction + controllable + alive + can refuel others
                mask[entity_id] = (
                    isinstance(entity, ControllableEntity) and
                    entity.is_alive and
                    entity.faction.value == self.config.our_faction and
                    entity.can_refuel_others  # TODO: Check if this property exists in sim
                )
        
        return mask
    
    def _get_target_group_mask(self) -> np.ndarray:
        """Mask for valid engagement targets (placeholder)"""
        # TODO: Implement when sensor model is clear
        # Should mask for: enemy faction + detected + in range + valid weapons available
        mask = np.zeros(self.config.max_target_groups, dtype=bool)
        
        for tg_id, target_group in self.target_groups.items():
            if tg_id < self.config.max_target_groups:
                # Enemy faction (simple version for now)
                mask[tg_id] = (target_group.faction.value != self.config.our_faction)
        
        return mask
    
    def _get_weapon_selection_mask(self) -> np.ndarray:
        """Mask for valid weapon combinations (placeholder)"""
        # TODO: Implement entity+target dependent weapon masking
        # Should mask for: compatible weapons + has ammo + in firing range
        return np.ones(self.config.max_weapon_combinations, dtype=bool)
    
    def close(self):
        """Clean up"""
        if self.simulation:
            SimulationInterface.destroy_simulation(self.simulation)
            self.simulation = None