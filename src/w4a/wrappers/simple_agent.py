from SimulationInterface import Agent
from SimulationInterface import *

class LoggingMixin():
	def log(self, message):
		print(f"{self.log_prefix}{message}")

	@property
	def log_prefix(self):
		return "Log: "

class SimpleAgent(Agent, LoggingMixin):
	def __init__(self, faction):
		super().__init__()
		self.faction = faction

		self.log(f"Constructed")

		self.entities = set()

		self.simulation_event_handlers = {
			EntitySpawned: self.__entity_spawned,
			AdversaryContact: self.__adversary_contact,
			ComponentSpawned: self.__component_spawned
		}

	@property
	def log_prefix(self):
		return f"{self.faction.name} {self.__class__.__name__} (frame {self.frame_index}): "

	def start_force_laydown(self, force_laydown):
		self.force_laydown = force_laydown

	def finalize_force_laydown(self):
		force_laydown = self.force_laydown

		entities = []

		for entity in force_laydown.ground_forces_entities:
			spawn_location = force_laydown.get_random_ground_force_spawn_location()

			entity.pos = spawn_location.pos
			entity.rot = spawn_location.rot

			entities.append(entity)

		for entity in force_laydown.sea_forces_entities:
			spawn_location = force_laydown.get_random_sea_force_spawn_location()

			entity.pos = spawn_location.pos
			entity.rot = spawn_location.rot

			entities.append(entity)

		for entity in force_laydown.air_forces_entities:
			spawn_location = force_laydown.get_random_air_force_spawn_location()

			entity.pos = spawn_location.pos
			entity.rot = spawn_location.rot

			NonCombatManouverQueue.create(entity.pos, lambda: 
				CAPManouver.create_from_spline_points(force_laydown.get_random_cap().spline_points))

			entities.append(entity)

		return entities	# We create no packages at the moment, but just place stuff on the map as is.

	def pre_simulation_tick(self, simulation_data):
		self.log("Received pre_simulation_tick")
		self.player_events = []

		self.__process_simulation_events(simulation_data.simulation_events)

		simulation_data.player_events = self.player_events

	def tick(self, simulation_data):
		self.log("Received tick")
		
		self.player_events = []

		self.__process_simulation_events(simulation_data.simulation_events)

		simulation_data.player_events = self.player_events

	def __process_simulation_events(self, events):
		for event in events:
			handler = self.simulation_event_handlers.get(type(event))

			if handler:
				handler(event)
			else:
				self.log(f"Unhandled {event.__class__.__name__}")

				if event.__class__.__name__ == "SimulationEvent":
					input("Debug!")
				
				pass

	def __entity_spawned(self, event):
		if issubclass(event.entity.__class__, ControllableEntity):
			self.__controllable_entity_spawned(event)

	def __component_spawned(self, event):
		self.log(f"{event.component.__class__.__name__} {event.component.entity.identifier} spawned")

	def __controllable_entity_spawned(self, event):
		if event.entity.has_parent:
			return

		identifier = event.entity.identifier

		self.log(f"{event.entity.__class__.__name__} {event.entity.identifier} spawned")

		self.entities.add(event.entity)

	def __adversary_contact(self, event):
		selected_weapons = event.entity.select_weapons(event.target_group, False)

		if len(selected_weapons) == 0:
			return

		#jamming = PlayerEventJam()
		#jamming.jamming_points = [Vector3(19, 0, 0), Vector3(19, 0, 0), Vector3(19, 0, 0), Vector3(19, 0, 0)]

		commit = PlayerEventCommit()
		commit.entity = event.entity
		commit.target_group = event.target_group
		commit.manouver_data.throttle = 1.0
		commit.manouver_data.engagement = 2
		commit.manouver_data.weapon_usage = 2
		commit.manouver_data.weapons = selected_weapons.keys()   # We just select all available weapons for now.
		commit.manouver_data.wez_scale = 1

		self.player_events.append(commit)
