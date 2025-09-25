import os

from pathlib import Path

from SimulationInterface import (Simulation, create_mock_entity)

# Todo: we could refactor this thing to an async io version
class W4AEntitiesRepository():
    def __init__(self, folder_path):
        self.entities = self.__read_entities(folder_path)

    def get_entity(self, name):
        assert name in self.entities

        return self.entities[name]

    def create_mock_entity(self, name):
        return create_mock_entity(Simulation.create_mission_event(self.get_entity(name)))

    def __read_entity(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def __read_entities(self, folder_path):
        print(f"Scanning {folder_path} for entities")
        entities = {}
        
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    extention = os.path.splitext(file)[1]

                    if extention != ".json":
                        continue

                    entity_path = os.path.join(root, file)
                    entity_name = os.path.splitext(file)[0]

                    

                    try:
                        entity = self.__read_entity(entity_path)
                        entities[entity_name] = entity
                    except (IOError, UnicodeDecodeError) as e:
                        print(f"Error reading entity {entity_path}: {e}")
                        continue
        except OSError as e:
            print(f"Error accessing folder {folder_path}: {e}")
        
        print(f"Read {len(entities)} entities")

        return entities

# Not super happy about this living on the global namespace
w4a_entities = W4AEntitiesRepository(Path(__file__).parent)