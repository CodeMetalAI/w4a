"""
W4A Constants

Game constants for the simulation engine.
"""

# Map definition
TRIDENT_ISLAND_MAP_SIZE = (2_500, 2_500)

# Game rules
SIMULATION_VICTORY_THRESHOLD = 0.9  # 90% enemy forces destroyed = victory
SIMULATION_DEFEAT_THRESHOLD = 0.1   # 90% own forces destroyed = defeat
CAPTURE_REQUIRED_SECONDS = 3 * 600.0  # Capture time in seconds
