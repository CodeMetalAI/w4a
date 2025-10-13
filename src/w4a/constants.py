"""
W4A Constants

Game constants for the simulation engine.
"""

# Map definition
TRIDENT_ISLAND_MAP_SIZE = (25_000, 25_000)  # 25km x 25km

# Game rules
SIMULATION_VICTORY_THRESHOLD = 0.9  # 90% enemy forces destroyed = victory
SIMULATION_DEFEAT_THRESHOLD = 0.1   # 90% own forces destroyed = defeat
CAPTURE_REQUIRED_SECONDS = 600.0  # Capture time in seconds
