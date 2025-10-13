"""
Environment Constants

This module contains shared constants used across multiple environment modules.
Separating constants here prevents circular import issues.
"""

from SimulationInterface import Faction


# Faction flag IDs used in simulation
FACTION_FLAG_IDS = {
    Faction.LEGACY: 2, 
    Faction.DYNASTY: 1, 
    Faction.NEUTRAL: 3
}

# Center island flag ID (neutral objective)
CENTER_ISLAND_FLAG_ID = FACTION_FLAG_IDS[Faction.NEUTRAL]

