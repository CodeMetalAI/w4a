"""
Enemy sensing data structures for tiered intelligence system.

This module defines the data structures used to track what we know about enemy forces
based on our sensor capabilities and intelligence gathering.
"""

# TODO: This is all pseudocode!

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class SensingTier(Enum):
    """Sensing capability tiers."""
    NONE = 0        # No detection
    DOMAIN = 1      # Domain detection only (air/surface/land)
    INDIVIDUAL = 2  # Individual unit identification and positions
    DETAILED = 3    # Detailed weapon and capability intelligence


class Domain(Enum):
    """Unit domain types."""
    AIR = "AIR"
    SURFACE = "SURFACE" 
    LAND = "LAND"
    UNKNOWN = "UNKNOWN"


@dataclass
class WeaponCapabilities:
    """Weapon system capabilities (Tier 3 intelligence)."""
    can_engage_air: bool = False
    can_engage_surface: bool = False
    can_engage_land: bool = False
    max_range_km: float = 0.0
    weapon_types: List[str] = field(default_factory=list)
    estimated_ammunition: int = 0


@dataclass
class EnemySensingData:
    """Complete sensing data for an enemy target group.
    
    This structure holds all intelligence we have gathered about an enemy target group,
    organized by sensing tier. Higher tiers include more detailed information.
    """
    
    # Detection status
    is_detected: bool = False
    tier: int = 0  # SensingTier value (0-3)
    confidence: float = 0.0  # Detection confidence [0.0, 1.0]
    last_contact_time: float = 0.0  # Time of last sensor contact
    
    # Tier 1: Domain detection (enables weapon selection)
    domain: Domain = Domain.UNKNOWN
    estimated_unit_count: int = 0
    approximate_position: Tuple[float, float] = (0.0, 0.0)  # (x, y) rough location
    estimated_threat_level: float = 0.0  # Threat assessment 0-10
    
    # Tier 2: Individual unit identification and tracking
    individual_positions: List[Tuple[float, float]] = field(default_factory=list)
    unit_types: List[str] = field(default_factory=list)  # Specific unit identifications
    average_heading: float = 0.0  # Average group heading in degrees
    average_speed: float = 0.0   # Average group speed
    formation_type: str = "unknown"  # Formation pattern if identifiable
    
    # Tier 3: Detailed weapon and capability intelligence
    weapon_capabilities: Optional[WeaponCapabilities] = None
    estimated_weapon_count: int = 0
    ammunition_status: str = "unknown"  # "full", "partial", "low", "unknown"
    electronic_warfare_capable: bool = False
    stealth_factor: float = 0.0  # Stealth capability [0.0, 1.0]
    
    # Engagement tracking
    is_engaging_friendly_forces: bool = False
    last_engagement_time: float = 0.0
    
    def get_group_center_position(self) -> Tuple[float, float]:
        """Get center position of the target group.
        
        Returns:
            (x, y) coordinates of group center
        """
        if self.tier >= 2 and self.individual_positions:
            # Tier 2+: Calculate actual centroid from individual positions
            total_x = sum(pos[0] for pos in self.individual_positions)
            total_y = sum(pos[1] for pos in self.individual_positions)
            count = len(self.individual_positions)
            return (total_x / count, total_y / count)
        else:
            # Tier 1: Use approximate position
            return self.approximate_position
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty factor.
        
        Returns:
            Uncertainty factor [0.0, 1.0] where 0.0 = very certain, 1.0 = very uncertain
        """
        if self.tier >= 2:
            return 0.1  # High confidence with individual tracking
        elif self.tier >= 1:
            return 0.8  # Low confidence with domain detection only
        else:
            return 1.0  # No detection, maximum uncertainty
    
    def get_estimated_capabilities(self) -> WeaponCapabilities:
        """Get weapon capabilities (actual or estimated).
        
        Returns:
            WeaponCapabilities object with best available information
        """
        if self.tier >= 3 and self.weapon_capabilities:
            # Tier 3: Actual weapon intelligence
            return self.weapon_capabilities
        elif self.tier >= 1:
            # Tier 1-2: Estimate capabilities from domain knowledge
            capabilities = WeaponCapabilities()
            
            if self.domain == Domain.AIR:
                capabilities.can_engage_air = True
                capabilities.can_engage_surface = True
                capabilities.max_range_km = 100.0  # Typical air-to-air/surface range
            elif self.domain == Domain.SURFACE:
                capabilities.can_engage_air = True
                capabilities.can_engage_surface = True
                capabilities.max_range_km = 150.0  # Typical surface-to-air/surface range
            elif self.domain == Domain.LAND:
                capabilities.can_engage_land = True
                capabilities.max_range_km = 50.0   # Typical land-based range
            
            return capabilities
        else:
            # No detection: Unknown capabilities
            return WeaponCapabilities()
    
    def update_from_sensor_contact(self, tier: int, confidence: float, current_time: float):
        """Update sensing data from new sensor contact.
        
        Args:
            tier: New sensing tier achieved
            confidence: Detection confidence
            current_time: Current simulation time
        """
        self.is_detected = True
        self.tier = max(self.tier, tier)  # Never downgrade tier within same contact
        self.confidence = max(self.confidence, confidence)  # Keep best confidence
        self.last_contact_time = current_time
    
    def age_data(self, current_time: float, max_age: float = 300.0):
        """Age the sensing data and reduce confidence over time.
        
        Args:
            current_time: Current simulation time
            max_age: Maximum age before data becomes unreliable (seconds)
        """
        time_since_contact = current_time - self.last_contact_time
        
        if time_since_contact > max_age:
            # Data too old, mark as not detected
            self.is_detected = False
            self.confidence = 0.0
        else:
            # Reduce confidence based on age
            age_factor = 1.0 - (time_since_contact / max_age)
            self.confidence *= age_factor


# Example usage and data structure documentation:
"""
Example enemy_sensing_data structure in environment:

env.enemy_sensing_data = {
    0: EnemySensingData(
        is_detected=True,
        tier=2,  # Individual unit tracking
        confidence=0.85,
        last_contact_time=1234.5,
        domain=Domain.AIR,
        estimated_unit_count=4,
        approximate_position=(12000.0, 8000.0),
        individual_positions=[(12100, 8100), (12000, 8000), (11900, 7900), (12050, 8050)],
        unit_types=["F-16C", "F-16C", "F-16C", "F-16C"],
        average_heading=270.0,
        average_speed=450.0,
        estimated_threat_level=7.5
    ),
    1: EnemySensingData(
        is_detected=True,
        tier=1,  # Domain detection only
        confidence=0.6,
        last_contact_time=1230.0,
        domain=Domain.SURFACE,
        estimated_unit_count=2,
        approximate_position=(15000.0, 5000.0),
        estimated_threat_level=8.0
    ),
    2: EnemySensingData(
        is_detected=False,  # Lost contact
        tier=1,
        confidence=0.0,
        last_contact_time=1200.0,  # 34.5 seconds ago
        domain=Domain.LAND,
        # ... stale data retained for reference
    )
}
"""
