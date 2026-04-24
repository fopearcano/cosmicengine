"""Physical constants and unit conversions used across the engine.

All internal state is stored in SI units (meters, seconds, kilograms).
These helpers exist so ingestion code can convert from catalog-native
units at the boundary and downstream code never has to guess.
"""

from __future__ import annotations

# Speed of light in vacuum (exact SI definition), m/s.
SPEED_OF_LIGHT_M_S: float = 299_792_458.0

# IAU 2012 definition of the astronomical unit, in meters.
AU_IN_METERS: float = 1.495_978_707e11

# Julian light-year: c * 365.25 days, in meters.
LIGHTYEAR_IN_METERS: float = SPEED_OF_LIGHT_M_S * 365.25 * 86_400.0

# IAU 2015 parsec: 648000/pi AU, in meters.
PARSEC_IN_METERS: float = 3.085_677_581_491_367_3e16


def au_to_meters(value: float) -> float:
    """Convert astronomical units to meters."""
    return value * AU_IN_METERS


def lightyear_to_meters(value: float) -> float:
    """Convert light-years to meters."""
    return value * LIGHTYEAR_IN_METERS


def parsec_to_meters(value: float) -> float:
    """Convert parsecs to meters."""
    return value * PARSEC_IN_METERS
