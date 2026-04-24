"""Time representation for CosmicEngine.

The engine uses Julian Date (JD, UTC) as its single scalar time coordinate.
JD is a continuous day count with fractional sub-day precision; it has no
calendar, leap-year, or timezone ambiguity, which makes it the natural
clock for astronomical data.

Reference: JD 2440587.5 == 1970-01-01T00:00:00 UTC (the POSIX epoch).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

# JD at the POSIX epoch, 1970-01-01T00:00:00 UTC.
_JD_UNIX_EPOCH = 2440587.5
_SECONDS_PER_DAY = 86400.0


def datetime_to_julian(dt: datetime) -> float:
    """Convert a :class:`datetime` to Julian Date (UTC).

    Naive datetimes are interpreted as UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / _SECONDS_PER_DAY + _JD_UNIX_EPOCH


def julian_to_datetime(jd: float) -> datetime:
    """Convert a Julian Date to a timezone-aware UTC :class:`datetime`."""
    return datetime.fromtimestamp(
        (jd - _JD_UNIX_EPOCH) * _SECONDS_PER_DAY, tz=timezone.utc
    )


@dataclass
class SimulationClock:
    """Monotonically advancing Julian-Date clock.

    Each :meth:`tick` adds ``delta_seconds * time_scale`` seconds of
    simulated time, unless the clock is paused. ``time_scale`` is
    dimensionless: 1.0 runs at real-time when ticked with real-seconds,
    negative values run the simulation backwards.
    """

    current_julian_date: float
    time_scale: float = 1.0
    paused: bool = False

    def tick(self, delta_seconds: float) -> None:
        """Advance simulated time by ``delta_seconds`` of wall clock."""
        if self.paused:
            return
        self.current_julian_date += (
            delta_seconds * self.time_scale / _SECONDS_PER_DAY
        )

    def set_time_scale(self, scale: float) -> None:
        """Set the wall-to-sim time multiplier."""
        self.time_scale = scale

    def pause(self) -> None:
        """Stop advancing on :meth:`tick`."""
        self.paused = True

    def resume(self) -> None:
        """Re-enable advancement on :meth:`tick`."""
        self.paused = False

    def set_julian_date(self, jd: float) -> None:
        """Jump the clock to an absolute Julian Date."""
        self.current_julian_date = jd

    def get_julian_date(self) -> float:
        """Return the current Julian Date."""
        return self.current_julian_date
