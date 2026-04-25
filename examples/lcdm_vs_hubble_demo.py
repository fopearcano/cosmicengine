"""Demo: compare flat-ΛCDM and Hubble-law distances across redshift."""

from __future__ import annotations

from pathlib import Path

from cosmic_engine.core.registry import UniverseRegistry
from cosmic_engine.data.desi_like_catalog import load_desi_into_registry
from cosmic_engine.physics.cosmology import (
    redshift_to_distance_m,
    set_cosmology_mode,
)
from cosmic_engine.physics.cosmology_lcdm import (
    MPC_IN_METERS,
    angular_diameter_distance_m,
    comoving_distance_m,
    luminosity_distance_m,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = _REPO_ROOT / "data" / "sample_desi_like.csv"

_TEST_Z = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def _hubble_distance_m(z: float) -> float:
    set_cosmology_mode("hubble")
    try:
        return redshift_to_distance_m(z)
    finally:
        set_cosmology_mode("lcdm")


def main() -> None:
    print(f"{'z':>6}  {'D_hubble (Mpc)':>16}  {'D_lcdm (Mpc)':>16}  "
          f"{'D_L (Mpc)':>14}  {'D_A (Mpc)':>14}  {'rel.err':>8}")
    print("-" * 90)

    max_rel_err = 0.0
    for z in _TEST_Z:
        d_hub = _hubble_distance_m(z) / MPC_IN_METERS
        d_c = comoving_distance_m(z) / MPC_IN_METERS
        d_l = luminosity_distance_m(z) / MPC_IN_METERS
        d_a = angular_diameter_distance_m(z) / MPC_IN_METERS
        rel = (d_hub - d_c) / d_c if d_c > 0 else 0.0
        max_rel_err = max(max_rel_err, abs(rel))
        print(
            f"{z:>6.2f}  {d_hub:>16.2f}  {d_c:>16.2f}  "
            f"{d_l:>14.2f}  {d_a:>14.2f}  {rel:>+8.2%}"
        )
    print()
    print(f"max |rel. err| over z range: {max_rel_err:.2%}")
    print()

    # Compare distance ranges of the bundled DESI sample under both modes.
    registry_lcdm = UniverseRegistry()
    set_cosmology_mode("lcdm")
    load_desi_into_registry(str(_CATALOG), registry_lcdm)

    registry_hubble = UniverseRegistry()
    set_cosmology_mode("hubble")
    load_desi_into_registry(str(_CATALOG), registry_hubble)
    set_cosmology_mode("lcdm")  # restore default

    def _range(reg: UniverseRegistry) -> tuple[float, float]:
        ds = []
        for o in reg.list_objects():
            ds.append(
                (o.position_m.x ** 2 + o.position_m.y ** 2 + o.position_m.z ** 2)
                ** 0.5
                / MPC_IN_METERS
            )
        return min(ds), max(ds)

    lcdm_min, lcdm_max = _range(registry_lcdm)
    hub_min, hub_max = _range(registry_hubble)
    print(f"sample_desi_like.csv distance ranges (Mpc):")
    print(f"  ΛCDM   : [{lcdm_min:.1f}, {lcdm_max:.1f}]")
    print(f"  Hubble : [{hub_min:.1f}, {hub_max:.1f}]")


if __name__ == "__main__":
    main()
