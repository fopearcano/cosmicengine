"""Barnes–Hut octree N-body approximation.

Replaces the O(N²) direct summation in
:mod:`cosmic_engine.physics.nbody` with an octree traversal that
treats distant clusters as a single point mass when the opening
criterion ``s/d < theta`` is satisfied, dropping the per-step cost
to roughly ``O(N log N)`` for well-distributed bodies.

Notes:
- Recursive Python tree walks; no GPU, no parallelism, no Cython.
- Works in tandem with the existing N-body integrator. The exact
  solver remains the reference; this module is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT


@dataclass
class OctreeNode:
    """One node in the Barnes–Hut octree.

    ``size`` is the cube **half-size** (so the node covers
    ``[center − size, center + size]`` on each axis). ``mass`` and
    ``center_of_mass`` are aggregates over all bodies under this node.

    A leaf has ``children is None``; if ``body_index is not None`` the
    leaf holds that single body, otherwise it is empty.
    """

    center: np.ndarray
    size: float
    mass: float
    center_of_mass: np.ndarray
    body_index: int | None
    children: list["OctreeNode"] | None

    def is_leaf(self) -> bool:
        """Return ``True`` for any node without subdivided children."""
        return self.children is None


def _empty_node(center: np.ndarray, size: float) -> OctreeNode:
    return OctreeNode(
        center=center,
        size=float(size),
        mass=0.0,
        center_of_mass=np.zeros(3, dtype=np.float64),
        body_index=None,
        children=None,
    )


def _octant_index(node: OctreeNode, pos: np.ndarray) -> int:
    idx = 0
    if pos[0] >= node.center[0]:
        idx |= 1
    if pos[1] >= node.center[1]:
        idx |= 2
    if pos[2] >= node.center[2]:
        idx |= 4
    return idx


def _create_children(node: OctreeNode) -> list[OctreeNode]:
    half = node.size * 0.5
    children: list[OctreeNode] = []
    for octant in range(8):
        offset = np.array(
            [
                half if octant & 1 else -half,
                half if octant & 2 else -half,
                half if octant & 4 else -half,
            ],
            dtype=np.float64,
        )
        children.append(_empty_node(node.center + offset, half))
    return children


def _insert(
    node: OctreeNode,
    body_index: int,
    positions: np.ndarray,
    masses: np.ndarray,
    max_depth: int,
    depth: int,
) -> None:
    pos = positions[body_index]
    m = float(masses[body_index])

    # Case 1: empty leaf — place the body.
    if node.children is None and node.body_index is None and node.mass == 0.0:
        node.body_index = body_index
        node.mass = m
        node.center_of_mass = pos.astype(np.float64).copy()
        return

    # Case 2: single-body leaf — subdivide (or merge if at max depth).
    if node.children is None and node.body_index is not None:
        existing_index = node.body_index

        if depth >= max_depth:
            total = node.mass + m
            node.center_of_mass = (
                node.center_of_mass * node.mass + pos * m
            ) / total
            node.mass = total
            return

        node.children = _create_children(node)
        # the existing leaf's body_index moves into its octant child
        existing_octant = _octant_index(node, positions[existing_index])
        _insert(
            node.children[existing_octant],
            existing_index,
            positions,
            masses,
            max_depth,
            depth + 1,
        )
        node.body_index = None  # node is now internal
        # node.mass / center_of_mass already reflect the existing body;
        # we'll add the new body below as in the internal-node case.

    # Case 3: internal node — accumulate and recurse.
    total = node.mass + m
    if total > 0.0:
        node.center_of_mass = (
            node.center_of_mass * node.mass + pos * m
        ) / total
    node.mass = total

    octant = _octant_index(node, pos)
    _insert(
        node.children[octant],
        body_index,
        positions,
        masses,
        max_depth,
        depth + 1,
    )


def build_octree(
    positions: np.ndarray,
    masses: np.ndarray,
    max_depth: int = 20,
) -> OctreeNode:
    """Build the Barnes–Hut octree over ``positions`` (shape ``(N, 3)``).

    The bounding cube is sized to comfortably contain every body.
    Returns the root node; an empty input yields an empty leaf at the
    origin.
    """
    n = len(masses)
    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must have shape ({n}, 3); got {positions.shape}"
        )
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative; got {max_depth}")

    if n == 0:
        return _empty_node(np.zeros(3, dtype=np.float64), 1.0)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) * 0.5
    extent = float((maxs - mins).max())
    half = max(extent * 0.5, 1.0e-9) * (1.0 + 1.0e-9)

    root = _empty_node(center.astype(np.float64), half)
    for i in range(n):
        _insert(root, i, positions, masses, max_depth, 0)
    return root


def compute_acceleration_bh(
    index: int,
    node: OctreeNode,
    positions: np.ndarray,
    masses: np.ndarray,
    theta: float,
    softening_m: float,
) -> np.ndarray:
    """Acceleration on body ``index`` from everything under ``node``."""
    if node.mass == 0.0:
        return np.zeros(3, dtype=np.float64)

    body_pos = positions[index]
    dx = node.center_of_mass - body_pos
    dist_sq = float(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])
    soft_sq = softening_m * softening_m
    eff_sq = dist_sq + soft_sq

    if node.is_leaf():
        # Self-leaf for this exact body → no contribution.
        if node.body_index == index:
            return np.zeros(3, dtype=np.float64)
        if eff_sq == 0.0:
            return np.zeros(3, dtype=np.float64)
        return (GRAVITATIONAL_CONSTANT * node.mass) * dx / (eff_sq ** 1.5)

    # Opening criterion: full cube width / distance.
    full_size = node.size * 2.0
    if dist_sq > 0.0 and full_size * full_size < theta * theta * dist_sq:
        return (GRAVITATIONAL_CONSTANT * node.mass) * dx / (eff_sq ** 1.5)

    accel = np.zeros(3, dtype=np.float64)
    for child in node.children:
        accel += compute_acceleration_bh(
            index, child, positions, masses, theta, softening_m
        )
    return accel


def compute_accelerations_bh(
    positions: np.ndarray,
    masses: np.ndarray,
    theta: float = 0.5,
    softening_m: float = 0.0,
) -> np.ndarray:
    """Return ``(N, 3)`` accelerations using one freshly-built octree."""
    n = len(masses)
    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must have shape ({n}, 3); got {positions.shape}"
        )
    if theta <= 0.0:
        raise ValueError(f"theta must be positive; got {theta}")
    if softening_m < 0.0:
        raise ValueError(f"softening_m must be non-negative; got {softening_m}")
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    root = build_octree(positions, masses)
    accel = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        accel[i] = compute_acceleration_bh(
            i, root, positions, masses, theta, softening_m
        )
    return accel
