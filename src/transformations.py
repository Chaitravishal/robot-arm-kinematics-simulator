"""2D homogeneous transformation utilities for a planar robot."""

from math import cos, sin
import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    """Create a 3x3 homogeneous rotation matrix for angle theta (radians)."""
    c, s = cos(theta), sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def translation_matrix(x: float, y: float) -> np.ndarray:
    """Create a 3x3 homogeneous translation matrix."""
    return np.array(
        [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def link_transform(theta: float, length: float) -> np.ndarray:
    """Rotate by theta, then translate along the local x-axis by length."""
    if length < 0:
        raise ValueError("Link length cannot be negative.")
    return rotation_matrix(theta) @ translation_matrix(length, 0.0)
