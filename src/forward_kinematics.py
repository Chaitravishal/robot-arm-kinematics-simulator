"""Forward kinematics for a planar two-link revolute (2R) robot."""

from math import cos, sin
from typing import Tuple


def forward_kinematics(
    theta1: float, theta2: float, link1: float, link2: float
) -> Tuple[float, float]:
    """Return the end-effector (x, y) position.

    Angles are supplied in radians. theta2 is relative to link 1.
    """
    if link1 <= 0 or link2 <= 0:
        raise ValueError("Link lengths must be positive.")

    x = link1 * cos(theta1) + link2 * cos(theta1 + theta2)
    y = link1 * sin(theta1) + link2 * sin(theta1 + theta2)
    return x, y
