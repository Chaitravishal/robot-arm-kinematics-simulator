"""Inverse kinematics for a planar two-link revolute (2R) robot."""

from math import atan2, acos, cos, sin, sqrt
from typing import Dict, List, Tuple


def inverse_kinematics(
    x: float, y: float, link1: float, link2: float
) -> List[Dict[str, Tuple[float, float]]]:
    """Return valid elbow-up/down solutions in radians.

    Returns an empty list when the target is outside the theoretical workspace.
    Joint limits are checked separately by the reachability module.
    """
    if link1 <= 0 or link2 <= 0:
        raise ValueError("Link lengths must be positive.")

    cos_theta2 = (x * x + y * y - link1 * link1 - link2 * link2) / (
        2.0 * link1 * link2
    )

    tolerance = 1e-10
    if cos_theta2 < -1.0 - tolerance or cos_theta2 > 1.0 + tolerance:
        return []

    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    solutions = []

    for sin_theta2 in (sqrt(max(0.0, 1.0 - cos_theta2**2)),
                       -sqrt(max(0.0, 1.0 - cos_theta2**2))):
        theta2 = atan2(sin_theta2, cos_theta2)
        theta1 = atan2(y, x) - atan2(
            link2 * sin(theta2), link1 + link2 * cos(theta2)
        )
        candidate = (theta1, theta2)
        if not any(
            abs(candidate[0] - old["angles"][0]) < 1e-9
            and abs(candidate[1] - old["angles"][1]) < 1e-9
            for old in solutions
        ):
            solutions.append({"angles": candidate})
    return solutions
