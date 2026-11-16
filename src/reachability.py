"""Reachability checks for a planar 2R manipulator."""

from math import hypot, radians
from typing import Dict, Tuple


def check_reachability(
    x: float,
    y: float,
    link1: float,
    link2: float,
    joint1_limits_deg: Tuple[float, float] = (-180.0, 180.0),
    joint2_limits_deg: Tuple[float, float] = (-180.0, 180.0),
) -> Dict[str, object]:
    """Check radial workspace and whether any IK solution meets joint limits."""
    from src.inverse_kinematics import inverse_kinematics

    distance = hypot(x, y)
    min_reach = abs(link1 - link2)
    max_reach = link1 + link2
    in_radial_workspace = min_reach - 1e-9 <= distance <= max_reach + 1e-9

    solutions = inverse_kinematics(x, y, link1, link2) if in_radial_workspace else []
    valid_solutions = []
    for solution in solutions:
        theta1, theta2 = solution["angles"]
        theta1_deg, theta2_deg = __import__("math").degrees(theta1), __import__("math").degrees(theta2)
        if (
            joint1_limits_deg[0] <= theta1_deg <= joint1_limits_deg[1]
            and joint2_limits_deg[0] <= theta2_deg <= joint2_limits_deg[1]
        ):
            valid_solutions.append((theta1, theta2))

    return {
        "reachable": bool(valid_solutions),
        "in_radial_workspace": in_radial_workspace,
        "target_distance": distance,
        "minimum_reach": min_reach,
        "maximum_reach": max_reach,
        "valid_solutions": valid_solutions,
    }
