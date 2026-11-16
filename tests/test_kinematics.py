import math

from src.forward_kinematics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics


def test_forward_kinematics_zero_angles():
    x, y = forward_kinematics(0.0, 0.0, 2.0, 1.5)
    assert math.isclose(x, 3.5, abs_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-9)


def test_inverse_kinematics_reconstructs_target():
    target = (2.0, 1.0)
    solutions = inverse_kinematics(*target, 2.0, 1.5)
    assert solutions

    for solution in solutions:
        x, y = forward_kinematics(*solution["angles"], 2.0, 1.5)
        assert math.isclose(x, target[0], abs_tol=1e-8)
        assert math.isclose(y, target[1], abs_tol=1e-8)


def test_inverse_kinematics_rejects_unreachable_target():
    assert inverse_kinematics(10.0, 10.0, 2.0, 1.5) == []
