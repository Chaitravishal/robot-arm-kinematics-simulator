from src.reachability import check_reachability


def test_reachable_target_inside_workspace():
    result = check_reachability(2.0, 1.0, 2.0, 1.5)
    assert result["in_radial_workspace"] is True
    assert result["reachable"] is True


def test_unreachable_target_outside_workspace():
    result = check_reachability(10.0, 10.0, 2.0, 1.5)
    assert result["in_radial_workspace"] is False
    assert result["reachable"] is False
