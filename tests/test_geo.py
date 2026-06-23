"""Tests for geographic utility functions."""

from src.utils.geo import bearing, great_circle_path, haversine


def test_haversine_jfk_lhr():
    """JFK to LHR is approximately 5,570 km."""
    d = haversine(40.6413, -73.7781, 51.4700, -0.4543)
    assert 5520 < d < 5620


def test_haversine_zero():
    """Same point gives zero distance."""
    assert haversine(51.5, -0.1, 51.5, -0.1) == 0.0


def test_bearing_east():
    """From equator/prime-meridian to equator/1°E should be ~90°."""
    b = bearing(0, 0, 0, 1)
    assert 89 < b < 91


def test_great_circle_endpoints():
    """First and last points of the great circle match origin/destination."""
    path = great_circle_path(40.6413, -73.7781, 51.4700, -0.4543, n_points=20)
    assert abs(path[0][0] - 40.6413) < 0.01
    assert abs(path[0][1] - (-73.7781)) < 0.01
    assert abs(path[-1][0] - 51.4700) < 0.01
    assert abs(path[-1][1] - (-0.4543)) < 0.01


def test_path_length():
    """Sum of segments should approximate total haversine distance."""
    path = great_circle_path(40.6413, -73.7781, 51.4700, -0.4543, n_points=50)
    total = sum(
        haversine(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        for i in range(len(path) - 1)
    )
    direct = haversine(40.6413, -73.7781, 51.4700, -0.4543)
    assert abs(total - direct) / direct < 0.01
