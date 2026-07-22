import sys
import unittest
from pathlib import Path

import numpy as np

COLLECT_DIR = Path(__file__).resolve().parents[1]
if str(COLLECT_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECT_DIR))

from trajectory import BlendedPath, duration_for_limits, smoothstep5  # noqa: E402


class SmoothStepTests(unittest.TestCase):
    def test_endpoint_velocity_is_zero(self):
        value, derivative = smoothstep5(np.array([0.0, 1.0]))
        np.testing.assert_allclose(value, [0.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(derivative, [0.0, 0.0], atol=1e-15)


class BlendedPathTests(unittest.TestCase):
    def test_corner_is_passed_with_continuous_tangent(self):
        path = BlendedPath([[0, 0, 0], [0.1, 0, 0], [0.1, 0, 0.1]], 0.012)
        corner_s = path.closest_fraction([0.1, 0, 0]) * path.length
        _, before = path.at(corner_s - 1e-5)
        _, after = path.at(corner_s + 1e-5)

        self.assertGreater(np.dot(before, after), 0.999)
        np.testing.assert_allclose(path.at(0.0)[0], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(path.at(path.length)[0], [0.1, 0, 0.1], atol=1e-12)

    def test_blend_stays_within_corner_radius(self):
        radius = 0.012
        path = BlendedPath([[0, 0, 0], [0.1, 0, 0], [0.1, 0, 0.1]], radius)
        corner_s = path.closest_fraction([0.1, 0, 0]) * path.length
        closest, _ = path.at(corner_s)
        self.assertLessEqual(np.linalg.norm(closest - [0.1, 0, 0]), radius)


class DurationTests(unittest.TestCase):
    def test_longer_paths_take_longer(self):
        limits = {
            "min_duration": 0.25,
            "linear_speed": 0.12,
            "linear_accel": 0.5,
            "linear_jerk": 4.0,
            "angular_speed": 1.0,
            "angular_accel": 3.0,
            "angular_jerk": 20.0,
        }
        short = duration_for_limits(0.1, 0.0, (0.0, 1.0), limits)
        long = duration_for_limits(0.2, 0.0, (0.0, 1.0), limits)
        self.assertGreater(long, short)
        self.assertGreaterEqual(short, 1.875 * 0.1 / limits["linear_speed"])


if __name__ == "__main__":
    unittest.main()
