"""Small dependency-free Cartesian trajectory utilities for the expert.

The semantic waypoints describe collision-safe geometry.  Interior points are
pass-through points: short quadratic fillets replace their sharp corners, and a
single quintic time law moves over the complete path.  Only the two ends have
zero velocity and acceleration.
"""

from dataclasses import dataclass

import numpy as np


def smoothstep5(x):
    """Quintic 0->1 time law and its derivative with respect to ``x``."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    value = x**3 * (10.0 + x * (-15.0 + 6.0 * x))
    derivative = 30.0 * x**2 * (1.0 - x)**2
    return value, derivative


@dataclass
class _Line:
    start: np.ndarray
    end: np.ndarray

    def __post_init__(self):
        delta = self.end - self.start
        self.length = float(np.linalg.norm(delta))
        self.direction = delta / self.length

    def at(self, distance):
        position = self.start + self.direction * np.clip(distance, 0.0, self.length)
        return position, self.direction

    def closest_distance(self, point):
        return float(np.clip(np.dot(point - self.start, self.direction), 0.0, self.length))


@dataclass
class _Quadratic:
    start: np.ndarray
    control: np.ndarray
    end: np.ndarray

    def __post_init__(self):
        # Arc-length lookup keeps speed constant around a fillet even though a
        # quadratic Bezier's parameter is not proportional to distance.
        self._u = np.linspace(0.0, 1.0, 65)
        points = np.array([self._position(u) for u in self._u])
        increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._s = np.concatenate([[0.0], np.cumsum(increments)])
        self.length = float(self._s[-1])

    def _position(self, u):
        return (1.0 - u)**2 * self.start + 2.0 * (1.0 - u) * u * self.control + u**2 * self.end

    def at(self, distance):
        u = float(np.interp(np.clip(distance, 0.0, self.length), self._s, self._u))
        tangent = 2.0 * ((1.0 - u) * (self.control - self.start) + u * (self.end - self.control))
        tangent /= np.linalg.norm(tangent)
        return self._position(u), tangent

    def closest_distance(self, point):
        points = np.array([self._position(u) for u in self._u])
        return float(self._s[np.argmin(np.linalg.norm(points - point, axis=1))])


class BlendedPath:
    """A polyline whose interior corners are replaced by tangent fillets."""

    def __init__(self, points, blend_radius):
        points = [np.asarray(p, dtype=float) for p in points]
        if len(points) < 2:
            raise ValueError("a path needs at least two points")
        if any(p.shape != (3,) for p in points):
            raise ValueError("path points must have shape (3,)")

        # Remove consecutive duplicates before computing corner directions.
        clean = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - clean[-1]) > 1e-9:
                clean.append(point)
        if len(clean) < 2:
            raise ValueError("path has zero length")

        segments = []
        cursor = clean[0]
        radius = max(float(blend_radius), 0.0)
        for i in range(1, len(clean) - 1):
            previous, corner, following = clean[i - 1:i + 2]
            incoming = corner - previous
            outgoing = following - corner
            in_len, out_len = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            in_dir, out_dir = incoming / in_len, outgoing / out_len

            # Collinear points need no fillet.  A quarter of each adjacent edge
            # prevents neighbouring fillets from consuming a whole short edge.
            trim = min(radius, 0.25 * in_len, 0.25 * out_len)
            direction_dot = np.dot(in_dir, out_dir)
            if trim > 1e-9 and -1.0 + 1e-8 < direction_dot < 1.0 - 1e-8:
                entry, exit = corner - trim * in_dir, corner + trim * out_dir
                if np.linalg.norm(entry - cursor) > 1e-9:
                    segments.append(_Line(cursor, entry))
                segments.append(_Quadratic(entry, corner, exit))
                cursor = exit
            else:
                if np.linalg.norm(corner - cursor) > 1e-9:
                    segments.append(_Line(cursor, corner))
                cursor = corner

        if np.linalg.norm(clean[-1] - cursor) > 1e-9:
            segments.append(_Line(cursor, clean[-1]))
        self.segments = segments
        self._ends = np.cumsum([segment.length for segment in segments])
        self.length = float(self._ends[-1])

    def at(self, distance):
        distance = float(np.clip(distance, 0.0, self.length))
        index = min(int(np.searchsorted(self._ends, distance, side="right")),
                    len(self.segments) - 1)
        start = 0.0 if index == 0 else self._ends[index - 1]
        return self.segments[index].at(distance - start)

    def closest_fraction(self, point):
        """Approximate the progress of the path location nearest ``point``."""
        point = np.asarray(point, dtype=float)
        best_norm, best_s, start = np.inf, 0.0, 0.0
        for segment in self.segments:
            local = segment.closest_distance(point)
            position, _ = segment.at(local)
            norm = np.linalg.norm(position - point)
            if norm < best_norm:
                best_norm, best_s = norm, start + local
            start += segment.length
        return best_s / self.length


def duration_for_limits(length, angle, rotation_window, limits):
    """Duration satisfying quintic linear and sampled angular limits."""
    duration = max(
        limits["min_duration"],
        1.875 * length / limits["linear_speed"],
        np.sqrt(5.7736 * length / limits["linear_accel"]),
        np.cbrt(60.0 * length / limits["linear_jerk"]),
    )
    if angle < 1e-9:
        return float(duration)

    lo, hi = rotation_window
    span = max(hi - lo, 1e-3)
    tau = np.linspace(0.0, 1.0, 4001)
    progress, _ = smoothstep5(tau)
    local, _ = smoothstep5((progress - lo) / span)
    dt = tau[1] - tau[0]
    d1 = np.gradient(local, dt)
    d2 = np.gradient(d1, dt)
    d3 = np.gradient(d2, dt)
    duration = max(
        duration,
        angle * np.max(np.abs(d1)) / limits["angular_speed"],
        np.sqrt(angle * np.max(np.abs(d2)) / limits["angular_accel"]),
        np.cbrt(angle * np.max(np.abs(d3)) / limits["angular_jerk"]),
    )
    return float(duration)
