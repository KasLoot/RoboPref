"""Convert open-vocabulary 2D detections into world-frame 3D anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from prefmem.execution.contracts import AnchorKind, ObjectReference
from prefmem.execution.frames import RGBDFrame
from prefmem.execution.sam import OpenVocabularyDetector, SamDetection


class GroundingError(RuntimeError):
    """A requested destination could not be grounded safely."""


@dataclass(frozen=True, slots=True)
class GroundedObject:
    query: str
    anchor: AnchorKind
    point_world: np.ndarray
    detection: SamDetection
    frame_sequence: int
    uncertainty_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be non-empty")
        try:
            anchor = AnchorKind(self.anchor)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid anchor: {self.anchor!r}") from error
        point = np.asarray(self.point_world, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_world must contain three finite coordinates")
        if not isinstance(self.detection, SamDetection):
            raise ValueError("detection must be SamDetection")
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative")
        if (
            isinstance(self.uncertainty_m, bool)
            or not isinstance(self.uncertainty_m, (int, float))
            or not math.isfinite(self.uncertainty_m)
            or self.uncertainty_m < 0
        ):
            raise ValueError("uncertainty_m must be finite and non-negative")
        point = point.copy()
        point.setflags(write=False)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "point_world", point)
        object.__setattr__(self, "uncertainty_m", float(self.uncertainty_m))


class RGBDGrounder:
    def __init__(
        self,
        detector: OpenVocabularyDetector,
        *,
        minimum_points: int = 12,
        maximum_uncertainty_m: float = 0.035,
    ) -> None:
        if not hasattr(detector, "detect"):
            raise TypeError("detector must expose detect()")
        if minimum_points < 1:
            raise ValueError("minimum_points must be positive")
        if maximum_uncertainty_m <= 0:
            raise ValueError("maximum_uncertainty_m must be positive")
        self.detector = detector
        self.minimum_points = int(minimum_points)
        self.maximum_uncertainty_m = float(maximum_uncertainty_m)

    def ground(self, frame: RGBDFrame, reference: ObjectReference) -> GroundedObject:
        if not isinstance(frame, RGBDFrame):
            raise TypeError("frame must be RGBDFrame")
        if not isinstance(reference, ObjectReference):
            raise TypeError("reference must be ObjectReference")
        detections = self.detector.detect(frame.rgb, reference.query)
        if not detections:
            raise GroundingError(f"SAM did not detect {reference.query!r}")
        if len(detections) > 1:
            ranked = sorted(
                detections,
                key=lambda item: (
                    -1.0 if item.score is None else item.score,
                    item.mask_area,
                ),
                reverse=True,
            )
            if ranked[0].score is None or ranked[1].score is None:
                raise GroundingError(
                    f"SAM returned multiple unranked matches for {reference.query!r}"
                )
            if ranked[0].score - ranked[1].score < 0.05:
                raise GroundingError(
                    f"SAM returned multiple ambiguous matches for {reference.query!r}"
                )
            detection = ranked[0]
        else:
            detection = detections[0]
        points = self._points_for_detection(frame, detection)
        point, uncertainty = self._anchor_point(points, reference.anchor)
        if uncertainty > self.maximum_uncertainty_m:
            raise GroundingError(
                f"3D grounding for {reference.query!r} is too uncertain "
                f"({uncertainty:.3f} m)"
            )
        return GroundedObject(
            query=reference.query,
            anchor=reference.anchor,
            point_world=point,
            detection=detection,
            frame_sequence=frame.sequence,
            uncertainty_m=uncertainty,
        )

    def count(self, frame: RGBDFrame, query: str) -> int:
        return len(self.detector.detect(frame.rgb, query))

    def _points_for_detection(
        self,
        frame: RGBDFrame,
        detection: SamDetection,
    ) -> np.ndarray:
        if detection.mask is not None:
            mask = detection.mask
        else:
            mask = np.zeros(frame.depth_m.shape, dtype=bool)
            x1, y1, x2, y2 = detection.box_xyxy
            # A conservative central crop avoids box-edge background depth.
            margin_x = max(1, int((x2 - x1) * 0.2))
            margin_y = max(1, int((y2 - y1) * 0.2))
            left, right = x1 + margin_x, x2 - margin_x
            top, bottom = y1 + margin_y, y2 - margin_y
            if right <= left or bottom <= top:
                left, right, top, bottom = x1, x2, y1, y2
            mask[top:bottom, left:right] = True
        ys, xs = np.nonzero(mask)
        if len(xs) < self.minimum_points:
            raise GroundingError("detection contains too few pixels for 3D grounding")
        depths = frame.depth_m[ys, xs].astype(np.float64)
        valid = np.isfinite(depths) & (depths > 0) & (depths < 10.0)
        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        if len(depths) < self.minimum_points:
            raise GroundingError("detection has too few valid depth samples")
        median = float(np.median(depths))
        mad = float(np.median(np.abs(depths - median)))
        tolerance = max(0.01, 4.0 * 1.4826 * mad)
        inliers = np.abs(depths - median) <= tolerance
        xs, ys, depths = xs[inliers], ys[inliers], depths[inliers]
        if len(depths) < self.minimum_points:
            raise GroundingError("depth filtering removed too many detection pixels")
        pixels = np.column_stack((xs.astype(float), ys.astype(float)))
        points = frame.calibration.backproject(pixels, depths)
        if len(points) > 5000:
            indices = np.linspace(0, len(points) - 1, 5000, dtype=int)
            points = points[indices]
        return points

    @staticmethod
    def _anchor_point(
        points: np.ndarray,
        anchor: AnchorKind,
    ) -> tuple[np.ndarray, float]:
        if anchor is AnchorKind.TOP_CENTER:
            top_threshold = float(np.quantile(points[:, 2], 0.8))
            selected = points[points[:, 2] >= top_threshold - 0.004]
            if len(selected) < 3:
                selected = points[np.argsort(points[:, 2])[-max(3, len(points) // 10) :]]
            point = np.array(
                [
                    np.median(selected[:, 0]),
                    np.median(selected[:, 1]),
                    np.median(selected[:, 2]),
                ]
            )
        elif anchor is AnchorKind.SURFACE_CENTER:
            # Image pixels are not uniformly distributed over an obliquely
            # viewed plane, so their median is biased toward the camera. Use
            # robust world-XY extents to recover the geometric surface centre;
            # small quantile trims reject antialiased mask-edge outliers while
            # remaining insensitive to holes caused by objects on the surface.
            lower, upper = np.quantile(points[:, :2], (0.02, 0.98), axis=0)
            point = np.array(
                [
                    0.5 * (lower[0] + upper[0]),
                    0.5 * (lower[1] + upper[1]),
                    np.median(points[:, 2]),
                ]
            )
        else:
            point = np.median(points, axis=0)
        # Do not treat an object's physical width as measurement uncertainty.
        # The residual in world Z is the useful signal for this tabletop
        # experiment: cube tops and the mat are locally horizontal, while bad
        # masks/background leakage produce a broad or multi-modal Z residual.
        residual = (
            np.abs(selected[:, 2] - point[2])
            if anchor is AnchorKind.TOP_CENTER
            else np.abs(points[:, 2] - point[2])
        )
        uncertainty = max(0.001, float(1.4826 * np.median(residual)))
        return point, uncertainty


__all__ = ["GroundedObject", "GroundingError", "RGBDGrounder"]
