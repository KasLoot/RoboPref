"""Explicit MuJoCo hidden-state provider for experiment-only continuation."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from prefmem.execution.contracts import AnchorKind, ObjectReference
from prefmem.execution.fallback import (
    OracleGrounding,
    OracleGroundingError,
    SIMULATOR_GROUND_TRUTH_SOURCE,
)
from prefmem.execution.frames import RGBDFrame


_COLOURS = ("red", "green", "blue", "yellow", "purple", "orange")


def _normalise_query(value: str) -> str:
    return " ".join(re.sub(r"[_-]+", " ", value.casefold()).split())


class MuJoCoOracleGroundingProvider:
    """Resolve exact supported selectors from MuJoCo state after SAM fails.

    This class is never constructed by the default runtime.  It intentionally
    supports only the suite's frozen cube/block and board/mat selectors, so a
    novel or ambiguous language query cannot silently gain hidden supervision.
    """

    source_id = SIMULATOR_GROUND_TRUTH_SOURCE

    def __init__(self, environment: Any) -> None:
        for attribute in ("model", "data", "_lock"):
            if not hasattr(environment, attribute):
                raise TypeError(
                    "environment must expose synchronized MuJoCo model/data state"
                )
        self.environment = environment

    def ground_from_simulator_truth(
        self,
        frame: RGBDFrame,
        reference: ObjectReference,
        *,
        role: str,
        strict_failure_reason_code: str,
    ) -> OracleGrounding:
        if not isinstance(frame, RGBDFrame):
            raise TypeError("frame must be RGBDFrame")
        if not isinstance(reference, ObjectReference):
            raise TypeError("reference must be ObjectReference")
        if role not in {"source", "target"}:
            raise ValueError("role must be source or target")
        if not strict_failure_reason_code:
            raise ValueError("strict_failure_reason_code is required")

        geom_name = self._geom_name(reference.query)
        environment = self.environment
        with environment._lock:
            geom_id = mujoco.mj_name2id(
                environment.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            if geom_id < 0:
                raise OracleGroundingError(
                    f"simulator scene has no exact oracle geom {geom_name!r}"
                )
            if int(environment.model.geom_type[geom_id]) != int(
                mujoco.mjtGeom.mjGEOM_BOX
            ):
                raise OracleGroundingError(
                    f"oracle geom {geom_name!r} is not a supported box"
                )
            centre = environment.data.geom_xpos[geom_id].astype(np.float64).copy()
            rotation = environment.data.geom_xmat[geom_id].reshape(3, 3)
            if reference.anchor is AnchorKind.CENTER:
                point = centre
                algorithm = "mujoco_geom_center"
            else:
                local_up = rotation[:, 2]
                half_height = float(environment.model.geom_size[geom_id, 2])
                point = centre + local_up * half_height
                algorithm = "mujoco_box_local_positive_z_face_center"
            truth_simulation_time = float(environment.data.time)

        return OracleGrounding(
            query=reference.query,
            anchor=reference.anchor,
            point_world=point,
            frame_sequence=frame.sequence,
            diagnostics={
                "source_id": self.source_id,
                "role": role,
                "strict_failure_reason_code": strict_failure_reason_code,
                "query_normalized": _normalise_query(reference.query),
                "resolved_geom_name": geom_name,
                "anchor_algorithm": algorithm,
                "associated_sam_frame_sequence": frame.sequence,
                "associated_sam_frame_simulation_time": frame.simulation_time,
                "oracle_truth_simulation_time": truth_simulation_time,
                "point_world_m": point.astype(float).tolist(),
            },
        )

    @staticmethod
    def _geom_name(query: str) -> str:
        normalized = _normalise_query(query)
        for colour in _COLOURS:
            if normalized in {f"{colour} cube", f"{colour} block"}:
                return f"{colour}_block_geom"
        surfaces = {
            "white board": "white_board",
            "cyan board": "cyan_board",
            "white mat": "white_mat",
        }
        try:
            return surfaces[normalized]
        except KeyError as error:
            raise OracleGroundingError(
                "simulator oracle has no exact selector mapping for "
                f"{query!r}"
            ) from error


__all__ = ["MuJoCoOracleGroundingProvider"]
