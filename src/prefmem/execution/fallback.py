"""Explicit experiment-only grounding assistance contracts.

The production execution path remains SAM-only unless a caller supplies both
an oracle provider and the matching opt-in flag.  These types keep a failed
SAM grounding and the downstream oracle-assisted continuation as two distinct
outcomes; assistance can never turn the strict result into a pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from prefmem.execution.contracts import AnchorKind, ObjectReference
from prefmem.execution.frames import RGBDFrame


SIMULATOR_GROUND_TRUTH_SOURCE = "SIMULATOR_GROUND_TRUTH"
SAM_GROUNDING_SOURCE = "SAM_3_1"


class StrictSystemResult(StrEnum):
    PASS = "PASS"
    FAIL_GROUNDING = "FAIL_GROUNDING"
    FAIL_EXECUTION = "FAIL_EXECUTION"
    CANCELLED = "CANCELLED"


class AssistedContinuationResult(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PASS = "PASS"
    FAIL = "FAIL"
    CANCELLED = "CANCELLED"


class DownstreamExecutionResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    CANCELLED = "CANCELLED"


class OracleGroundingError(RuntimeError):
    """Simulator truth could not safely resolve the failed reference."""


@dataclass(frozen=True, slots=True)
class OracleGrounding:
    """One explicitly labelled simulator-truth anchor.

    ``frame_sequence`` associates the assistance with the SAM request that
    failed.  Provider diagnostics should separately record the simulator time
    at which hidden truth was read.
    """

    query: str
    anchor: AnchorKind
    point_world: np.ndarray
    frame_sequence: int
    source_id: str = SIMULATOR_GROUND_TRUTH_SOURCE
    diagnostics: Mapping[str, Any] | None = None

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
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative")
        if self.source_id != SIMULATOR_GROUND_TRUTH_SOURCE:
            raise ValueError(
                "oracle grounding source must be SIMULATOR_GROUND_TRUTH"
            )
        point = point.copy()
        point.setflags(write=False)
        diagnostics = (
            {}
            if self.diagnostics is None
            else copy.deepcopy(dict(self.diagnostics))
        )
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "point_world", point)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))


class OracleGroundingProvider(Protocol):
    """Experiment seam that may read simulator truth after a strict failure."""

    source_id: str

    def ground_from_simulator_truth(
        self,
        frame: RGBDFrame,
        reference: ObjectReference,
        *,
        role: str,
        strict_failure_reason_code: str,
    ) -> OracleGrounding: ...


@dataclass(frozen=True, slots=True)
class GroundingAttempt:
    role: str
    query: str
    anchor: str
    strict_grounding_result: str
    strict_failure_reason_code: str | None
    strict_failure_message: str | None
    grounding_source: str | None
    oracle_fallback_used: bool
    point_world_m: tuple[float, float, float] | None
    frame_sequence: int

    def __post_init__(self) -> None:
        if self.role not in {"source", "target"}:
            raise ValueError("grounding role must be source or target")
        if self.strict_grounding_result not in {"PASS", "FAIL"}:
            raise ValueError("strict grounding result must be PASS or FAIL")
        if type(self.oracle_fallback_used) is not bool:
            raise TypeError("oracle_fallback_used must be bool")
        if self.oracle_fallback_used:
            if self.strict_grounding_result != "FAIL":
                raise ValueError("oracle assistance requires a strict failure")
            if self.grounding_source != SIMULATOR_GROUND_TRUTH_SOURCE:
                raise ValueError("oracle assistance must name its simulator source")
        if self.point_world_m is not None and (
            len(self.point_world_m) != 3
            or not all(math.isfinite(value) for value in self.point_world_m)
        ):
            raise ValueError("point_world_m must contain three finite values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "query": self.query,
            "anchor": self.anchor,
            "strict_grounding_result": self.strict_grounding_result,
            "strict_failure_reason_code": self.strict_failure_reason_code,
            "strict_failure_message": self.strict_failure_message,
            "grounding_source": self.grounding_source,
            "oracle_fallback_used": self.oracle_fallback_used,
            "point_world_m": (
                None
                if self.point_world_m is None
                else list(self.point_world_m)
            ),
            "frame_sequence": self.frame_sequence,
        }


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    publication_id: str
    terminal_state: str
    strict_system_result: StrictSystemResult
    oracle_fallback_used: bool
    oracle_fallback_source: str | None
    assisted_continuation_result: AssistedContinuationResult
    downstream_execution_result: DownstreamExecutionResult
    grounding_attempts: tuple[GroundingAttempt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.publication_id, str) or not self.publication_id:
            raise ValueError("publication_id must be non-empty")
        if type(self.oracle_fallback_used) is not bool:
            raise TypeError("oracle_fallback_used must be bool")
        if self.oracle_fallback_used:
            if self.oracle_fallback_source != SIMULATOR_GROUND_TRUTH_SOURCE:
                raise ValueError("assisted outcome must name simulator truth")
            if self.strict_system_result is not StrictSystemResult.FAIL_GROUNDING:
                raise ValueError("assisted outcome must retain FAIL_GROUNDING")
            if (
                self.assisted_continuation_result
                is AssistedContinuationResult.NOT_APPLICABLE
            ):
                raise ValueError("assisted outcome requires a continuation result")
        elif self.oracle_fallback_source is not None:
            raise ValueError("unassisted outcome cannot name an oracle source")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "publication_id": self.publication_id,
            "terminal_state": self.terminal_state,
            "strict_system_result": self.strict_system_result.value,
            "oracle_fallback_used": self.oracle_fallback_used,
            "oracle_fallback_source": self.oracle_fallback_source,
            "assisted_continuation_result": (
                self.assisted_continuation_result.value
            ),
            "downstream_execution_result": (
                self.downstream_execution_result.value
            ),
            "grounding_attempts": [
                attempt.to_dict() for attempt in self.grounding_attempts
            ],
        }


__all__ = [
    "AssistedContinuationResult",
    "DownstreamExecutionResult",
    "ExecutionOutcome",
    "GroundingAttempt",
    "OracleGrounding",
    "OracleGroundingError",
    "OracleGroundingProvider",
    "SAM_GROUNDING_SOURCE",
    "SIMULATOR_GROUND_TRUTH_SOURCE",
    "StrictSystemResult",
]
