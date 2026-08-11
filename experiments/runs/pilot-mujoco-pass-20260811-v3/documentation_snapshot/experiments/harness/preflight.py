"""Local-only model-service preflights with typed, durable evidence.

Preflights never open SSH connections or mutate a remote service.  They reuse
the configured loopback endpoint, require two consecutive health successes,
and then issue one minimal inference appropriate to the actual client protocol.
Transport/service availability is classified separately from a valid service
that fails the smoke-test semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import ipaddress
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from experiments.harness.recording import (
    durable_json,
    redact_text,
    sha256_file,
)


class ServiceKind(StrEnum):
    GEMMA = "gemma"
    EMBEDDING = "embedding"
    SAM = "sam"


class FailureClass(StrEnum):
    NONE = "NONE"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    SYSTEM = "SYSTEM"
    HARNESS = "HARNESS"


class ProbePhase(StrEnum):
    HEALTH = "HEALTH"
    SMOKE = "SMOKE"


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    kind: ServiceKind
    base_url: str
    label: str
    model_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ServiceKind):
            object.__setattr__(self, "kind", ServiceKind(self.kind))
        if not self.label.strip():
            raise ValueError("service label must be non-empty")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("service base_url must be absolute HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("service base_url may not contain credentials/query/fragment")
        hostname = parsed.hostname.casefold()
        if hostname != "localhost":
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError as error:
                raise ValueError("preflight endpoints must use a loopback host") from error
            if not address.is_loopback:
                raise ValueError("preflight endpoints must use a loopback host")

    @property
    def root_url(self) -> str:
        value = self.base_url.rstrip("/")
        return value[:-3] if value.endswith("/v1") else value

    @property
    def v1_url(self) -> str:
        value = self.base_url.rstrip("/")
        return value if value.endswith("/v1") else f"{value}/v1"


@dataclass(frozen=True, slots=True)
class PreflightPolicy:
    timeout_seconds: float = 5.0
    health_successes_required: int = 2
    maximum_attempts: int = 3
    retry_backoff_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if self.health_successes_required != 2:
            raise ValueError("protocol requires exactly two consecutive health successes")
        if (
            isinstance(self.maximum_attempts, bool)
            or not 2 <= self.maximum_attempts <= 3
        ):
            raise ValueError("maximum_attempts must be 2 or 3")
        if (
            not math.isfinite(self.retry_backoff_seconds)
            or self.retry_backoff_seconds < 0
        ):
            raise ValueError("retry_backoff_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    sequence: int
    service_label: str
    service_kind: ServiceKind
    phase: ProbePhase
    attempt: int
    utc: str
    monotonic_ns: int
    method: str
    path: str
    latency_seconds: float
    ok: bool
    failure_class: FailureClass
    status_code: int | None = None
    detail: str = ""
    model_ids: tuple[str, ...] = ()
    response_excerpt: str | None = None


@dataclass(frozen=True, slots=True)
class ServicePreflightResult:
    service_label: str
    service_kind: ServiceKind
    passed: bool
    failure_class: FailureClass
    model_ids: tuple[str, ...]
    health_successes: int
    smoke_passed: bool
    evidence: tuple[ProbeEvidence, ...]


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: int
    started_utc: str
    ended_utc: str
    passed: bool
    services: tuple[ServicePreflightResult, ...]

    @property
    def evidence(self) -> tuple[ProbeEvidence, ...]:
        return tuple(item for service in self.services for item in service.evidence)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "passed": self.passed,
            "services": [
                {
                    **asdict(service),
                    "service_kind": service.service_kind.value,
                    "failure_class": service.failure_class.value,
                    "evidence": [
                        {
                            **asdict(evidence),
                            "service_kind": evidence.service_kind.value,
                            "phase": evidence.phase.value,
                            "failure_class": evidence.failure_class.value,
                        }
                        for evidence in service.evidence
                    ],
                }
                for service in self.services
            ],
        }


@dataclass(slots=True)
class _ProbeOutcome:
    ok: bool
    failure_class: FailureClass
    detail: str
    status_code: int | None
    response_excerpt: str | None
    model_ids: tuple[str, ...] = ()


def _utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _excerpt(response: httpx.Response) -> str:
    media_type = response.headers.get("content-type", "").casefold()
    if "json" in media_type:
        try:
            value = response.json()
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except ValueError:
            text = response.text
    else:
        text = response.text
    return redact_text(text)[:1000]


def _http_failure(response: httpx.Response) -> _ProbeOutcome | None:
    if 200 <= response.status_code < 300:
        return None
    if response.status_code in {408, 425, 429} or response.status_code >= 500:
        failure = FailureClass.INFRASTRUCTURE
    else:
        # Authentication, missing route, and invalid request indicate a local
        # endpoint/harness contract issue, not experimental system behavior.
        failure = FailureClass.HARNESS
    return _ProbeOutcome(
        ok=False,
        failure_class=failure,
        detail=f"HTTP {response.status_code}",
        status_code=response.status_code,
        response_excerpt=_excerpt(response),
    )


def _transport_failure(error: Exception) -> _ProbeOutcome:
    return _ProbeOutcome(
        ok=False,
        failure_class=FailureClass.INFRASTRUCTURE,
        detail=f"{type(error).__name__}: {redact_text(str(error))}",
        status_code=None,
        response_excerpt=None,
    )


def _json_object(response: httpx.Response) -> tuple[dict[str, Any] | None, _ProbeOutcome | None]:
    try:
        payload = response.json()
    except ValueError as error:
        return None, _ProbeOutcome(
            ok=False,
            failure_class=FailureClass.HARNESS,
            detail=f"response is not JSON: {error}",
            status_code=response.status_code,
            response_excerpt=_excerpt(response),
        )
    if not isinstance(payload, dict):
        return None, _ProbeOutcome(
            ok=False,
            failure_class=FailureClass.HARNESS,
            detail="response is not a JSON object",
            status_code=response.status_code,
            response_excerpt=_excerpt(response),
        )
    return payload, None


def _model_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    data = payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return ()
    ids = []
    for item in data:
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return tuple(ids)


def _openai_health(
    client: httpx.Client, config: ServiceConfig
) -> tuple[str, str, _ProbeOutcome]:
    path = "/v1/models"
    try:
        response = client.get(f"{config.v1_url}/models")
    except httpx.HTTPError as error:
        return "GET", path, _transport_failure(error)
    failure = _http_failure(response)
    if failure is not None:
        return "GET", path, failure
    payload, failure = _json_object(response)
    if failure is not None or payload is None:
        return "GET", path, failure  # type: ignore[return-value]
    models = _model_ids(payload)
    if not models:
        return "GET", path, _ProbeOutcome(
            ok=False,
            failure_class=FailureClass.INFRASTRUCTURE,
            detail="health response advertised no model IDs",
            status_code=response.status_code,
            response_excerpt=_excerpt(response),
        )
    return "GET", path, _ProbeOutcome(
        ok=True,
        failure_class=FailureClass.NONE,
        detail="model registry healthy",
        status_code=response.status_code,
        response_excerpt=_excerpt(response),
        model_ids=models,
    )


def _sam_health(
    client: httpx.Client, config: ServiceConfig
) -> tuple[str, str, _ProbeOutcome]:
    path = "/health"
    try:
        response = client.get(f"{config.root_url}/health")
    except httpx.HTTPError as error:
        return "GET", path, _transport_failure(error)
    failure = _http_failure(response)
    if failure is not None:
        return "GET", path, failure
    payload, failure = _json_object(response)
    if failure is not None or payload is None:
        return "GET", path, failure  # type: ignore[return-value]
    if str(payload.get("status", "")).casefold() not in {"ok", "healthy", "ready"}:
        return "GET", path, _ProbeOutcome(
            ok=False,
            failure_class=FailureClass.INFRASTRUCTURE,
            detail="SAM health status is not ready",
            status_code=response.status_code,
            response_excerpt=_excerpt(response),
        )
    model = payload.get("model")
    models = (str(model),) if model else ()
    return "GET", path, _ProbeOutcome(
        ok=True,
        failure_class=FailureClass.NONE,
        detail="SAM health endpoint ready",
        status_code=response.status_code,
        response_excerpt=_excerpt(response),
        model_ids=models,
    )


def _select_model(config: ServiceConfig, advertised: Sequence[str]) -> str:
    if config.model_id:
        if advertised and config.model_id not in advertised:
            raise ValueError(
                f"configured model {config.model_id!r} is not advertised by {config.label}"
            )
        return config.model_id
    if not advertised:
        raise ValueError(f"{config.label} advertised no model IDs")
    return advertised[0]


def _gemma_smoke(
    client: httpx.Client, config: ServiceConfig, advertised: Sequence[str]
) -> tuple[str, str, _ProbeOutcome]:
    path = "/v1/chat/completions"
    try:
        model = _select_model(config, advertised)
    except ValueError as error:
        return "POST", path, _ProbeOutcome(
            False, FailureClass.HARNESS, str(error), None, None
        )
    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Preflight only. Reply with exactly OK.",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    try:
        response = client.post(f"{config.v1_url}/chat/completions", json=request)
    except httpx.HTTPError as error:
        return "POST", path, _transport_failure(error)
    failure = _http_failure(response)
    if failure is not None:
        return "POST", path, failure
    payload, failure = _json_object(response)
    if failure is not None or payload is None:
        return "POST", path, failure  # type: ignore[return-value]
    choices = payload.get("choices")
    content: object = None
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
    normalized = str(content or "").strip().strip(".`").casefold()
    if normalized != "ok":
        return "POST", path, _ProbeOutcome(
            ok=False,
            failure_class=FailureClass.SYSTEM,
            detail="Gemma did not satisfy the minimal deterministic OK instruction",
            status_code=response.status_code,
            response_excerpt=_excerpt(response),
            model_ids=(model,),
        )
    return "POST", path, _ProbeOutcome(
        True,
        FailureClass.NONE,
        "minimal chat completion passed",
        response.status_code,
        _excerpt(response),
        (model,),
    )


def _embedding_smoke(
    client: httpx.Client, config: ServiceConfig, advertised: Sequence[str]
) -> tuple[str, str, _ProbeOutcome]:
    path = "/v1/embeddings"
    try:
        model = _select_model(config, advertised)
    except ValueError as error:
        return "POST", path, _ProbeOutcome(
            False, FailureClass.HARNESS, str(error), None, None
        )
    try:
        response = client.post(
            f"{config.v1_url}/embeddings",
            json={"model": model, "input": ["RoboPref preflight"]},
        )
    except httpx.HTTPError as error:
        return "POST", path, _transport_failure(error)
    failure = _http_failure(response)
    if failure is not None:
        return "POST", path, failure
    payload, failure = _json_object(response)
    if failure is not None or payload is None:
        return "POST", path, failure  # type: ignore[return-value]
    data = payload.get("data")
    vector: object = None
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):
        vector = data[0].get("embedding")
    valid = (
        isinstance(vector, Sequence)
        and not isinstance(vector, (str, bytes))
        and len(vector) > 0
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in vector
        )
    )
    if not valid:
        return "POST", path, _ProbeOutcome(
            False,
            FailureClass.SYSTEM,
            "embedding response did not contain a finite non-empty vector",
            response.status_code,
            _excerpt(response),
            (model,),
        )
    return "POST", path, _ProbeOutcome(
        True,
        FailureClass.NONE,
        f"minimal embedding passed ({len(vector)} dimensions)",  # type: ignore[arg-type]
        response.status_code,
        _excerpt(response),
        (model,),
    )


def calibration_image_jpeg() -> bytes:
    """Return a deterministic, non-sensitive red-square calibration image."""

    image = np.full((96, 96, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (71, 71), (0, 0, 220), thickness=-1)
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )
    if not ok:
        raise RuntimeError("could not encode SAM calibration image")
    return encoded.tobytes()


def _sam_smoke(
    client: httpx.Client, config: ServiceConfig, advertised: Sequence[str]
) -> tuple[str, str, _ProbeOutcome]:
    path = "/detect"
    try:
        response = client.post(
            f"{config.root_url}/detect",
            data={"prompt": "red square", "threshold": "0.1"},
            files={"image": ("calibration.jpg", calibration_image_jpeg(), "image/jpeg")},
        )
    except httpx.HTTPError as error:
        return "POST", path, _transport_failure(error)
    failure = _http_failure(response)
    if failure is not None:
        return "POST", path, failure
    payload, failure = _json_object(response)
    if failure is not None or payload is None:
        return "POST", path, failure  # type: ignore[return-value]
    image = payload.get("image")
    detections = payload.get("detections")
    count = payload.get("count")
    contract_ok = (
        isinstance(image, Mapping)
        and image.get("width") == 96
        and image.get("height") == 96
        and isinstance(detections, Sequence)
        and not isinstance(detections, (str, bytes))
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count == len(detections)
    )
    if not contract_ok:
        return "POST", path, _ProbeOutcome(
            False,
            FailureClass.SYSTEM,
            "SAM response violated the detector schema/calibration dimensions",
            response.status_code,
            _excerpt(response),
            tuple(advertised),
        )
    return "POST", path, _ProbeOutcome(
        True,
        FailureClass.NONE,
        (
            "calibration segmentation request and response schema passed "
            f"({count} detection(s)); positive grounding is checked per scenario"
        ),
        response.status_code,
        _excerpt(response),
        tuple(advertised),
    )


class PreflightRunner:
    def __init__(
        self,
        *,
        policy: PreflightPolicy | None = None,
        client_factory: Any | None = None,
    ) -> None:
        self.policy = policy or PreflightPolicy()
        self._client_factory = client_factory
        self._sequence = 0

    def _client(self) -> httpx.Client:
        if self._client_factory is not None:
            return self._client_factory(self.policy.timeout_seconds)
        return httpx.Client(
            timeout=httpx.Timeout(self.policy.timeout_seconds), trust_env=False
        )

    def _evidence(
        self,
        config: ServiceConfig,
        phase: ProbePhase,
        attempt: int,
        method: str,
        path: str,
        started_ns: int,
        outcome: _ProbeOutcome,
    ) -> ProbeEvidence:
        self._sequence += 1
        return ProbeEvidence(
            sequence=self._sequence,
            service_label=config.label,
            service_kind=config.kind,
            phase=phase,
            attempt=attempt,
            utc=_utc(),
            monotonic_ns=time.monotonic_ns(),
            method=method,
            path=path,
            latency_seconds=(time.monotonic_ns() - started_ns) / 1_000_000_000,
            ok=outcome.ok,
            failure_class=outcome.failure_class,
            status_code=outcome.status_code,
            detail=redact_text(outcome.detail),
            model_ids=outcome.model_ids,
            response_excerpt=(
                None
                if outcome.response_excerpt is None
                else redact_text(outcome.response_excerpt)
            ),
        )

    def run_service(self, config: ServiceConfig) -> ServicePreflightResult:
        evidence: list[ProbeEvidence] = []
        advertised: tuple[str, ...] = ()
        consecutive = 0
        terminal_failure = FailureClass.INFRASTRUCTURE
        with self._client() as client:
            for attempt in range(1, self.policy.maximum_attempts + 1):
                started = time.monotonic_ns()
                if config.kind is ServiceKind.SAM:
                    method, path, outcome = _sam_health(client, config)
                else:
                    method, path, outcome = _openai_health(client, config)
                evidence.append(
                    self._evidence(
                        config,
                        ProbePhase.HEALTH,
                        attempt,
                        method,
                        path,
                        started,
                        outcome,
                    )
                )
                if outcome.ok:
                    consecutive += 1
                    advertised = outcome.model_ids
                    if consecutive >= self.policy.health_successes_required:
                        break
                else:
                    consecutive = 0
                    terminal_failure = outcome.failure_class
                    if outcome.failure_class in {FailureClass.HARNESS, FailureClass.SYSTEM}:
                        break
                if attempt < self.policy.maximum_attempts and self.policy.retry_backoff_seconds:
                    time.sleep(self.policy.retry_backoff_seconds)

            if consecutive < self.policy.health_successes_required:
                return ServicePreflightResult(
                    service_label=config.label,
                    service_kind=config.kind,
                    passed=False,
                    failure_class=terminal_failure,
                    model_ids=advertised,
                    health_successes=consecutive,
                    smoke_passed=False,
                    evidence=tuple(evidence),
                )

            started = time.monotonic_ns()
            if config.kind is ServiceKind.GEMMA:
                method, path, outcome = _gemma_smoke(client, config, advertised)
            elif config.kind is ServiceKind.EMBEDDING:
                method, path, outcome = _embedding_smoke(client, config, advertised)
            else:
                method, path, outcome = _sam_smoke(client, config, advertised)
            evidence.append(
                self._evidence(
                    config,
                    ProbePhase.SMOKE,
                    1,
                    method,
                    path,
                    started,
                    outcome,
                )
            )
            return ServicePreflightResult(
                service_label=config.label,
                service_kind=config.kind,
                passed=outcome.ok,
                failure_class=outcome.failure_class,
                model_ids=outcome.model_ids or advertised,
                health_successes=consecutive,
                smoke_passed=outcome.ok,
                evidence=tuple(evidence),
            )

    def run(self, configs: Iterable[ServiceConfig]) -> PreflightReport:
        self._sequence = 0
        started = _utc()
        services = tuple(self.run_service(config) for config in configs)
        if not services:
            raise ValueError("at least one service preflight is required")
        return PreflightReport(
            schema_version=1,
            started_utc=started,
            ended_utc=_utc(),
            passed=all(service.passed for service in services),
            services=services,
        )


def default_service_configs() -> tuple[ServiceConfig, ...]:
    return (
        ServiceConfig(
            ServiceKind.GEMMA,
            "http://127.0.0.1:8000/v1",
            "Gemma",
            "/workspace/models/gemma-4-26B-A4B-it",
        ),
        ServiceConfig(
            ServiceKind.EMBEDDING,
            "http://127.0.0.1:8080/v1",
            "EmbeddingGemma",
            "/data/models/embeddinggemma-300m",
        ),
        ServiceConfig(ServiceKind.SAM, "http://127.0.0.1:9000", "SAM 3.1"),
    )


def write_preflight_report(path: Path, report: PreflightReport) -> Path:
    durable_json(Path(path), report.to_json(), exclusive=True)
    return Path(path)


def write_recovery_bundle(
    campaign_root: str | Path,
    schedule_id: str,
    config: ServiceConfig,
    *,
    runner: PreflightRunner | None = None,
    tunnel_epoch: str | None = None,
) -> Path:
    """Run the frozen recovery gate and write immutable raw/report evidence.

    The latest attempt must already be sealed ``INFRA_INTERRUPTED``.  This
    helper never authorizes the retry: callers must separately pass the
    returned campaign-relative report path to
    :meth:`CampaignRegistry.authorize_infrastructure_retry`, which reopens and
    revalidates every byte under the registry lock.
    """

    from experiments.harness.campaign import CampaignRegistry, RunStatus

    root = Path(campaign_root).resolve()
    registry = CampaignRegistry(root)
    state = registry.snapshot()
    try:
        entry = state["entries"][schedule_id]
        attempt = entry["attempts"][-1]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("schedule has no interrupted attempt to recover") from error
    if attempt.get("status") != RunStatus.INFRA_INTERRUPTED.value:
        raise ValueError("latest attempt is not INFRA_INTERRUPTED")
    attempt_number = attempt.get("attempt_number")
    binding = attempt.get("infrastructure_evidence")
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or not isinstance(binding, Mapping)
        or not isinstance(binding.get("path"), str)
        or not isinstance(binding.get("sha256"), str)
    ):
        raise ValueError("interrupted attempt lacks an immutable evidence binding")
    evidence_path = root / str(attempt["artifact_path"]) / binding["path"]
    try:
        interruption = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read interrupted-attempt evidence") from error
    service_id = interruption.get("service_label") if isinstance(interruption, dict) else None
    if not isinstance(service_id, str) or not service_id:
        raise ValueError("interrupted-attempt evidence lacks a service label")
    if config.label != service_id:
        raise ValueError(
            "recovery service label does not match the interrupted service"
        )
    if tunnel_epoch is not None and (
        not isinstance(tunnel_epoch, str)
        or not tunnel_epoch.strip()
        or len(tunnel_epoch) > 128
        or any(character in tunnel_epoch for character in "\r\n")
    ):
        raise ValueError("tunnel_epoch must be a short, non-empty public label")

    result = (runner or PreflightRunner()).run_service(config)
    if not result.passed:
        raise RuntimeError(
            f"service recovery preflight failed as {result.failure_class.value}"
        )
    health = [item for item in result.evidence if item.phase is ProbePhase.HEALTH and item.ok]
    smoke = [item for item in result.evidence if item.phase is ProbePhase.SMOKE and item.ok]
    if len(health) < 2 or len(smoke) != 1:
        raise RuntimeError("recovery did not produce two health checks and one smoke test")
    selected = (health[-2], health[-1], smoke[0])

    recovery_dir = root / "environment" / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    token = f"{schedule_id}-a{attempt_number}"
    observations: list[dict[str, Any]] = []
    for sequence, (kind, evidence) in enumerate(
        zip(("health-1", "health-2", "smoke"), selected), start=1
    ):
        check_id = f"recovery-{attempt_number}-{sequence}-{kind}"
        raw_name = f"{token}-{kind}.json"
        raw_path = recovery_dir / raw_name
        raw_payload = {
            "schema_version": 1,
            "sequence": sequence,
            "observed_utc": evidence.utc,
            "passed": True,
            "service_id": service_id,
            "check_id": check_id,
            "non_mutating": kind == "smoke",
            "probe": {
                "service_kind": evidence.service_kind.value,
                "phase": evidence.phase.value,
                "method": evidence.method,
                "path": evidence.path,
                "latency_seconds": evidence.latency_seconds,
                "status_code": evidence.status_code,
                "model_ids": list(evidence.model_ids),
                "detail": evidence.detail,
            },
        }
        durable_json(raw_path, raw_payload, exclusive=True)
        relative = raw_path.relative_to(root).as_posix()
        summary = {
            "sequence": sequence,
            "observed_utc": evidence.utc,
            "passed": True,
            "evidence_path": relative,
            "evidence_sha256": sha256_file(raw_path),
            "service_id": service_id,
            "check_id": check_id,
        }
        if kind == "smoke":
            summary["non_mutating"] = True
        observations.append(summary)

    report_path = recovery_dir / f"{token}-report.json"
    durable_json(
        report_path,
        {
            "schema_version": 1,
            "schedule_id": schedule_id,
            "after_attempt_number": attempt_number,
            "interruption_evidence_sha256": binding["sha256"],
            "service_restored": True,
            "health_checks": observations[:2],
            "smoke_test": observations[2],
            **(
                {"tunnel_epoch": tunnel_epoch.strip()}
                if tunnel_epoch is not None
                else {}
            ),
        },
        exclusive=True,
    )
    return report_path


def classify_attempt_service_failure(
    report: PreflightReport,
) -> FailureClass:
    """Return the strongest blocker without conflating system and transport."""

    if report.passed:
        return FailureClass.NONE
    classes = {service.failure_class for service in report.services if not service.passed}
    if FailureClass.HARNESS in classes:
        return FailureClass.HARNESS
    if FailureClass.INFRASTRUCTURE in classes:
        return FailureClass.INFRASTRUCTURE
    if FailureClass.SYSTEM in classes:
        return FailureClass.SYSTEM
    return FailureClass.HARNESS


__all__ = [
    "FailureClass",
    "PreflightPolicy",
    "PreflightReport",
    "PreflightRunner",
    "ProbeEvidence",
    "ProbePhase",
    "ServiceConfig",
    "ServiceKind",
    "ServicePreflightResult",
    "calibration_image_jpeg",
    "classify_attempt_service_failure",
    "default_service_configs",
    "write_preflight_report",
    "write_recovery_bundle",
]
