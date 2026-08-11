"""Executable topology assembly without contacting remote model services.

The assembler creates exactly one :class:`ModelSession` for every actual model
instance declared by an experiment profile.  Logical-role routers share that
session (and its serialized context ledger) only when the frozen profile maps
them to the same instance ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

from .profiles import (
    LOGICAL_ROLES,
    ExperimentProfile,
    assert_executor_allowed,
    load_profiles,
)


class AssemblyError(RuntimeError):
    """Raised before an invalid or ambiguous experiment system is built."""


class ModelSurfaceError(RuntimeError):
    """Raised when an injected backend lacks an advertised model surface."""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    sequence: int
    call_id: str
    session_id: str
    model_instance_id: str
    logical_role: str
    phase: str
    surface: str
    request: Any
    response: Any
    ok: bool
    error_type: str | None
    started_at_ns: int
    completed_at_ns: int


class SessionObserver(Protocol):
    def __call__(self, record: SessionRecord) -> None: ...


class ModelSession:
    """One serialized backend and one cross-phase context ledger.

    ``invoke``, iteration of ``stream``, and ``bind_tools`` are all protected by
    the same re-entrant lock.  A merged topology therefore cannot interleave
    calls from two logical roles on one stateful model backend.
    """

    def __init__(
        self,
        *,
        profile_id: str,
        model_instance_id: str,
        backend: object,
        observer: SessionObserver | None = None,
    ) -> None:
        if not isinstance(model_instance_id, str) or not model_instance_id:
            raise ValueError("model_instance_id must be non-empty")
        if backend is None:
            raise ValueError("backend must not be None")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        self.profile_id = profile_id
        self.model_instance_id = model_instance_id
        self.session_id = f"{profile_id}:{model_instance_id}"
        self.backend = backend
        self._observer = observer
        self._lock = threading.RLock()
        self._ledger: list[SessionRecord] = []
        self._next_sequence = 1

    def snapshot(self) -> tuple[SessionRecord, ...]:
        with self._lock:
            return tuple(self._ledger)

    def _recorded_call(
        self,
        *,
        logical_role: str,
        phase: str,
        surface: str,
        request: Any,
        operation: Callable[[], Any],
    ) -> Any:
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            started = time.monotonic_ns()
            response: Any = None
            error_type: str | None = None
            ok = False
            try:
                response = operation()
                ok = True
                return response
            except BaseException as error:
                error_type = error.__class__.__name__
                raise
            finally:
                record = SessionRecord(
                    sequence=sequence,
                    call_id=f"{self.session_id}:{sequence:06d}",
                    session_id=self.session_id,
                    model_instance_id=self.model_instance_id,
                    logical_role=logical_role,
                    phase=phase,
                    surface=surface,
                    request=request,
                    response=response,
                    ok=ok,
                    error_type=error_type,
                    started_at_ns=started,
                    completed_at_ns=time.monotonic_ns(),
                )
                self._ledger.append(record)
                if self._observer is not None:
                    self._observer(record)

    @staticmethod
    def _surface(target: object, name: str) -> Callable[..., Any]:
        surface = getattr(target, name, None)
        if not callable(surface):
            raise ModelSurfaceError(
                f"backend {type(target).__name__} does not expose {name}()"
            )
        return surface

    def invoke(
        self,
        target: object,
        *,
        logical_role: str,
        phase: str,
        model_input: Any,
        kwargs: Mapping[str, Any],
    ) -> Any:
        surface = self._surface(target, "invoke")
        return self._recorded_call(
            logical_role=logical_role,
            phase=phase,
            surface="invoke",
            request={"input": model_input, "kwargs": dict(kwargs)},
            operation=lambda: surface(model_input, **kwargs),
        )

    def stream(
        self,
        target: object,
        *,
        logical_role: str,
        phase: str,
        model_input: Any,
        kwargs: Mapping[str, Any],
    ) -> Iterator[Any]:
        """Iterate a backend stream while retaining the session lock."""

        surface = self._surface(target, "stream")

        def generate() -> Iterator[Any]:
            with self._lock:
                sequence = self._next_sequence
                self._next_sequence += 1
                started = time.monotonic_ns()
                chunks: list[Any] = []
                error_type: str | None = None
                ok = False
                try:
                    for chunk in surface(model_input, **kwargs):
                        chunks.append(chunk)
                        yield chunk
                    ok = True
                except BaseException as error:
                    error_type = error.__class__.__name__
                    raise
                finally:
                    record = SessionRecord(
                        sequence=sequence,
                        call_id=f"{self.session_id}:{sequence:06d}",
                        session_id=self.session_id,
                        model_instance_id=self.model_instance_id,
                        logical_role=logical_role,
                        phase=phase,
                        surface="stream",
                        request={"input": model_input, "kwargs": dict(kwargs)},
                        response=tuple(chunks),
                        ok=ok,
                        error_type=error_type,
                        started_at_ns=started,
                        completed_at_ns=time.monotonic_ns(),
                    )
                    self._ledger.append(record)
                    if self._observer is not None:
                        self._observer(record)

        return generate()

    def bind_tools(
        self,
        target: object,
        *,
        logical_role: str,
        phase: str,
        tools: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> object:
        tool_tuple = tuple(tools)
        surface = self._surface(target, "bind_tools")
        return self._recorded_call(
            logical_role=logical_role,
            phase=phase,
            surface="bind_tools",
            request={
                "tool_names": tuple(
                    str(getattr(tool, "name", type(tool).__name__)) for tool in tool_tuple
                ),
                "kwargs": dict(kwargs),
            },
            operation=lambda: surface(list(tool_tuple), **kwargs),
        )


class RoutedModel:
    """Logical-role model surface routed through a shared session."""

    def __init__(
        self,
        session: ModelSession,
        logical_role: str,
        *,
        phase: str,
        target: object | None = None,
    ) -> None:
        if logical_role not in LOGICAL_ROLES:
            raise ValueError(f"unknown logical role: {logical_role}")
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be non-empty")
        self._session = session
        self.logical_role = logical_role
        self.default_phase = phase
        self._target = session.backend if target is None else target

    @property
    def model_instance_id(self) -> str:
        return self._session.model_instance_id

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def context_ledger(self) -> tuple[SessionRecord, ...]:
        return self._session.snapshot()

    @property
    def backend(self) -> object:
        """The sole root backend for identity audits; calls must use the router."""

        return self._session.backend

    def for_phase(self, phase: str) -> "RoutedModel":
        return RoutedModel(
            self._session,
            self.logical_role,
            phase=phase,
            target=self._target,
        )

    def _phase(self, kwargs: dict[str, Any]) -> str:
        phase = kwargs.pop("_experiment_phase", self.default_phase)
        if not isinstance(phase, str) or not phase:
            raise ValueError("_experiment_phase must be a non-empty string")
        return phase

    def invoke(self, model_input: Any, **kwargs: Any) -> Any:
        phase = self._phase(kwargs)
        return self._session.invoke(
            self._target,
            logical_role=self.logical_role,
            phase=phase,
            model_input=model_input,
            kwargs=kwargs,
        )

    def stream(self, model_input: Any, **kwargs: Any) -> Iterator[Any]:
        phase = self._phase(kwargs)
        return self._session.stream(
            self._target,
            logical_role=self.logical_role,
            phase=phase,
            model_input=model_input,
            kwargs=kwargs,
        )

    def bind_tools(self, tools: Iterable[Any], **kwargs: Any) -> "RoutedModel":
        phase = self._phase(kwargs)
        bound = self._session.bind_tools(
            self._target,
            logical_role=self.logical_role,
            phase=phase,
            tools=tools,
            kwargs=kwargs,
        )
        return RoutedModel(
            self._session,
            self.logical_role,
            phase=self.default_phase,
            target=bound,
        )


@dataclass(frozen=True, slots=True)
class BackendBuildRequest:
    profile: ExperimentProfile
    model_instance_id: str
    logical_roles: tuple[str, ...]
    factory_ordinal: int


@dataclass(frozen=True, slots=True)
class EmbeddingBuildRequest:
    profile: ExperimentProfile
    logical_role: str


@dataclass(frozen=True, slots=True)
class RoleBuildRequest:
    profile: ExperimentProfile
    logical_role: str
    model_instance_id: str
    model: RoutedModel
    prompt_ref: str
    schema_ref: str
    embedding_model: object | None
    existing_components: Mapping[str, object]


BackendFactory = Callable[[BackendBuildRequest], object]
EmbeddingFactory = Callable[[EmbeddingBuildRequest], object]
ModelFactory = Callable[[RoleBuildRequest], object]


@dataclass(frozen=True, slots=True)
class DefaultRoleInterface:
    """Executable model boundary used when no production role factory is supplied."""

    logical_role: str
    model_instance_id: str
    model: RoutedModel
    prompt_ref: str
    schema_ref: str
    embedding_model: object | None = None

    def invoke_phase(self, phase: str, payload: Any, **kwargs: Any) -> Any:
        return self.model.invoke(payload, _experiment_phase=phase, **kwargs)

    def stream_phase(self, phase: str, payload: Any, **kwargs: Any) -> Iterator[Any]:
        return self.model.stream(payload, _experiment_phase=phase, **kwargs)

    def bind_tools(self, tools: Iterable[Any], **kwargs: Any) -> "DefaultRoleInterface":
        return DefaultRoleInterface(
            logical_role=self.logical_role,
            model_instance_id=self.model_instance_id,
            model=self.model.bind_tools(tools, **kwargs),
            prompt_ref=self.prompt_ref,
            schema_ref=self.schema_ref,
            embedding_model=self.embedding_model,
        )


@dataclass(frozen=True, slots=True)
class InstanceManifest:
    model_instance_id: str
    session_id: str
    logical_roles: tuple[str, ...]
    backend_type: str
    factory_ordinal: int
    context_ledger_id: str
    prompt_refs: Mapping[str, str]
    schema_refs: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_instance_id": self.model_instance_id,
            "session_id": self.session_id,
            "logical_roles": list(self.logical_roles),
            "backend_type": self.backend_type,
            "factory_ordinal": self.factory_ordinal,
            "context_ledger_id": self.context_ledger_id,
            "prompt_refs": dict(self.prompt_refs),
            "schema_refs": dict(self.schema_refs),
        }


@dataclass(frozen=True, slots=True)
class TopologyManifest:
    profile_id: str
    profile_kind: str
    declared_unique_agent_count: int
    assembled_unique_agent_count: int
    logical_to_instance: Mapping[str, str | None]
    instances: tuple[InstanceManifest, ...]
    assertions: Mapping[str, bool]

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind,
            "declared_unique_agent_count": self.declared_unique_agent_count,
            "assembled_unique_agent_count": self.assembled_unique_agent_count,
            "logical_to_instance": dict(self.logical_to_instance),
            "instances": [instance.to_dict() for instance in self.instances],
            "assertions": dict(self.assertions),
        }
        if include_sha256:
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class AssembledSystem:
    profile: ExperimentProfile
    executor: str
    sessions: Mapping[str, ModelSession]
    routed_models: Mapping[str, RoutedModel | None]
    components: Mapping[str, object]
    variant: Any
    topology_manifest: TopologyManifest
    embedding_model: object | None = None

    def component(self, logical_role: str) -> object:
        try:
            return self.components[logical_role]
        except KeyError as error:
            raise AssemblyError(f"unknown logical role: {logical_role}") from error


class SystemAssembler:
    """Build one executable system from a frozen profile.

    Safety checks happen before any factory is called.  Thus an unsupported
    robot/human/profile combination cannot initialize a backend, embedding
    model, or source component as a side effect.
    """

    _DEFAULT_PHASE = {
        "hri": "INTERACT",
        "memory": "MEMORY",
        "planner": "PLAN",
        "monitor": "STEP_ASSESS",
        "validator": "FINAL_ASSESS",
    }
    _BUILD_ORDER = ("memory", "planner", "monitor", "validator", "hri")

    def __init__(
        self,
        profile: str | ExperimentProfile,
        *,
        executor: str,
        backend_factory: BackendFactory | None,
        model_factory: ModelFactory | None = None,
        embedding_factory: EmbeddingFactory | None = None,
        observer: SessionObserver | None = None,
        scenario_allowed_executors: frozenset[str] | None = None,
        safety_screened: bool = False,
        ethics_approved: bool = False,
    ) -> None:
        self.profile = (
            load_profiles().get(profile) if isinstance(profile, str) else profile
        )
        if not isinstance(self.profile, ExperimentProfile):
            raise TypeError("profile must be an ID or ExperimentProfile")
        self.executor = executor
        self.backend_factory = backend_factory
        self.model_factory = model_factory or self._default_model_factory
        self.embedding_factory = embedding_factory
        self.observer = observer
        self.scenario_allowed_executors = scenario_allowed_executors
        self.safety_screened = safety_screened
        self.ethics_approved = ethics_approved

    @staticmethod
    def _default_model_factory(request: RoleBuildRequest) -> DefaultRoleInterface:
        return DefaultRoleInterface(
            logical_role=request.logical_role,
            model_instance_id=request.model_instance_id,
            model=request.model,
            prompt_ref=request.prompt_ref,
            schema_ref=request.schema_ref,
            embedding_model=request.embedding_model,
        )

    def _validate_before_factories(self) -> None:
        assert_executor_allowed(
            self.profile,
            self.executor,
            scenario_allowed=self.scenario_allowed_executors,
            safety_screened=self.safety_screened,
            ethics_approved=self.ethics_approved,
        )
        from .variants import validate_profile_modes

        validate_profile_modes(self.profile, executor=self.executor)
        if self.profile.unique_agent_count and self.backend_factory is None:
            raise AssemblyError("an active profile requires a backend_factory")

    def assemble(self) -> AssembledSystem:
        self._validate_before_factories()
        grouped_roles: dict[str, list[str]] = {}
        for role in LOGICAL_ROLES:
            instance_id = self.profile.logical_to_instance[role]
            if instance_id is not None:
                grouped_roles.setdefault(instance_id, []).append(role)

        sessions: dict[str, ModelSession] = {}
        backend_owners: dict[int, str] = {}
        for ordinal, (instance_id, roles) in enumerate(grouped_roles.items(), start=1):
            if self.backend_factory is None:
                raise AssemblyError("active instance has no backend_factory")
            backend = self.backend_factory(
                BackendBuildRequest(
                    profile=self.profile,
                    model_instance_id=instance_id,
                    logical_roles=tuple(roles),
                    factory_ordinal=ordinal,
                )
            )
            owner = backend_owners.get(id(backend))
            if owner is not None:
                raise AssemblyError(
                    f"backend_factory reused one backend for distinct instances "
                    f"{owner} and {instance_id}"
                )
            backend_owners[id(backend)] = instance_id
            sessions[instance_id] = ModelSession(
                profile_id=self.profile.profile_id,
                model_instance_id=instance_id,
                backend=backend,
                observer=self.observer,
            )

        if len(sessions) != self.profile.unique_agent_count:
            raise AssemblyError(
                f"assembled {len(sessions)} sessions for a profile declaring "
                f"{self.profile.unique_agent_count} agents"
            )

        routed: dict[str, RoutedModel | None] = {}
        for role in LOGICAL_ROLES:
            instance_id = self.profile.logical_to_instance[role]
            routed[role] = (
                None
                if instance_id is None
                else RoutedModel(
                    sessions[instance_id], role, phase=self._DEFAULT_PHASE[role]
                )
            )

        embedding_model = None
        if routed["memory"] is not None and self.embedding_factory is not None:
            embedding_model = self.embedding_factory(
                EmbeddingBuildRequest(profile=self.profile, logical_role="memory")
            )

        from .variants import BypassedRole, build_variant_runtime

        components: dict[str, object] = {}
        component_owners: dict[int, str] = {}
        for role in self._BUILD_ORDER:
            model = routed[role]
            if model is None:
                mechanism_key = "planning" if role == "planner" else role
                components[role] = BypassedRole(
                    logical_role=role,
                    reason=self.profile.mechanism_modes[mechanism_key],
                )
                continue
            prompt_ref = self.profile.prompt_refs[role]
            schema_ref = self.profile.schema_refs[role]
            if prompt_ref is None or schema_ref is None:
                raise AssemblyError(f"active role {role} lacks prompt/schema references")
            component = self.model_factory(
                RoleBuildRequest(
                    profile=self.profile,
                    logical_role=role,
                    model_instance_id=model.model_instance_id,
                    model=model,
                    prompt_ref=prompt_ref,
                    schema_ref=schema_ref,
                    embedding_model=embedding_model if role == "memory" else None,
                    existing_components=MappingProxyType(dict(components)),
                )
            )
            if component is None:
                raise AssemblyError(f"model_factory returned None for active role {role}")
            existing_owner = component_owners.get(id(component))
            if existing_owner is not None and existing_owner != model.model_instance_id:
                raise AssemblyError(
                    "model_factory reused one component across distinct model instances "
                    f"{existing_owner} and {model.model_instance_id}"
                )
            component_owners[id(component)] = model.model_instance_id
            components[role] = component

        instance_manifests = tuple(
            InstanceManifest(
                model_instance_id=instance_id,
                session_id=session.session_id,
                logical_roles=tuple(grouped_roles[instance_id]),
                backend_type=(
                    f"{type(session.backend).__module__}."
                    f"{type(session.backend).__qualname__}"
                ),
                factory_ordinal=ordinal,
                context_ledger_id=hashlib.sha256(
                    f"{session.session_id}:context-ledger-v1".encode("utf-8")
                ).hexdigest(),
                prompt_refs=MappingProxyType(
                    {
                        role: str(self.profile.prompt_refs[role])
                        for role in grouped_roles[instance_id]
                    }
                ),
                schema_refs=MappingProxyType(
                    {
                        role: str(self.profile.schema_refs[role])
                        for role in grouped_roles[instance_id]
                    }
                ),
            )
            for ordinal, (instance_id, session) in enumerate(sessions.items(), start=1)
        )
        manifest = TopologyManifest(
            profile_id=self.profile.profile_id,
            profile_kind=self.profile.kind,
            declared_unique_agent_count=self.profile.unique_agent_count,
            assembled_unique_agent_count=len(sessions),
            logical_to_instance=MappingProxyType(
                dict(self.profile.logical_to_instance)
            ),
            instances=instance_manifests,
            assertions=MappingProxyType(
                {
                    "mapping_matches_profile": True,
                    "one_root_backend_per_instance": True,
                    "distinct_backends_across_instances": True,
                    "one_context_ledger_per_instance": True,
                    "no_component_cross_instance_aliasing": True,
                }
            ),
        )
        variant = build_variant_runtime(
            self.profile,
            executor=self.executor,
            components=MappingProxyType(dict(components)),
        )
        return AssembledSystem(
            profile=self.profile,
            executor=self.executor,
            sessions=MappingProxyType(sessions),
            routed_models=MappingProxyType(routed),
            components=MappingProxyType(components),
            variant=variant,
            topology_manifest=manifest,
            embedding_model=embedding_model,
        )
