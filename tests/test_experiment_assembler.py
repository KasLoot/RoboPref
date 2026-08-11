from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from experiments.harness.assembler import (
    AssemblyError,
    DefaultRoleInterface,
    SystemAssembler,
)
from experiments.harness.profiles import UnsafeExecutorError
from experiments.harness.variants import BypassedRole


class FakeBackend:
    def __init__(self, instance_id: str, *, delay: float = 0.0) -> None:
        self.instance_id = instance_id
        self.delay = delay
        self.calls: list[tuple[str, object]] = []
        self._state_lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def _enter(self) -> None:
        with self._state_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)

    def _leave(self) -> None:
        with self._state_lock:
            self.active_calls -= 1

    def invoke(self, model_input, **kwargs):
        self._enter()
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(("invoke", model_input))
            return {
                "echo": model_input,
                "actions": ["pick:red", "place:target"],
                "complete": True,
                "kwargs": kwargs,
            }
        finally:
            self._leave()

    def stream(self, model_input, **kwargs):
        self._enter()
        try:
            self.calls.append(("stream", model_input))
            yield {"chunk": 1, "input": model_input, "kwargs": kwargs}
            yield {"chunk": 2, "input": model_input, "kwargs": kwargs}
        finally:
            self._leave()

    def bind_tools(self, tools, **kwargs):
        self.calls.append(("bind_tools", tuple(tools)))
        return FakeBoundBackend(self, tuple(tools), dict(kwargs))


class FakeBoundBackend:
    def __init__(self, root: FakeBackend, tools: tuple, options: dict) -> None:
        self.root = root
        self.tools = tools
        self.options = options

    def invoke(self, model_input, **kwargs):
        return self.root.invoke(model_input, bound_tools=self.tools, **kwargs)

    def stream(self, model_input, **kwargs):
        return self.root.stream(model_input, bound_tools=self.tools, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return FakeBoundBackend(self.root, tuple(tools), dict(kwargs))


class BackendFactory:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.requests = []
        self.backends: dict[str, FakeBackend] = {}

    def __call__(self, request):
        self.requests.append(request)
        backend = FakeBackend(request.model_instance_id, delay=self.delay)
        self.backends[request.model_instance_id] = backend
        return backend


class ExperimentAssemblerTests(unittest.TestCase):
    def test_every_topology_has_exact_session_count_and_role_mapping(self) -> None:
        expected = {"T5": 5, "T4": 4, "T3": 3, "T2": 2, "T1": 1}
        for profile_id, count in expected.items():
            with self.subTest(profile_id=profile_id):
                factory = BackendFactory()
                system = SystemAssembler(
                    profile_id,
                    executor="synthetic_event",
                    backend_factory=factory,
                ).assemble()
                manifest = system.topology_manifest.to_dict()
                self.assertEqual(len(factory.requests), count)
                self.assertEqual(len(system.sessions), count)
                self.assertEqual(manifest["declared_unique_agent_count"], count)
                self.assertEqual(manifest["assembled_unique_agent_count"], count)
                self.assertEqual(
                    manifest["logical_to_instance"],
                    dict(system.profile.logical_to_instance),
                )
                self.assertTrue(all(manifest["assertions"].values()))
                self.assertEqual(len(manifest["manifest_sha256"]), 64)
                manifested_roles = {
                    role
                    for instance in manifest["instances"]
                    for role in instance["prompt_refs"]
                }
                self.assertEqual(
                    manifested_roles,
                    {
                        role
                        for role, instance in system.profile.logical_to_instance.items()
                        if instance is not None
                    },
                )

    def test_t5_contexts_are_isolated_and_t1_context_is_shared(self) -> None:
        t5 = SystemAssembler(
            "T5", executor="synthetic_event", backend_factory=BackendFactory()
        ).assemble()
        for role, routed in t5.routed_models.items():
            self.assertIsNotNone(routed)
            assert routed is not None
            routed.invoke({"role": role})
        for role, routed in t5.routed_models.items():
            assert routed is not None
            self.assertEqual({item.logical_role for item in routed.context_ledger}, {role})
        self.assertEqual(
            len({id(routed.backend) for routed in t5.routed_models.values() if routed}), 5
        )

        t1 = SystemAssembler(
            "T1", executor="synthetic_event", backend_factory=BackendFactory()
        ).assemble()
        for role, routed in t1.routed_models.items():
            assert routed is not None
            routed.invoke({"role": role}, _experiment_phase=f"PHASE_{role.upper()}")
        ledgers = [routed.context_ledger for routed in t1.routed_models.values() if routed]
        self.assertTrue(all(ledger == ledgers[0] for ledger in ledgers))
        self.assertEqual([entry.logical_role for entry in ledgers[0]], list(t1.routed_models))
        self.assertEqual(
            [entry.phase for entry in ledgers[0]],
            [f"PHASE_{role.upper()}" for role in t1.routed_models],
        )
        self.assertEqual(
            len({id(routed.backend) for routed in t1.routed_models.values() if routed}), 1
        )

    def test_merged_session_serializes_concurrent_logical_role_calls(self) -> None:
        factory = BackendFactory(delay=0.01)
        system = SystemAssembler(
            "T1", executor="synthetic_event", backend_factory=factory
        ).assemble()
        barrier = threading.Barrier(9)
        errors: list[BaseException] = []

        def invoke(index: int) -> None:
            try:
                barrier.wait()
                role = "hri" if index % 2 == 0 else "planner"
                routed = system.routed_models[role]
                assert routed is not None
                routed.invoke({"index": index})
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=invoke, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        backend = next(iter(factory.backends.values()))
        self.assertEqual(backend.max_active_calls, 1)
        self.assertEqual(len(next(iter(system.sessions.values())).snapshot()), 8)

    def test_bind_invoke_and_stream_preserve_session_identity(self) -> None:
        system = SystemAssembler(
            "T4", executor="synthetic_event", backend_factory=BackendFactory()
        ).assemble()
        hri = system.routed_models["hri"]
        memory = system.routed_models["memory"]
        assert hri is not None and memory is not None
        bound = hri.bind_tools(["preview", "confirm"])
        bound.invoke("request", _experiment_phase="PREVIEW")
        self.assertEqual(list(memory.stream("retrieve")), [
            {"chunk": 1, "input": "retrieve", "kwargs": {}},
            {"chunk": 2, "input": "retrieve", "kwargs": {}},
        ])

        self.assertEqual(hri.session_id, memory.session_id)
        records = hri.context_ledger
        self.assertEqual([record.surface for record in records], [
            "bind_tools", "invoke", "stream"
        ])
        self.assertEqual([record.logical_role for record in records], [
            "hri", "hri", "memory"
        ])
        self.assertEqual(records[1].phase, "PREVIEW")

    def test_backend_factory_cannot_reuse_one_object_across_t5_instances(self) -> None:
        shared = FakeBackend("shared")
        with self.assertRaisesRegex(AssemblyError, "reused one backend"):
            SystemAssembler(
                "T5",
                executor="synthetic_event",
                backend_factory=lambda request: shared,
            ).assemble()

    def test_model_factory_cannot_alias_components_across_t5_instances(self) -> None:
        shared = object()
        with self.assertRaisesRegex(AssemblyError, "reused one component"):
            SystemAssembler(
                "T5",
                executor="synthetic_event",
                backend_factory=BackendFactory(),
                model_factory=lambda request: shared,
            ).assemble()

    def test_unsafe_executor_fails_before_any_factory_side_effect(self) -> None:
        factory = BackendFactory()
        with self.assertRaises(UnsafeExecutorError):
            SystemAssembler(
                "A-HostFence", executor="mujoco", backend_factory=factory
            ).assemble()
        self.assertEqual(factory.requests, [])

        with self.assertRaises(UnsafeExecutorError):
            SystemAssembler(
                "T5", executor="human_executor", backend_factory=factory
            ).assemble()
        self.assertEqual(factory.requests, [])

    def test_model_and_embedding_factories_receive_injection_dependencies(self) -> None:
        embedding = object()
        role_requests = []
        embedding_requests = []

        def model_factory(request):
            role_requests.append(request)
            return DefaultRoleInterface(
                logical_role=request.logical_role,
                model_instance_id=request.model_instance_id,
                model=request.model,
                prompt_ref=request.prompt_ref,
                schema_ref=request.schema_ref,
                embedding_model=request.embedding_model,
            )

        def embedding_factory(request):
            embedding_requests.append(request)
            return embedding

        system = SystemAssembler(
            "T5",
            executor="synthetic_event",
            backend_factory=BackendFactory(),
            model_factory=model_factory,
            embedding_factory=embedding_factory,
        ).assemble()

        self.assertEqual(len(embedding_requests), 1)
        memory_request = next(item for item in role_requests if item.logical_role == "memory")
        hri_request = next(item for item in role_requests if item.logical_role == "hri")
        self.assertIs(memory_request.embedding_model, embedding)
        self.assertIn("planner", hri_request.existing_components)
        self.assertIn("memory", hri_request.existing_components)
        self.assertIs(system.embedding_model, embedding)

    def test_oracle_ceiling_is_an_explicit_zero_backend_system(self) -> None:
        system = SystemAssembler(
            "B3", executor="synthetic_event", backend_factory=None
        ).assemble()
        self.assertEqual(dict(system.sessions), {})
        self.assertEqual(system.topology_manifest.assembled_unique_agent_count, 0)
        self.assertTrue(
            all(isinstance(component, BypassedRole) for component in system.components.values())
        )

    def test_manifest_can_be_written_and_hash_is_self_consistent(self) -> None:
        system = SystemAssembler(
            "T2", executor="synthetic_event", backend_factory=BackendFactory()
        ).assemble()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology_manifest.json"
            system.topology_manifest.write(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved, system.topology_manifest.to_dict())


if __name__ == "__main__":
    unittest.main()
