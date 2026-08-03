from __future__ import annotations

import threading
import unittest

from prefmem.emergency import EmergencyStopCoordinator


class EmergencyStopCoordinatorTests(unittest.TestCase):
    def test_concurrent_requests_invoke_placeholder_exactly_once(self) -> None:
        calls: list[str] = []
        notifications = []
        coordinator = EmergencyStopCoordinator(
            calls.append,
            on_stop=notifications.append,
            clock=lambda: 12.5,
        )
        barrier = threading.Barrier(16)
        results: list[bool] = []
        lock = threading.Lock()

        def trigger(index: int) -> None:
            barrier.wait()
            result = coordinator.trigger(
                f"Visible danger {index}.",
                publication_id=f"publication-{index}",
            )
            with lock:
                results.append(result)

        threads = [
            threading.Thread(target=trigger, args=(index,))
            for index in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results.count(True), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(notifications), 1)
        self.assertTrue(coordinator.shutdown_event.is_set())
        self.assertTrue(coordinator.latched)
        self.assertEqual(coordinator.event.observed_at, 12.5)

    def test_stop_hook_error_does_not_unlatch_or_exit_caller(self) -> None:
        def raises_system_exit(reason: str) -> None:
            raise SystemExit(99)

        coordinator = EmergencyStopCoordinator(raises_system_exit)

        with self.assertLogs("prefmem.emergency", level="ERROR"):
            self.assertTrue(
                coordinator.trigger("A person is in the robot path.")
            )
        self.assertTrue(coordinator.latched)
        self.assertIsInstance(coordinator.stop_error, SystemExit)
        self.assertFalse(
            coordinator.trigger("A second result must not call the hook again.")
        )


if __name__ == "__main__":
    unittest.main()
