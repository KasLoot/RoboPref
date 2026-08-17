from __future__ import annotations

import unittest

import mujoco
import numpy as np

from prefmem.execution.contracts import AnchorKind, ObjectReference
from prefmem.execution.fallback import OracleGroundingError
from simulation.oracle_grounding import MuJoCoOracleGroundingProvider
from simulation.stacking import StackingEnvironment
from tests.test_execution_agent import frame


class MuJoCoOracleGroundingProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = StackingEnvironment(
            start=False,
            viewer=False,
            width=128,
            height=128,
        )
        self.addCleanup(self.environment.close)
        self.provider = MuJoCoOracleGroundingProvider(self.environment)

    def test_exact_cube_alias_resolves_box_top_face(self) -> None:
        grounded = self.provider.ground_from_simulator_truth(
            frame(),
            ObjectReference("red cube", AnchorKind.TOP_CENTER),
            role="source",
            strict_failure_reason_code="SAM_ZERO_DETECTIONS",
        )
        geom_id = mujoco.mj_name2id(
            self.environment.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "red_block_geom",
        )
        centre = self.environment.data.geom_xpos[geom_id]
        rotation = self.environment.data.geom_xmat[geom_id].reshape(3, 3)
        expected = centre + rotation[:, 2] * self.environment.model.geom_size[
            geom_id, 2
        ]

        np.testing.assert_allclose(grounded.point_world, expected)
        self.assertEqual(
            grounded.diagnostics["strict_failure_reason_code"],
            "SAM_ZERO_DETECTIONS",
        )

    def test_unsupported_query_does_not_gain_hidden_supervision(self) -> None:
        with self.assertRaisesRegex(OracleGroundingError, "no exact selector"):
            self.provider.ground_from_simulator_truth(
                frame(),
                ObjectReference("the nearest thing", AnchorKind.TOP_CENTER),
                role="source",
                strict_failure_reason_code="SAM_ZERO_DETECTIONS",
            )


if __name__ == "__main__":
    unittest.main()
