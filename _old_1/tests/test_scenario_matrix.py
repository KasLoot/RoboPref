import json
import unittest
from pathlib import Path

from assurance.task_assurance import TaskAssurance


ROOT = Path(__file__).resolve().parents[1]


class ScenarioMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (Path(__file__).parent / "scenario_cases.json").open(encoding="utf-8") as handle:
            cls.matrix = json.load(handle)

    def test_all_task_scenarios_follow_assurance_policy(self):
        assurance = TaskAssurance()
        for case in self.matrix["task_scenarios"]:
            with self.subTest(case=case["id"]):
                self.assertTrue((ROOT / case["dataset"]).is_dir())
                plan_result = assurance.assess_plan(case["planner_output"])
                if plan_result.proceed:
                    result = assurance.assess_validation(case["validator_output"])
                else:
                    result = plan_result
                self.assertEqual(result.outcome, case["expected_outcome"])
                self.assertEqual(result.next_action, case["expected_next_action"])
                if result.failure:
                    self.assertEqual(result.failure["memory_effect"], "NONE")

    def test_scenario_ids_are_unique_and_user_queries_are_present(self):
        task_cases = self.matrix["task_scenarios"]
        identifiers = [case["id"] for case in task_cases]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(all(case["user_query"].strip() for case in task_cases))


if __name__ == "__main__":
    unittest.main()
