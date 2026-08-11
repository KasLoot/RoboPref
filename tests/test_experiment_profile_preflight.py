import unittest

from experiments.harness.profile_preflight import run_profile_preflights
from experiments.harness.profiles import PROFILE_IDS


class ProfilePreflightTests(unittest.TestCase):
    def test_all_profiles_instantiate_and_activate_declared_modes(self) -> None:
        results = run_profile_preflights()
        self.assertEqual({item.profile_id for item in results}, PROFILE_IDS)
        self.assertEqual(len(results), 18)
        for result in results:
            self.assertTrue(result.passed, (result.profile_id, result.errors))
            self.assertTrue(all(result.assertions.values()))


if __name__ == "__main__":
    unittest.main()
