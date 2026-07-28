from __future__ import annotations

import unittest

from prefmem.cli import parse_args


class CliPrintRawTests(unittest.TestCase):
    def test_print_raw_is_disabled_by_default(self) -> None:
        self.assertFalse(parse_args([]).print_raw)

    def test_print_raw_can_be_enabled(self) -> None:
        self.assertTrue(parse_args(["--print-raw"]).print_raw)


if __name__ == "__main__":
    unittest.main()
