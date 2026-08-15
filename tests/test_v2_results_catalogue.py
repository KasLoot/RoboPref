from __future__ import annotations

import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit

from experiments_suite_v2.build_results_index import (
    build_catalogue,
    build_legacy_map,
)
from experiments_suite_v2.result_paths import (
    LEGACY_PREFIX_RELOCATIONS,
    RESULTS_ROOT,
    SUITE_ROOT,
    resolve_result_path,
)
from experiments_suite_v2.runners.bundles import verify_row_bundle


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_KNOWN_IMMUTABLE_HISTORICAL_LINKS = {
    (
        "shared-campaigns/PrefMem-v2-DEV-AB-MEM-Q-20260813/README.md",
        "AB-MEM-Q/AB-MEM-Q__GRID-BATCH__ctx-BATCH__var-BATCH__seed-2718281__A0/results.md",
    ),
}


class ResultsCatalogueTests(unittest.TestCase):
    def test_catalogue_exactly_matches_registered_reports(self) -> None:
        stored = json.loads((RESULTS_ROOT / "index.json").read_text())
        self.assertEqual(stored, build_catalogue())
        self.assertEqual(stored["registered_experiment_count"], 98)
        self.assertEqual(
            stored["classification_counts"],
            {"PASS": 28, "CAPABILITY_FAIL": 70},
        )
        reports = sorted(RESULTS_ROOT.rglob("RESULTS.md"))
        self.assertEqual(len(reports), 98)
        self.assertEqual(
            {row["aim_id"] for row in stored["experiments"]},
            {path.parent.name for path in reports},
        )

    def test_legacy_map_and_resolver_preserve_results_only_layout(self) -> None:
        stored = json.loads((RESULTS_ROOT / "legacy_path_map.json").read_text())
        self.assertEqual(stored, build_legacy_map())
        for retired in ("preflight", "development", "runs"):
            self.assertFalse((SUITE_ROOT / retired).exists())
        for old, new in LEGACY_PREFIX_RELOCATIONS:
            resolved = resolve_result_path(old)
            self.assertEqual(resolved, SUITE_ROOT / new)
            self.assertEqual(
                resolve_result_path(f"experiments_suite_v2/{old}"),
                SUITE_ROOT / new,
            )
            self.assertEqual(
                resolve_result_path(SUITE_ROOT / old),
                SUITE_ROOT / new,
            )
            if old != "runs":
                self.assertTrue(resolved.exists(), old)

    def test_all_local_markdown_links_resolve(self) -> None:
        missing: list[str] = []
        for markdown in sorted(RESULTS_ROOT.rglob("*.md")):
            for raw_target in _MARKDOWN_LINK.findall(markdown.read_text()):
                target = raw_target.strip().strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                path = (markdown.parent / unquote(parsed.path)).resolve()
                if not path.exists():
                    relative_markdown = str(markdown.relative_to(RESULTS_ROOT))
                    if (relative_markdown, target) not in _KNOWN_IMMUTABLE_HISTORICAL_LINKS:
                        missing.append(f"{relative_markdown} -> {target}")
        self.assertEqual(missing, [])

    def test_archived_cal_x03_attempts_are_inspectable_after_relocation(self) -> None:
        attempts = (
            "development/cal_x03_threshold_diagnostic_v1/A1",
            "development/cal_x03_context_diagnostic_v2/A1",
            "development/cal_x03_multiscale_diagnostic_v1/A1",
            "development/cal_x03_multiscale_diagnostic_v1/A2",
            "development/cal_x03_multipart_encoding_diagnostic_v1/A0",
            "development/cal_x03_five_tile_winner_v1/A0",
            "development/cal_x03_five_tile_winner_v1/A1",
        )
        for legacy in attempts:
            relocated = resolve_result_path(legacy)
            self.assertTrue(relocated.is_dir(), legacy)
            self.assertEqual(verify_row_bundle(relocated), [], legacy)


if __name__ == "__main__":
    unittest.main()
