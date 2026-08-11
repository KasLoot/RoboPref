#!/usr/bin/env python3
"""Run and optionally persist the deterministic invariant campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiments.harness.invariants import require_invariants, run_invariant_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-per-family", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_invariant_suite(
        traces_per_family=args.traces_per_family,
        master_seed=args.master_seed,
    )
    payload = {
        "schema_version": 1,
        "master_seed": args.master_seed,
        "traces_per_family": args.traces_per_family,
        "total_traces": sum(item.trace_count for item in results),
        "passed_traces": sum(item.passed for item in results),
        "families": [asdict(item) | {"ok": item.ok} for item in results],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    require_invariants(results)


if __name__ == "__main__":
    main()
