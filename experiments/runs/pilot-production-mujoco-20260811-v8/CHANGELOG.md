# Campaign changelog

- 2026-08-11T11:34:16.900581Z: protocol version frozen.
- 2026-08-11T11:41:33.893134Z: the sole scheduled pilot attempt was preserved
  as `INVALID_HARNESS`. The runtime reached controller `COMPLETE`, the objective
  physical predicate and both scenario oracles evaluated `PASS`, and 19
  application-level model calls were retained, but these diagnostics are not a
  valid outcome: independent video audit found a 56.6-second undeclared stale
  source interval above the frozen 30.0-second limit. The artifact check failed
  only on that video-integrity rule (`artifact_checks.json` SHA-256
  `01a931c71ae9fca8057ded903f0562c64bd57eeac5d73163fc9d8cfc9591b7c5`;
  attempt checksum-file SHA-256
  `224d9897a9c852f713a516ce7b5e457f2cad94186a56186d532b881cdc700790`).
  No behavioral result is admitted and no retry or threshold relaxation is
  authorized in v8.
