# Campaign changelog

- 2026-08-11T11:23:01.149380Z: protocol version frozen.
- 2026-08-11T11:27:13.434521Z: the sole scheduled pilot attempt was preserved
  as `INVALID_HARNESS` after five completed application-level model calls and
  before oracle evaluation. The attempt-owned evidence pump exposed an
  over-rate raw camera acquisition in a later CFR slot; the frozen video writer
  failed closed with `VideoError: source ticks exceed the configured video
  frame rate`. Independent partial-attempt artifact and video audit passed
  without errors (`artifact_checks.json` SHA-256
  `f5f2b6afd34636506b163cbcc1fb7dbc7d7aba7b19bc6b9004e40956463976d0`;
  attempt checksum-file SHA-256
  `d41ea52992a597eb6961103b471ae705726399677936a3bb41927d3ed25d5354`).
  No behavioral outcome was produced and no retry is authorized in v7.
