# Campaign changelog

- 2026-08-11T10:51:52.608668Z: protocol version frozen.
- 2026-08-11T10:56:04.828257Z: S000001 attempt 1 was sealed and preserved as
  `INVALID_HARNESS`. The independent artifact audit rejected an 89.2-second
  undeclared frozen source-pane run; metadata retained 1,263 `source_gap`
  frames (1,240 after the first camera acquisition) and only 96 fresh camera
  acquisitions across the 137.1-second video.
  The engine also raised `VideoError: new source capture does not map to an
  increasing CFR slot`. No oracle verdict or behavioral outcome was produced.
- 2026-08-11T11:04:54.608506Z: this campaign was terminated without retry. The
  diagnosed recorder changes (an independent continuous camera pump and
  jitter-bounded first-unused CFR-slot assignment) require a new frozen source
  version and fresh batch preflight; they must not be applied retrospectively
  to this attempt.
