# Campaign changelog

- 2026-08-11T06:41:52.746600Z: protocol version frozen.
- 2026-08-11T06:46:43.044425Z: campaign version terminated after preserving S000001 attempt 1 as `INVALID_HARNESS`. Production initialization delayed the first evidence tick, exposing an inconsistent event-to-encoded-video offset, and normal MuJoCo render jitter then arrived approximately 4 ms before the next CFR slot and was rejected before any model call. The clock/start and bounded-resampling contracts will be corrected and the production condition restarted as v5; the observed oracle failure is not treated as system data.
