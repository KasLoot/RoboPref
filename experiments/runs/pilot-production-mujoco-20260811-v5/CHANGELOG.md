# Campaign changelog

- 2026-08-11T07:09:00.204934Z: protocol version frozen.
- 2026-08-11T07:13:03Z: campaign version terminated after preserving S000001 attempt 1 as `INVALID_HARNESS` with a passing partial-attempt artifact audit. The initial HRI request and its nested Planner request completed, but the post-tool HRI call reused the immutable HRI-turn image while the harness incorrectly compared it with the newer nested Planner capture. No oracle evaluation or behavioral result was produced. Request-specific immutable-frame resolution will be corrected and the production condition restarted as v6; this attempt is excluded from studied-system evidence.
