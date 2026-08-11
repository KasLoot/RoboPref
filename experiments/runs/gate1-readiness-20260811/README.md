# RoboPref Mandatory Gate 1 readiness assessment

Generated 2026-08-11; finalized 2026-08-11T11:50:10.145449Z. The
machine-readable companion is
[`gate1_readiness.json`](gate1_readiness.json).

## Verdict

**MANDATORY GATE 1: NOT PASSED.**

Locked execution is not authorized. No locked campaign, locked protocol hash,
locked schedule, start/resume command, or approval request exists. The pilot
hashes below identify excluded diagnostics only and cannot authorize a locked
run.

The decisive evidence is:

- The final full repository suite passed 447/447 tests in 240.873 seconds with
  host loopback and `MUJOCO_GL=egl`; the experiment-only suite passed 190/190
  in 136.572 seconds. Test success does not override failed campaign evidence.
- Five publication-oriented pilot versions, v4 through v8, are preserved as
  `INVALID_HARNESS`. V8 reached controller `COMPLETE`, the physical predicate,
  and diagnostic oracle `PASS`, but its independent audit rejected a
  56.6-second stale-source interval. That candidate result is excluded.
- The publication engine supports only T5/NM02/primary/MuJoCo on the pilot
  split and refuses locked assignments. Locked-executable coverage is zero.
- No paired T5/B1 production pilot, powered publication curve, or selected
  repetition count `r` exists. Alternative/small model backbones and most
  profiles/scenarios are not executable.
- Exact raw HTTP request/response capture for upper and embedding clients is
  absent; current records are redacted application-level semantic evidence.
- At `r=10`, the low five-minute/6-Mbit/s video sensitivity is 1.386 TB. With
  current free space, a 20% reserve leaves 1.320163 TB and a 25% reserve leaves
  1.237652 TB before non-video artifacts or retries.

## Gate checklist

| Required Gate 1 item | Disposition |
|---|---|
| Implementation and changed-file summary | Present below and in the [canonical crosswalk](../../REQUIREMENTS_CROSSWALK.md) |
| Tests, preflights, and pilot results | Present; test suites pass, production pilots remain invalid |
| Unresolved limitations | Present and blocking |
| Locked run count and resource estimate | Formula/sensitivities present; exact count remains unset because `r` is unset |
| Frozen locked protocol path/hash | Absent by design; Gate failed |
| Frozen locked source/config/prompt hashes | Absent |
| Randomized locked schedule | Absent |
| Exact locked start/resume command | Withheld because no valid locked campaign exists |
| Artifact examples | Linked below |

## Implementation summary

The repository now has a substantial fail-closed experiment control plane:

- 18 strict profiles: five topologies, four baselines, and nine mechanism
  ablations; all 18 construction/activation preflights pass.
- 62 scenarios across six families with distinct development, pilot, and
  locked variants; frozen topology/sensitivity/real-system subsets contain
  24/20/12 templates.
- Five strict executable oracle types, runner-only one-shot oracle capability,
  exact trigger boundaries, response bounds, and hidden-state journaling.
- Deterministic blocked schedules, CRN assignments, append-only registries,
  fresh attempt directories, source/protocol/config hashes, locked-approval
  refusal, three-attempt infrastructure policy, and one-use recovery evidence.
- Append-only events, exact visual request/source hashes, application-level
  model-call records, synchronized H.264 composite/robot videos, state/oracle
  ledgers, diagnoses, checksums, immutable attempt audits, and analysis
  regeneration.
- ITT accounting, missingness bounds, paired summaries, Holm adjustment, and a
  narrow Bernoulli-logit nested mixed-model/power implementation.

This control plane is not a full publication data plane. The current
[`production_engine.py`](../../harness/production_engine.py) deliberately
fails closed outside T5/NM02/primary/MuJoCo/pilot. It does not execute B1,
T4–T1, the mechanism ablations, the other 61 templates, locked fixtures, or
alternative/small backbones. One request containing multiple distinct views
also fails the current singular capture contract.

## Changed files and provenance

Repository HEAD is `4bdd81b9f51942e15635544b0c8bbfbfb5ec8537`
(`2.2.1`, dirty). Final `git status --porcelain` contains 787 rows, including
745 tracked deletions that predated or are unrelated to this experiment work.
Those user changes were preserved; no cleanup/reset was performed.

Experiment work is principally under `experiments/` and the
`tests/test_experiment_*.py` files, with current-runtime seams and physical
fixes in `src/prefmem` and `src/simulation`. Every campaign manifest binds its
own commit, dirty-patch/status digests, frozen inputs, and source manifest, so
historical pilots remain interpretable despite the live dirty tree. A clean,
reviewable release commit remains a locked-freeze prerequisite.

Selected final-source SHA-256 values:

| File | SHA-256 |
|---|---|
| [`production_engine.py`](../../harness/production_engine.py) | `b7ca8b0d5ccb70208e5ecc151b94291bf4a48812e1457d9a3f18b8be6f7fb037` |
| [`runner.py`](../../harness/runner.py) | `7bf8fbf1c1f8f6139be116b66c1360ae1e86301f234087b6d90a26087c78b396` |
| [`video.py`](../../harness/video.py) | `80aa17ee1b6b21a2a36680408389a9bcd790125d19f2837418285082ef3153ab` |
| [`recording.py`](../../harness/recording.py) | `60ec8f67b5993dc36eff74979dc02028c4604a20e0fb512d919e36d3276263d1` |
| [`controller.py`](../../../src/simulation/controller.py) | `ed158f480b3a64b87a63386788bd633ae76131fb7fe6cfda3725cc14341a787e` |
| [`stacking.py`](../../../src/simulation/stacking.py) | `f2ea73ccbd1123bf4b1907a3c85a9ef3d53effaa83836847bcfd2605b9c87638` |

## Calibration and tests

| Check | Result | Evidence |
|---|---:|---|
| Deterministic invariants | 7,680/7,680 PASS across 30 families | [JSON](../../calibration/invariants_20260811.json), SHA-256 `c026ec4c9e6f33a205a3dabace50882f9e6d4adbcc645073733b6aa941bae5f0` |
| Profile construction/activation | 18/18 PASS | [JSON](../../calibration/profile_preflights.json), SHA-256 `0f00bc3084f21d6241f6eef6ca1c8d7113383310dcef6b7e32db5605ea22c3ef` |
| Corrected mechanism-scene preflight | 29/29 PASS | [JSON](../../calibration/mechanism_scene_preflights_attempt2.json), SHA-256 `1b0f69ebc63382288166bb74bb3e190d674a43f91567f362cdf3b8e25e439392` |
| Final service preflight attempt 10 | Three services × two health checks + one smoke; PASS | [JSON](../../calibration/service_preflight_20260811_attempt10.json), SHA-256 `68d6015c72797701adb49eddba72e984a8f3150650c27d64b6f4454c7f798f26` |
| Experiment contract suite | 190/190 PASS in 136.572 s | Operator-observed console result; `MUJOCO_GL=egl` |
| Physical stacking suite | 10/10 PASS in 162.453 s | Operator-observed console result; `MUJOCO_GL=egl` |
| Full repository suite | 447/447 PASS in 240.873 s | Operator-observed console result; host loopback + `MUJOCO_GL=egl` |
| Campaign/documentation audit tests | 23/23 PASS in 2.948 s | Operator-observed console result |

These tests close the earlier sandbox/X11 and controller-regression uncertainty.
They do not close the attempt-long camera-source failure that v8 exposed.

## Excluded instrumentation pilots

Six publication-invalid instrumentation campaigns have passing independent
audits and validate plumbing only: [scripted pass](../pilot-scripted-pass-20260811-v3/artifact_audit/audit.json),
[scripted controlled failure](../pilot-scripted-failure-20260811-v3/artifact_audit/audit.json),
[direct MuJoCo pass](../pilot-mujoco-pass-20260811-v3/artifact_audit/audit.json),
[direct MuJoCo controlled failure](../pilot-mujoco-failure-20260811-v3/artifact_audit/audit.json),
[live service](../pilot-service-live-20260811-v3/artifact_audit/audit.json), and
[disconnect/recovery](../pilot-service-disconnect-20260811-v4/artifact_audit/audit.json).
Their common excluded protocol SHA-256 is
`124e57a852f062388306fe254b6957cd121aa9d1fbe4018657f9a628fc2cb927`.
It is not a locked protocol.

## Production pilot history

All five rows are excluded `INVALID_HARNESS` diagnostics.

| Version | Decisive result | Canonical manifest SHA-256 | Key artifact |
|---|---|---|---|
| v4 | Clock/source-origin and CFR mapping failed before model calls | `60b1c508a4cd3bf89c6fbdcad57215c064f949f7e7ef915503e9d3e7cf48616e` | [artifact check](../pilot-production-mujoco-20260811-v4/episodes/S000001/attempt_1/artifact_checks.json) |
| v5 | Nested Planner capture alias invalidated post-tool HRI request after two calls | `ceda547f504856fb5f7b27e1806c479919900b4971d095fbec031cd68fe283b0` | [error](../pilot-production-mujoco-20260811-v5/episodes/S000001/attempt_1/error.json) |
| v6 | 17 calls retained; 89.2-second stale-source run and same-slot CFR collision | `27014b40c32265585959db2c04628be8076ac9f07c02ef9f872811c638dd1c5e` | [artifact check](../pilot-production-mujoco-20260811-v6/episodes/S000001/attempt_1/artifact_checks.json) |
| v7 | Five calls retained; later-slot raw camera tick violated independent cadence gate; partial audit otherwise passed | `353107e531e17f1ee61c582ff648c3d8cd104f3d3ff0c4c8eb629f1009b0fba1` | [error](../pilot-production-mujoco-20260811-v7/episodes/S000001/attempt_1/error.json) |
| v8 | Candidate controller/oracle PASS excluded because video audit found a 56.6-second stale-source interval | `b7ec5804a3f44483cc6de5433225c0620144965dc469fd36ea45496fd5ad1b0e` | [artifact check](../pilot-production-mujoco-20260811-v8/episodes/S000001/attempt_1/artifact_checks.json) |

V8 retained 19 application-level calls, 18 visual request links, 147 events,
and 255.9 seconds/2,559 frames in each video. Its candidate physical and safety
oracles passed, but `RUN_STATE.json` and `results.json` correctly separate that
diagnostic verdict from final `INVALID_HARNESS`. The largest fresh-acquisition
gaps were 56.600394 and 40.185834 seconds. No v9, retry, or threshold relaxation
was authorized.

## Locked schedule and resource sensitivity

The full fresh-reference design is:

```text
main comparison        248r
topology comparison    120r
mechanism ablations    128r
model sensitivity      1,200
total                   496r + 1,200 episodes
```

| `r` | Episodes | Max preserved attempts at 3/episode | Sequential hours at 3 / 5 / 10 min | Video TB at 5 min and 6 / 12 / 20 Mbit/s |
|---:|---:|---:|---:|---:|
| 10 | 6,160 | 18,480 | 308 / 513.333 / 1,026.667 | 1.386 / 2.772 / 4.620 |
| 15 | 8,640 | 25,920 | 432 / 720 / 1,440 | 1.944 / 3.888 / 6.480 |
| 20 | 11,120 | 33,360 | 556 / 926.667 / 1,853.333 | 2.502 / 5.004 / 8.340 |
| 25 | 13,600 | 40,800 | 680 / 1,133.333 / 2,266.667 | 3.060 / 6.120 / 10.200 |

At 2026-08-11T11:50:10Z, the filesystem had 1,650,203,160,576 bytes
available. Retaining 20%/25% leaves 1,320,162,528,461 / 1,237,652,370,432
bytes. The lowest `r=10` video-only sensitivity exceeds those budgets by about
65.837 GB / 148.348 GB before frames, calls, logs, analyses, or retries.

An exact inference-call total is intentionally not invented: valid production
distributions for cycles, monitor polls, repairs, malformed responses, and
timeouts do not exist. V8's 19 calls and 42.9-MB campaign are a single invalid
diagnostic, not a campaign-wide estimator.

## Blocking gaps

1. Redesign and validate the MuJoCo producer/recorder lifecycle so acquisition
   continues independently for an entire attempt; v8 disproved the current
   short-liveness tests. Keep the 30-second limit unchanged.
2. Implement publication execution for every frozen profile, scenario,
   topology/ablation, trigger, backbone, and locked variant. Current locked
   coverage is zero.
3. Capture and independently audit exact redacted raw HTTP request/response and
   malformed/error bodies for upper and embedding services.
4. Obtain valid excluded T5/B1 production pilots; freeze exact GLMM factors,
   references, multiplicity alpha, and seed; run at least 1,000 simulations per
   candidate and select `r`.
5. Implement every retained nonbinary analysis and deterministic paper
   table/figure command, or remove the corresponding claims in a versioned
   preregistration.
6. Resolve three immutable backbones/tokenizers/serving parameters and the
   single-call multi-view evidence limitation.
7. Provision adequate storage and obtain representative production
   duration/call/bitrate/compute distributions.
8. Establish a reviewable release source state, licence, and anonymized sample.
   Human/real-robot work remains separately ethics/safety authorization-gated.

## Gate disposition

No locked freeze was created, no locked identifiers or hashes are available,
and no exact approval phrase is requested. The safe next step is engineering
and excluded-pilot work on the blockers above, followed by a new Gate 1
assessment. Locked execution must remain unavailable until that assessment
passes and the user supplies the exact approval required by the frozen
protocol.
