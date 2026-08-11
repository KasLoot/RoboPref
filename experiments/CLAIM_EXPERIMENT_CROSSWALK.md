# Claim-to-experiment crosswalk

This is a concise claim-level supplement. The canonical exhaustive trace of
user and design requirements to implementation, verification, evidence, and
open gaps is
[REQUIREMENTS_CROSSWALK.md](REQUIREMENTS_CROSSWALK.md).

Status values here describe what the repository can currently encode, not a
claim that the experiment has been executed. `Implemented` means a declarative
profile/scenario/oracle or analysis primitive exists. `Gated` means evidence is
unavailable until the named approval or pilot condition is satisfied. The
publication-valid execution engine currently supports only T5/NM02/MuJoCo, so
every full-matrix claim remains blocked even where its declarative profile is
implemented.

Six excluded instrumentation campaigns have clean independent audits, including
pass/failure and disconnection/recovery paths; exact paths and hashes are in the
[canonical crosswalk](REQUIREMENTS_CROSSWALK.md#excluded-pilot-evidence).
Production v4, v5, v6, v7, and v8 are preserved `INVALID_HARNESS` diagnostics. V6
retained 17 calls but failed continuous-source/CFR evidence; v7 retained five
calls with a clean partial audit but exposed an unmirrored raw-cadence gate.
The post-v7 replacement passes 190 experiment tests and 10 physical stacking
tests. V8 reached controller `COMPLETE` and diagnostic oracle `PASS`, but its
artifact audit rejected a 56.6-second stale-source interval, so it supplies no
behavioral outcome. These facts close no claim row: there is no
valid production outcome, no paired T5/B1 pilot, no power-selected `r`, and no
locked freeze or approval.

| Claim | Confirmatory comparison | Frozen scenario coverage and outcome | Implementation/check | Required evidence | Current gap |
|---|---|---|---|---|---|
| RQ1: full architecture improves safe holistic completion | T5 vs B1, B2, B4 | All 62 locked variants; contract success and safe completion | Strict profiles, executable oracles, main-system CRN blocks, ITT and paired summaries | Complete attempt trees, matched paired estimate, mixed-model diagnostics, videos and audit | No valid production or paired T5/B1 pilot. Only invalid T5/NM02/primary diagnostics exist (<0.5% of the minimum schedule); `r`, models, engine coverage, freeze and approval remain open |
| RQ2: consented preference memory reduces interaction without unsafe writes | T5 vs A-Memory; human T5 vs A-Memory | Preference-memory plus ambiguity/override/update/forget markers; correction, clarification, unauthorized mutation | A-Memory removes the role/store path; memory-event oracles and mutation evidence | Simulation episode records; later consented participant records | Technical arm unrun; human claim gated on ethics/consent and powered participant design |
| RQ3: receding-horizon feedback improves perturbation recovery | T5 vs A-OpenLoop and B1 | Perturbation-recovery family and A-OpenLoop subset; trigger firing, recovery, safe completion | Frozen-queue/open-loop profile; event-order and hidden-state oracles | Identical predicate-triggered CRN instances, recovery latency and final state | Instrumentation controlled-failure pilots pass, but no production perturbation/open-loop path or matched outcome exists |
| RQ4: step monitoring prevents stale-success propagation | T5 vs A-Monitor | A-Monitor subset; false step success, downstream contract failure | No-Monitor profile plus monitor/oracle event capture | Step assessment, executor, state, frame and video links | Unrun; instantiated graph/profile preflight required |
| RQ5: independent holistic validation reduces false completion | T5 vs A-Validator | Evidence-validation and A-Validator subset; false completion | No-Validator profile, final hidden-state oracle | Terminal transition, checklist assessment, hidden final state and video | Unrun; final-state recording must pass |
| RQ6: exact confirmation and host fencing prevent unauthorized execution | T5 vs A-Confirmation plus A-HostFence deterministic traces | Confirmation subset and synthetic host-fence subset; unauthorized publication | Confirmation profile, host-fence synthetic-only policy, sequence oracle | Publication/confirmation event order and command IDs | Unrun; unsafe conditions forbidden outside synthetic/MuJoCo |
| RQ7: repeated evidence resists transient observations | T5 vs A-SingleEvidence | Single-evidence subset; transient-frame terminal error and false completion | Zero-stability single-evidence profile, frame/event mapping | Exact request frames, evidence timestamps, video alignment | Instrumentation recording audits pass, but v8 invalidated the post-v7 production contract on attempt-long source continuity. It permits different captures across sequential calls and rejects multiple distinct views in one request; required multi-view validation remains unsupported |
| RQ8: logical-role decomposition changes reliability/cost | T5, T4, T3, T2, T1 | Frozen topology24 strata; contract success, latency, tokens/calls | Declarative logical-to-instance maps and blocked topology schedule | Actual instantiated graph in setup, server model IDs, cost journals | All 18 profile construction preflights pass, but publication runtime rejects T4-T1 and only invalid T5/NM02 attempts exist |
| RQ9: success/safety/latency/token trade-offs | All executed technical systems | All applicable templates; preregistered outcomes plus exploratory cost metrics | Event/model-call/runtime journals and taxonomy | Complete raw per-episode values and explicitly exploratory Pareto analysis | Excluded instrumentation reports exercise plumbing only; no production outcome distribution/Pareto analysis and no opaque aggregate is allowed |
| RQ10: technical mechanisms improve interaction for people | Within-subject T5 vs A-Memory | Human task sets must be counterbalanced and separate from simulation catalogue | Technical profiles only; no participant runner or approved protocol is evidence | Ethics approval, consent/withdrawal, compensation/privacy/retention, powered sample, anonymized data | Gated: no human study is authorized or present |

## Cross-cutting protocol requirements

| Requirement | Implementation/artifact | Verification | Unresolved gate or risk |
|---|---|---|---|
| Development/pilot/locked separation | Three template-level split variants; campaign mode and split validation | Catalogue tests plus six excluded campaign audits | Instrumentation pilots remain excluded from behavior/power; no locked freeze exists |
| Common random numbers | One seed and CRN key per randomized block | Schedule validator rejects within-block mismatch | External executor/model nondeterminism remains a reported threat |
| Blocked randomized, sequential order | Deterministic block/profile shuffle and one open attempt globally | Reproducibility and registry tests | Parallel execution needs a new preregistered isolation proof |
| Freeze and full provenance | PROTOCOL hash, source/frozen-input file hashes, commit/status/diff hashes | Six clean pilot freeze/audits validate mechanics | Post-pilot docs/source need a new snapshot; no locked freeze exists and the dirty tree is a reproducibility risk |
| Exact approval | Exact full-match parser bound to campaign ID and protocol hash | Negative parser and locked-refusal tests | Approval covers locked simulation only, not human/real operation |
| Immutable attempts and retry limit | Append-only histories, frozen-order enforcement, fresh attempt paths, maximum three evidenced infrastructure attempts, and one-use hashed recovery/health/smoke authorization | Registry tests plus audited service-disconnect v4 interruption/recovery/fresh same-seed retry | Production outage/recovery and third-interruption quarantine remain unpiloted |
| Status/oracle/diagnosis separation | Distinct registry fields, results fields, and diagnosis document | Registry/result consistency audit | Post-run diagnosis is not a running-system input |
| Complete request evidence | Exact transport images; optional lossless `source_image_path`/`source_raw_image_sha256`; semantic request-body/redaction hashes including bound tools/options; terminal `frame_response_link`/strict `frame_error_link` | Final-source experiment suite passes 190/190; v6 retained 16, v7 retained five, and v8 retained 18 exact visual requests before their separate video-integrity failures | V8 failed attempt-long continuous-source integrity. One visual call must bind exactly one distinct capture; single-call multi-view is rejected, and no screenshot substitution is permitted. Exact raw HTTP request/response bodies for upper and embedding services are not captured |
| Full synchronized video | Composite and clean robot-camera streams plus metadata | Six clean instrumentation audits; v4 correctly failed clock/CFR audit and v5 partial audit passed before capture invalidation | Current fix lacks production audit. Low r10 video sensitivity (1.386 TB) exceeds measured 20% and 25% retained capacity |
| ITT/missingness and confirmatory binary analysis | Every assigned schedule entry retained; unresolved outcomes return NOT ESTIMABLE and bounds; narrow Bernoulli-logit GLMM primitive is fail-closed | Six excluded pilots regenerate generic analysis; mixed-effects contract tests | No real paired T5/B1 rows, nuisance fit, frozen GLMM config, ≥1,000-simulation power curves, or selected `r` |
| Multiple comparisons | Holm correction within the preregistered reference-profile family | Deterministic analysis tests | Exploratory analyses remain separately labelled |
| Historical evidence | No compatible prior campaign is treated as current locked evidence | Campaign directory/provenance audit | The design's referenced 39-scenario history is not recoverable here as auditable evidence |
| Simulator identity | MuJoCo executor is named in schedule/setup | Executor/profile safety validation | Webots is not implemented and must not be named as the evaluated simulator |
| Real robot | Only safety-screened profiles/scenarios may become eligible | Fail-closed profile/scenario intersection | Separate authorization, hardware pilot, safety case, and capacity evidence required |
| Human participants | Technical comparison is specified but not executable as a study | Ethics/consent documents must be reviewed outside this harness | Recruitment is forbidden until institutional and user gates are satisfied |
| Gate 1 / locked execution | Exact approval parser and manifest-bound runner | Excluded pilots and targeted tests only | **NOT PASSED:** no full production engine, valid production pair, alternate/small models, adequate storage, licence/ethics/real-robot authority, locked campaign/hash, or exact approval |

No row in this crosswalk is closed merely by the existence of code. A claim row
closes only after its frozen scheduled entries resolve, artifacts audit, and
the predeclared analysis regenerates from those immutable artifacts.
