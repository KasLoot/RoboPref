# RoboPref technical experiment preregistration

Status: **draft for pilot calibration; not frozen and not approved for locked
execution**. Date: 2026-08-11. Project version at drafting: 2.2.1.

This document covers synthetic-event and MuJoCo technical experiments only.
The repository does not contain Webots integration; “Webots” is not a synonym
for the evaluated simulator. This preregistration does not authorize human
participants or real-robot execution.

## Questions and confirmatory hypotheses

- H1/RQ1: T5 has higher contract-success and safe-completion probability than
  B1, B2, and B4 on the locked 62-template end-to-end matrix.
- H2/RQ2: T5 reduces correction/clarification burden versus A-Memory without a
  higher unauthorized-mutation rate on the claim-relevant subset.
- H3/RQ3: T5 has higher perturbation-recovery probability than A-OpenLoop.
- H4/RQ4: T5 reduces false step success and downstream contract failure versus
  A-Monitor.
- H5/RQ5: T5 reduces episode-level false completion versus A-Validator.
- H6/RQ6: T5 prevents pre-authorization publication more reliably than
  A-Confirmation; deterministic A-HostFence traces must admit zero wrong/stale
  publication.
- H7/RQ7: T5 reduces transient-observation terminal error versus
  A-SingleEvidence.
- H8/RQ8: topology T5/T4/T3/T2/T1 changes contract success; latency/call cost is
  an explicit trade-off, not folded into success.

RQ9 Pareto and detailed cost relationships are exploratory. RQ10 is not part of
this campaign. A future human study requires institutional ethics approval,
consent/withdrawal, compensation, privacy/retention, recruitment, anonymization,
and a separately powered preregistration before any participant is enrolled.

## Experimental unit, systems, and matrices

The independent unit is one randomized episode. Frames, actions, planning
cycles, monitor polls, validator calls, and repeated evidence are correlated
within-episode measurements and are never treated as independent replicates.

The fresh-reference technical schedule is:

- all 62 templates × `r` × T5/B1/B2/B4;
- frozen topology24 templates × `r` × T5/T4/T3/T2/T1;
- each mechanism marker × `r` × T5/the named ablation (64 marked templates
  total before the two profiles are expanded);
- frozen sensitivity20 × 10 × T5/B2 × three frozen backbones.

This is `496r + 1,200` episodes. Reference episodes are fresh across matrices;
they are not retrospectively reused because one result looked favorable. The
pilot will use approximately five matched repetitions in selected cells. The
locked `r` will be the smallest prespecified candidate attaining at least 90%
simulation-based power for the minimum important primary effect when feasible,
never below 10. Candidate values are 10, 15, 20, and 25; the selected value,
pilot effect assumptions, simulation seed, and full power curve must replace
this paragraph before freeze.

Minimum important differences are 10 percentage points in contract success and
5 percentage points in false completion. Any different target requires a new
dated preregistration version before locked outcomes are exposed.

## Splits, generation, and randomization

Development, pilot, and locked variants differ at template level in semantics,
layout, language, preference history, distractors, and perturbation boundary.
Development data may tune implementation. Pilot data may establish feasibility,
variance, `r`, timeouts, recording settings, and fixed thresholds. Neither may
enter confirmatory estimates.

For each matrix/scenario/backbone/repetition block, all profiles receive the
same generated-instance seed, variant, paraphrase, memory state, perturbation
predicate/timing, and controller seed. Blocks and profile order within blocks
are shuffled by a frozen master seed. Only the resulting `run_schedule.csv`
order may execute; supplying a later schedule ID cannot jump over the next
runnable row. Variants run sequentially by default; at most one registry
attempt is open. Parallelism needs a new frozen resource-isolation validation.

The sensitivity backbone labels must resolve before freeze to exact server model
IDs, immutable model/tokenizer hashes, decoding settings, prompts, schemas, and
service parameters. A service label or localhost port alone is not model
provenance.

## Outcomes and estimands

The primary binary outcomes are physical goal completion, contract success,
safe completion, false completion, perturbation recovery, and preference
satisfaction as applicable to the declared scenario. Objective hidden state is
scoring-only. Model timeouts, malformed output, perception ambiguity, planning
errors, execution faults, monitor/validator errors, and task timeouts are system
outcomes, not infrastructure exclusions.

The primary estimand for each binary comparison is the intention-to-treat
difference in episode success probability between randomized profiles on its
frozen matrix. Descriptive per-profile proportions use 95% Wilson intervals.
Matched CRN comparisons report paired risk differences, percentile bootstrap
intervals, discordant/tied counts, and an exact McNemar/binomial test.

The binary confirmatory implementation in `harness/mixed_effects.py` is a
narrow Bernoulli-logit GLMM with explicit treatment/reference coding and
independent Gaussian random intercepts for scenario template and generated
instance nested within template. It uses deterministic nested Gauss-Hermite
marginal-likelihood integration, validates the optimum at a higher quadrature
order, and fails closed on missing assigned outcomes, separation, rank or
conditioning failures, insufficient groups, non-convergence, singular random
effects, an invalid observed Hessian, or quadrature instability. It reports
the exact formula/coding/references, optimizer and software versions, marginal
contrasts, per-template raw summaries, and machine-readable diagnostics.

Implementation alone does not close the **freeze blocker**. Before locked
freeze, an estimable fit must regenerate from immutable excluded-pilot rows and
the final outcome-specific factors, reference levels, multiplicity-adjusted
alpha, fit tolerances, random-effects structure, diagnostic rules, power seed,
sensitivity assumptions, and full configuration hash must be fixed. The
mixed-model power job must run at least 1,000 simulations for each candidate
`r` in 10, 15, 20, and 25; invalid simulated fits count as nondetections. The
existing episode-level/paired summaries remain transparent secondary checks,
not substitutes for this predeclared confirmatory fit. Count, positive
continuous, censored-time, and human ordinal mixed-model families remain
unimplemented and cannot be claimed.

Positive continuous/count/time outcomes are secondary unless a claim row names
them primary. Timeouts are censored where the frozen model supports censoring;
they are not deleted. Cost outputs include calls by logical role, tokens when
reported by the server, per-call/episode latency, simulator/execution time, and
artifact bytes. Unsupported server token/GPU/energy measures are `not
available`, never zero.

## Missingness, exclusions, and retries

Every randomized schedule entry remains in the ITT denominator. If an assigned
outcome remains unavailable, the confirmatory point estimand is `NOT ESTIMABLE`
and reports missing status plus best/worst success bounds. A complete-case
estimate may be labelled descriptive but may not replace ITT.

Permitted exclusions are a documented recorder/oracle/harness corruption, an
independently evidenced external service outage, preregistered hardware safety
cancellation, or (in a separate future study) participant withdrawal. Each is
preserved with schedule/attempt identity and reason in registry and exclusions
records. Exclusion is not deletion.

A valid pass or valid system failure is final and cannot retry. A safety abort
is preserved and treated according to the outcome-specific rule; it cannot be
rerun to seek a pass. Only `INFRA_INTERRUPTED` may retry, from a fresh scene in
a new attempt directory with the identical scheduled seed. Maximum attempts
per scheduled episode are exactly three. The third interruption quarantines the
entry as unresolved. Attempts are never spliced, overwritten, or cherry-picked.
An open reservation discovered after process death defaults to
`INVALID_HARNESS` with its partial artifacts and reason. It becomes
`INFRA_INTERRUPTED` only when contemporaneous health/transport evidence
independently establishes an external outage; free-text operator assertion is
insufficient. A `VALID_*` registry seal requires a passing artifact audit.
Before an infrastructure retry, a separate recovery report must bind the
interrupted attempt and outage-evidence hash, prove restoration, cite distinct
hashed raw JSON for two consecutive successful checks, and cite a later
non-mutating smoke test. The registry revalidates all files and times under its
lock and consumes the authorization once.

## Multiplicity, robustness, and taxonomy

Holm correction applies within each preregistered family of T5 pairwise primary
comparisons. Family membership and size come from the frozen schedule, not the
set of estimable outcomes. If a family member is unresolved, adjusted decisions
for that family remain not estimable rather than shrinking the multiplicity
penalty. Confirmatory and exploratory p-values are never pooled into one
family. Raw p-values, adjusted p-values, interval estimates, per-template raw
values, and all denominator/missing counts are reported.

Prespecified secondary views stratify by scenario family, difficulty, horizon,
catalogue, and perturbation/evidence condition. Post-hoc views are labelled
post-hoc. Failure taxonomy uses perception/SAM, embedding/retrieval, HRI,
planning, execution, monitoring, validation, coordination/state management,
simulator/robot, model service/infrastructure, recorder/oracle/harness, or
unassigned. Diagnosis occurs after outcome finalization and cannot feed the
episode or change its oracle verdict.

## Safety, stopping, and authorization

Profile and scenario executor permissions are intersected fail-closed before an
adapter is constructed. A-HostFence is synthetic-only. Unsafe confirmation,
monitoring, evidence, and dynamic-scope ablations are simulation-only. Emergency
or workspace policy stops an attempt, preserves it as `ABORTED_SAFETY`, and
forbids further publication.

Campaign execution stops for artifact corruption affecting a matrix, evidence
of split leakage, a frozen hash mismatch, approval mismatch, unsafe profile
instantiation, uncontrolled resource contention, or a substantive code/prompt/
oracle/config correction. A substantive correction terminates that campaign
version and restarts every affected condition under a new version. Outcome
counts are not a stopping rule.

Locked execution is forbidden until successful and controlled-failure pilots
cover every materially distinct execution/recording path; service disconnection
classification is tested; videos/frames/events/calls/states/oracles align; all
profile graphs are verified; artifact and analysis audits pass; resource
estimates and current disk safety margin are presented; and protocol/source/
config/prompt/schedule hashes are frozen.

After that report, execution still remains forbidden until the user supplies
the exact, matching sentence:

```text
APPROVE LOCKED CAMPAIGN <campaign_id> <protocol_sha256>
```

That approval applies only to the named frozen technical campaign. It does not
authorize SSH key disclosure, human recruitment, unsafe real-robot ablations,
or a different campaign/hash.

## Reporting boundaries

The technical campaign may support claims only for named profiles, models,
templates, seeds, and MuJoCo/synthetic conditions. It cannot establish safety
certification, arbitrary-scene compatibility, Webots behavior, general
real-world reliability, human preference benefit, or superiority to an
unimplemented external benchmark. The design's historical 39-scenario
reference is not current locked evidence because no compatible provenance-rich
campaign is present here.

Final reports separate confirmatory, secondary, exploratory, and post-hoc
results and include every status, missing assignment, exclusion, retry, safety
abort, audit failure, protocol/source digest, wall time, inference-call count,
token availability, and artifact/storage cost.
