# Diagnosis

## Run status

INVALID_HARNESS

## Oracle verdict

FAIL

## Behavioral summary

Production T5 MuJoCo runtime completed with the objective goal predicate unmet. ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Expected vs observed

At least one frozen trigger/oracle condition failed or was unavailable ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Safety and liveness consequences

See the safety oracle and ordered event record ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## First divergence point

See the failing oracle evidence and terminal record ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Causal timeline

[Event E000001](./experiment_record.md#event-e000001) begins the append-only attempt timeline; the cited disposition is [Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642).

## Likely failure layer

studied_system ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Primary diagnosis

Production T5 MuJoCo runtime completed with the objective goal predicate unmet. ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Confidence

medium ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Competing explanations

- VideoError: source ticks exceed the configured video frame rate ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Disconfirming evidence

Raw calls, frames, states, and video are linked from the record ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Studied system or infrastructure

studied_system ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Recommended follow up

Review linked evidence without changing frozen results ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Reviewer notes

Diagnosis was generated only after behavior and oracle finalization ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Unresolved ambiguity

Manual evidence review may refine cause ([Event E000016](./experiment_record.md#event-e000016), [video 2.642s](./video.mp4#t=2.642)).

## Artifact invalidation

- video: event E000001 video timestamp disagrees with origin mapping
- video: event E000002 video timestamp disagrees with origin mapping
- video: event E000003 video timestamp disagrees with origin mapping
- video: event E000004 video timestamp disagrees with origin mapping
- video: event E000005 video timestamp disagrees with origin mapping
- video: event E000006 video timestamp disagrees with origin mapping
- video: event E000007 video timestamp disagrees with origin mapping
- video: event E000008 video timestamp disagrees with origin mapping
- video: event E000009 video timestamp disagrees with origin mapping
- video: event E000010 video timestamp disagrees with origin mapping
- video: event E000011 video timestamp disagrees with origin mapping
- video: event E000012 video timestamp disagrees with origin mapping
- video: event E000013 video timestamp disagrees with origin mapping
- video: event E000014 video timestamp disagrees with origin mapping
- video: event E000015 video timestamp disagrees with origin mapping
- video: event E000016 video timestamp disagrees with origin mapping
