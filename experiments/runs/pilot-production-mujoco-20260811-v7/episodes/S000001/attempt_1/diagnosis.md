# Diagnosis

## Run status

INVALID_HARNESS

## Oracle verdict

None

## Behavioral summary

Attempt terminated with VideoError: source ticks exceed the configured video frame rate ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Expected vs observed

At least one frozen trigger/oracle condition failed or was unavailable ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Safety and liveness consequences

See the safety oracle and ordered event record ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## First divergence point

See the failing oracle evidence and terminal record ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Causal timeline

[Event E000001](./experiment_record.md#event-e000001) begins the append-only attempt timeline; the cited disposition is [Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075).

## Likely failure layer

recorder/oracle/harness ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Primary diagnosis

Attempt terminated with VideoError: source ticks exceed the configured video frame rate ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Confidence

high ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Competing explanations

- none recorded ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Disconfirming evidence

Raw calls, frames, states, and video are linked from the record ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Studied system or infrastructure

harness ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Recommended follow up

Review linked evidence without changing frozen results ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Reviewer notes

Diagnosis was generated only after behavior and oracle finalization ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).

## Unresolved ambiguity

Manual evidence review may refine cause ([Event E000040](./experiment_record.md#event-e000040), [video 18.075s](./video.mp4#t=18.075)).
