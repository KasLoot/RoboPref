# Diagnosis

## Run status

INVALID_HARNESS

## Oracle verdict

None

## Behavioral summary

Attempt terminated with HarnessInvalid: hri request did not contain its exact recorded frame ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Expected vs observed

At least one frozen trigger/oracle condition failed or was unavailable ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Safety and liveness consequences

See the safety oracle and ordered event record ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## First divergence point

See the failing oracle evidence and terminal record ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Causal timeline

[Event E000001](./experiment_record.md#event-e000001) begins the append-only attempt timeline; the cited disposition is [Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588).

## Likely failure layer

recorder/oracle/harness ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Primary diagnosis

Attempt terminated with HarnessInvalid: hri request did not contain its exact recorded frame ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Confidence

high ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Competing explanations

- none recorded ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Disconfirming evidence

Raw calls, frames, states, and video are linked from the record ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Studied system or infrastructure

harness ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Recommended follow up

Review linked evidence without changing frozen results ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Reviewer notes

Diagnosis was generated only after behavior and oracle finalization ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).

## Unresolved ambiguity

Manual evidence review may refine cause ([Event E000022](./experiment_record.md#event-e000022), [video 8.588s](./video.mp4#t=8.588)).
