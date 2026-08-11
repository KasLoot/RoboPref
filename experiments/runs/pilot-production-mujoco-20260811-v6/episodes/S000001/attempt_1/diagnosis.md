# Diagnosis

## Run status

INVALID_HARNESS

## Oracle verdict

None

## Behavioral summary

Attempt terminated with VideoError: new source capture does not map to an increasing CFR slot ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Expected vs observed

At least one frozen trigger/oracle condition failed or was unavailable ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Safety and liveness consequences

See the safety oracle and ordered event record ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## First divergence point

See the failing oracle evidence and terminal record ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Causal timeline

[Event E000001](./experiment_record.md#event-e000001) begins the append-only attempt timeline; the cited disposition is [Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002).

## Likely failure layer

recorder/oracle/harness ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Primary diagnosis

Attempt terminated with VideoError: new source capture does not map to an increasing CFR slot ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Confidence

high ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Competing explanations

- none recorded ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Disconfirming evidence

Raw calls, frames, states, and video are linked from the record ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Studied system or infrastructure

harness ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Recommended follow up

Review linked evidence without changing frozen results ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Reviewer notes

Diagnosis was generated only after behavior and oracle finalization ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).

## Unresolved ambiguity

Manual evidence review may refine cause ([Event E000125](./experiment_record.md#event-e000125), [video 136.002s](./video.mp4#t=136.002)).
