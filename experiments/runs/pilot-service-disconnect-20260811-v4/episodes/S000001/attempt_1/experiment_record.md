# Experiment record

Generated from `events.jsonl`.

<a id="event-e000001"></a>
## Event E000001

- Type: `attempt_started`
- UTC: `2026-08-11T06:42:23.043483Z`
- Elapsed: `0.093279` s
- Evidence: [video 00:00:00.093](./video.mp4#t=0.093)

```json
{
  "attempt_id": "S000001-A1",
  "elapsed_seconds": 0.093279382,
  "engine_id": "live_model_service_pilot_v1",
  "event_id": "E000001",
  "event_type": "attempt_started",
  "monotonic_ns": 19886377733433,
  "profile_id": "T5",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:42:23.043483Z",
  "video_time_seconds": 0.093279382
}
```

<a id="event-e000002"></a>
## Event E000002

- Type: `hidden_state_snapshot`
- UTC: `2026-08-11T06:42:23.051395Z`
- Elapsed: `0.101191` s
- Evidence: [video 00:00:00.101](./video.mp4#t=0.101)

```json
{
  "elapsed_seconds": 0.101191174,
  "event_id": "E000002",
  "event_type": "hidden_state_snapshot",
  "evidence": "capture:PRE_TASK",
  "monotonic_ns": 19886385645225,
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:42:23.051395Z",
  "video_time_seconds": 0.101191174
}
```

<a id="event-e000003"></a>
## Event E000003

- Type: `video_frame_captured`
- UTC: `2026-08-11T06:42:23.166430Z`
- Elapsed: `0.216226` s
- Evidence: [video 00:00:00.216](./video.mp4#t=0.216)

```json
{
  "elapsed_seconds": 0.216226227,
  "event_id": "E000003",
  "event_type": "video_frame_captured",
  "frame_sequence": 1,
  "hidden_state_recorded": true,
  "monotonic_ns": 19886500680278,
  "phase": "PRE_TASK",
  "robot_image_sha256": "35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:42:23.166430Z",
  "video_frame_index": 2,
  "video_time_seconds": 0.216226227
}
```

<a id="event-e000004"></a>
## Event E000004

- Type: `model_request`
- UTC: `2026-08-11T06:42:23.185275Z`
- Elapsed: `0.235071` s
- Evidence: [video 00:00:00.235](./video.mp4#t=0.235)

```json
{
  "call_id": "CALL-000001",
  "camera_frame_id": "robot_camera:1",
  "elapsed_seconds": 0.23507116,
  "event_id": "E000004",
  "event_type": "model_request",
  "frame_sequence": 1,
  "logical_agent": "hri",
  "model_id": "/workspace/models/gemma-4-26B-A4B-it",
  "monotonic_ns": 19886519525211,
  "request_id": "REQ-000001",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:42:23.185275Z",
  "video_time_seconds": 0.23507116
}
```

<a id="event-e000005"></a>
## Event E000005

- Type: `model_error`
- UTC: `2026-08-11T06:42:29.010107Z`
- Elapsed: `6.059903` s
- Evidence: [video 00:00:06.060](./video.mp4#t=6.060)

```json
{
  "call_id": "CALL-000001",
  "elapsed_seconds": 6.059903133,
  "error_message": "Gemma transport failed and the independent model-registry probe failed",
  "error_type": "InfrastructureInterruption",
  "event_id": "E000005",
  "event_type": "model_error",
  "logical_agent": "hri",
  "monotonic_ns": 19892344357184,
  "request_id": "REQ-000001",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:42:29.010107Z",
  "video_time_seconds": 6.059903133
}
```

<a id="event-e000006"></a>
## Event E000006

- Type: `request_frame_linked`
- UTC: `2026-08-11T06:42:29.013673Z`
- Elapsed: `6.063469` s
- Evidence: [video 00:00:06.063](./video.mp4#t=6.063)

- Request frame: [frames/35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3.png](./frames/35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3.png)

```json
{
  "call_id": "CALL-000001",
  "elapsed_seconds": 6.06346929,
  "error_event_id": "E000005",
  "event_id": "E000006",
  "event_type": "request_frame_linked",
  "image_path": "frames/35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3.png",
  "monotonic_ns": 19892347923341,
  "raw_image_sha256": "35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3",
  "request_id": "REQ-000001",
  "response_id": null,
  "scenario_id": "NM02",
  "source_frame_sha256": "04b571085e16b9f58c7f200a1060fe8f5f4f92e5d657626e52cbaf0a1ea791b9",
  "utc": "2026-08-11T06:42:29.013673Z",
  "video_time_seconds": 6.06346929
}
```

<a id="event-e000007"></a>
## Event E000007

- Type: `attempt_terminated`
- UTC: `2026-08-11T06:42:29.016068Z`
- Elapsed: `6.065864` s
- Evidence: [video 00:00:06.066](./video.mp4#t=6.066)

```json
{
  "elapsed_seconds": 6.065863848,
  "error_message": "Gemma transport failed and the independent model-registry probe failed",
  "error_type": "InfrastructureInterruption",
  "event_id": "E000007",
  "event_type": "attempt_terminated",
  "monotonic_ns": 19892350317899,
  "run_status": "INFRA_INTERRUPTED",
  "utc": "2026-08-11T06:42:29.016068Z",
  "video_time_seconds": 6.065863848
}
```
