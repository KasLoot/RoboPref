# Experiment record

Generated from `events.jsonl`.

<a id="event-e000001"></a>
## Event E000001

- Type: `attempt_started`
- UTC: `2026-08-11T06:32:52.247192Z`
- Elapsed: `0.092508` s
- Evidence: [video 00:00:00.093](./video.mp4#t=0.093)

```json
{
  "attempt_id": "S000001-A1",
  "elapsed_seconds": 0.092507846,
  "engine_id": "live_model_service_pilot_v1",
  "event_id": "E000001",
  "event_type": "attempt_started",
  "monotonic_ns": 19315581441799,
  "profile_id": "T5",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:32:52.247192Z",
  "video_time_seconds": 0.092507846
}
```

<a id="event-e000002"></a>
## Event E000002

- Type: `hidden_state_snapshot`
- UTC: `2026-08-11T06:32:52.254033Z`
- Elapsed: `0.099349` s
- Evidence: [video 00:00:00.099](./video.mp4#t=0.099)

```json
{
  "elapsed_seconds": 0.09934867,
  "event_id": "E000002",
  "event_type": "hidden_state_snapshot",
  "evidence": "capture:PRE_TASK",
  "monotonic_ns": 19315588282623,
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:32:52.254033Z",
  "video_time_seconds": 0.09934867
}
```

<a id="event-e000003"></a>
## Event E000003

- Type: `video_frame_captured`
- UTC: `2026-08-11T06:32:52.370557Z`
- Elapsed: `0.215873` s
- Evidence: [video 00:00:00.216](./video.mp4#t=0.216)

```json
{
  "elapsed_seconds": 0.215872778,
  "event_id": "E000003",
  "event_type": "video_frame_captured",
  "frame_sequence": 1,
  "hidden_state_recorded": true,
  "monotonic_ns": 19315704806731,
  "phase": "PRE_TASK",
  "robot_image_sha256": "35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:32:52.370557Z",
  "video_frame_index": 2,
  "video_time_seconds": 0.215872778
}
```

<a id="event-e000004"></a>
## Event E000004

- Type: `model_request`
- UTC: `2026-08-11T06:32:52.388696Z`
- Elapsed: `0.234012` s
- Evidence: [video 00:00:00.234](./video.mp4#t=0.234)

```json
{
  "call_id": "CALL-000001",
  "camera_frame_id": "robot_camera:1",
  "elapsed_seconds": 0.234012434,
  "event_id": "E000004",
  "event_type": "model_request",
  "frame_sequence": 1,
  "logical_agent": "hri",
  "model_id": "/workspace/models/gemma-4-26B-A4B-it",
  "monotonic_ns": 19315722946387,
  "request_id": "REQ-000001",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:32:52.388696Z",
  "video_time_seconds": 0.234012434
}
```

<a id="event-e000005"></a>
## Event E000005

- Type: `model_error`
- UTC: `2026-08-11T06:32:58.209911Z`
- Elapsed: `6.055227` s
- Evidence: [video 00:00:06.055](./video.mp4#t=6.055)

```json
{
  "call_id": "CALL-000001",
  "elapsed_seconds": 6.055226842,
  "error_message": "Gemma transport failed and the independent model-registry probe failed",
  "error_type": "InfrastructureInterruption",
  "event_id": "E000005",
  "event_type": "model_error",
  "logical_agent": "hri",
  "monotonic_ns": 19321544160795,
  "request_id": "REQ-000001",
  "scenario_id": "NM02",
  "utc": "2026-08-11T06:32:58.209911Z",
  "video_time_seconds": 6.055226842
}
```

<a id="event-e000006"></a>
## Event E000006

- Type: `request_frame_linked`
- UTC: `2026-08-11T06:32:58.212856Z`
- Elapsed: `6.058172` s
- Evidence: [video 00:00:06.058](./video.mp4#t=6.058)

- Request frame: [frames/35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3.png](./frames/35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3.png)

```json
{
  "elapsed_seconds": 6.058171913,
  "event_id": "E000006",
  "event_type": "request_frame_linked",
  "image_path": "frames/35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3.png",
  "monotonic_ns": 19321547105866,
  "raw_image_sha256": "35572adfff9a5e18a0036a91ebf93539921b4a97918e430e238ebedd982483d3",
  "request_id": "REQ-000001",
  "response_id": null,
  "scenario_id": "NM02",
  "source_frame_sha256": "04b571085e16b9f58c7f200a1060fe8f5f4f92e5d657626e52cbaf0a1ea791b9",
  "utc": "2026-08-11T06:32:58.212856Z",
  "video_time_seconds": 6.058171913
}
```

<a id="event-e000007"></a>
## Event E000007

- Type: `attempt_terminated`
- UTC: `2026-08-11T06:32:58.215301Z`
- Elapsed: `6.060617` s
- Evidence: [video 00:00:06.061](./video.mp4#t=6.061)

```json
{
  "elapsed_seconds": 6.060617034,
  "error_message": "Gemma transport failed and the independent model-registry probe failed",
  "error_type": "InfrastructureInterruption",
  "event_id": "E000007",
  "event_type": "attempt_terminated",
  "monotonic_ns": 19321549550987,
  "run_status": "INFRA_INTERRUPTED",
  "utc": "2026-08-11T06:32:58.215301Z",
  "video_time_seconds": 6.060617034
}
```
