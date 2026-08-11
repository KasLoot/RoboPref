# Campaign changelog

- 2026-08-11T06:29:50.116452Z: protocol version frozen.
- 2026-08-11T06:34:02.795633Z: campaign version terminated after preserving S000001 attempt 1 as `INFRA_INTERRUPTED`; the outage and independent failed probe were correctly classified, but the partial-artifact audit found that request `REQ-000001` was not machine-linked to its model-error event. The linkage contract will be corrected and the full disconnect/recovery condition restarted as v4 without authorizing a v3 retry.
