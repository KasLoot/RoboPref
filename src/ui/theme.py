"""Light visual system for the RoboPref operator workspace."""

from __future__ import annotations


APP_CSS = r"""
:root {
  --rp-bg: #f4f6fb;
  --rp-panel: #ffffff;
  --rp-panel-soft: #f8fafc;
  --rp-line: #e2e8f0;
  --rp-line-strong: #cbd5e1;
  --rp-text: #172033;
  --rp-muted: #68758a;
  --rp-faint: #94a3b8;
  --rp-primary: #4f46e5;
  --rp-primary-soft: #eef2ff;
  --rp-cyan: #0284c7;
  --rp-green: #059669;
  --rp-green-soft: #ecfdf5;
  --rp-amber: #d97706;
  --rp-amber-soft: #fffbeb;
  --rp-red: #dc2626;
  --rp-red-soft: #fef2f2;
  --rp-shadow: 0 14px 38px rgba(30, 41, 59, 0.08);
  --rp-shadow-soft: 0 5px 18px rgba(30, 41, 59, 0.06);
  --rp-radius: 18px;
}

html, body, #app {
  min-height: 100%;
  background: var(--rp-bg);
}

body {
  color: var(--rp-text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background-image:
    radial-gradient(circle at 8% -5%, rgba(79, 70, 229, 0.07), transparent 28rem),
    radial-gradient(circle at 95% 5%, rgba(14, 165, 233, 0.06), transparent 30rem);
}

.q-layout, .q-page-container, .nicegui-content {
  background: transparent !important;
}

.nicegui-content {
  padding: 0 !important;
}

.rp-header {
  z-index: 60;
  min-height: 68px;
  padding: 0 20px;
  color: var(--rp-text) !important;
  background: rgba(255, 255, 255, 0.92) !important;
  border-bottom: 1px solid rgba(203, 213, 225, 0.82);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}

.rp-header-inner {
  width: min(1880px, 100%);
  min-height: 68px;
  margin: 0 auto;
  gap: 10px;
}

.rp-brand-mark {
  width: 40px;
  height: 40px;
  flex: 0 0 auto;
  color: #fff;
  background: linear-gradient(145deg, #625bf6, #4338ca);
  border-radius: 13px;
  box-shadow: 0 8px 20px rgba(79, 70, 229, 0.24);
}

.rp-brand-title {
  color: var(--rp-text);
  font-size: 1.02rem;
  font-weight: 760;
  letter-spacing: -0.025em;
  line-height: 1.15;
}

.rp-brand-subtitle {
  color: var(--rp-muted);
  font-size: 0.69rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.rp-busy-bar {
  position: fixed;
  z-index: 70;
  top: 68px;
  left: 0;
  right: 0;
  height: 3px !important;
}

.rp-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  width: min(1880px, 100%);
  height: calc(100dvh - 68px);
  margin: 0 auto;
  padding: 16px;
  gap: 16px;
  overflow: hidden;
}

.rp-left-pane {
  min-width: 0;
  min-height: 0;
  height: 100%;
  gap: 14px;
  overflow-x: hidden;
  overflow-y: auto;
  padding-right: 2px;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e1 transparent;
}

.rp-surface {
  width: 100%;
  margin: 0;
  padding: 0;
  color: var(--rp-text);
  background: rgba(255, 255, 255, 0.97) !important;
  border: 1px solid var(--rp-line);
  border-radius: var(--rp-radius) !important;
  box-shadow: var(--rp-shadow-soft) !important;
  overflow: hidden;
}

.rp-camera-card {
  display: flex !important;
  flex: 0 0 calc(50dvh - 50px);
  min-height: 330px;
  flex-direction: column;
  align-items: stretch;
}

.rp-monitor-card {
  flex: 0 0 auto;
  min-height: 280px;
}

.rp-section-header {
  width: 100%;
  min-height: 64px;
  padding: 12px 15px;
  gap: 10px;
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid var(--rp-line);
}

.rp-section-icon {
  width: 36px;
  height: 36px;
  flex: 0 0 auto;
  border-radius: 11px;
}

.rp-section-icon.camera {
  color: #0369a1;
  background: #e0f2fe;
}

.rp-section-icon.monitor {
  color: #047857;
  background: #d1fae5;
}

.rp-eyebrow {
  color: var(--rp-muted);
  font-size: 0.62rem;
  font-weight: 720;
  letter-spacing: 0.12em;
  line-height: 1.2;
  text-transform: uppercase;
}

.rp-section-title {
  max-width: 42ch;
  overflow: hidden;
  color: var(--rp-text);
  font-size: 0.94rem;
  font-weight: 710;
  letter-spacing: -0.012em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rp-quiet-link {
  padding: 7px 10px;
  color: var(--rp-primary) !important;
  background: var(--rp-primary-soft);
  border: 1px solid #dfe3ff;
  border-radius: 10px;
  font-size: 0.7rem;
  font-weight: 680;
  text-decoration: none !important;
}

.rp-quiet-link:hover,
.rp-quiet-link:focus-visible {
  color: #3730a3 !important;
  background: #e0e7ff;
}

.rp-camera-stage {
  position: relative;
  width: 100%;
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  background: #111827;
}

.rp-camera-feed {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #111827;
}

.rp-camera-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  min-height: 29px;
  padding: 6px 9px;
  gap: 7px;
  color: #f8fafc;
  background: rgba(15, 23, 42, 0.77);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 999px;
  backdrop-filter: blur(8px);
  font-size: 0.66rem;
  font-weight: 740;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.rp-camera-badge.degraded .rp-status-dot {
  background: #f87171;
  box-shadow: 0 0 0 4px rgba(248, 113, 113, 0.17);
}

.rp-camera-badge.unknown .rp-status-dot {
  background: #94a3b8;
  box-shadow: none;
}

.rp-camera-footer {
  width: 100%;
  min-height: 43px;
  padding: 9px 14px;
  gap: 8px;
  color: var(--rp-muted);
  background: #fff;
  border-top: 1px solid var(--rp-line);
  font-size: 0.73rem;
}

.rp-mono {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.66rem;
  letter-spacing: 0.04em;
}

.rp-agent-section {
  flex: 0 0 auto;
  padding: 1px 1px 5px;
}

.rp-group-title {
  color: var(--rp-text);
  font-size: 0.78rem;
  font-weight: 720;
}

.rp-muted {
  color: var(--rp-muted);
  font-size: 0.69rem;
  line-height: 1.45;
}

.rp-agent-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  width: 100%;
  gap: 9px;
}

.rp-mini-agent {
  min-width: 0;
  margin: 0;
  padding: 11px;
  color: var(--rp-text);
  background: rgba(255, 255, 255, 0.94) !important;
  border: 1px solid var(--rp-line);
  border-radius: 14px !important;
  box-shadow: 0 3px 12px rgba(30, 41, 59, 0.04) !important;
}

.rp-mini-agent-row {
  gap: 8px;
}

.rp-mini-agent-icon {
  width: 28px;
  height: 28px;
  flex: 0 0 auto;
  color: #475569;
  background: #f1f5f9;
  border-radius: 9px;
}

.rp-mini-agent-name {
  min-width: 0;
  overflow: hidden;
  font-size: 0.72rem;
  font-weight: 690;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rp-status-dot {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  background: #94a3b8;
  border-radius: 999px;
}

.rp-agent-status {
  min-height: 27px;
  padding: 4px 8px;
  gap: 6px;
  color: #64748b;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  font-size: 0.62rem;
  font-weight: 700;
  white-space: nowrap;
}

.rp-agent-status.working {
  color: #047857;
  background: var(--rp-green-soft);
  border-color: #a7f3d0;
}

.rp-agent-status.working .rp-status-dot,
.rp-agent-status.ready .rp-status-dot,
.rp-health-pill.online .rp-status-dot,
.rp-camera-badge.online .rp-status-dot {
  background: #10b981;
  box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.12);
}

.rp-agent-status.ready {
  color: #0369a1;
  background: #f0f9ff;
  border-color: #bae6fd;
}

.rp-agent-status.ready .rp-status-dot {
  background: #0ea5e9;
  box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1);
}

.rp-agent-status.danger {
  color: #b91c1c;
  background: var(--rp-red-soft);
  border-color: #fecaca;
}

.rp-agent-status.danger .rp-status-dot,
.rp-health-pill.degraded .rp-status-dot {
  background: #ef4444;
  box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.1);
}

.rp-monitor-body {
  padding: 15px;
  gap: 12px;
}

.rp-monitor-empty {
  width: 100%;
  min-height: 78px;
  padding: 12px 14px;
  gap: 12px;
  color: var(--rp-faint);
  background: var(--rp-panel-soft);
  border: 1px dashed var(--rp-line-strong);
  border-radius: 13px;
}

.rp-monitor-empty-title {
  color: var(--rp-text);
  font-size: 0.82rem;
  font-weight: 690;
}

.rp-monitor-summary {
  gap: 15px;
}

.rp-field-label {
  color: var(--rp-muted);
  font-size: 0.59rem;
  font-weight: 730;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.rp-monitor-instruction {
  color: var(--rp-text);
  font-size: 0.92rem;
  font-weight: 670;
  line-height: 1.42;
}

.rp-monitor-cycle {
  min-width: 64px;
  padding: 8px 10px;
  text-align: center;
  background: #f8fafc;
  border: 1px solid var(--rp-line);
  border-radius: 11px;
}

.rp-monitor-cycle-value {
  color: var(--rp-text);
  font-size: 1rem;
  font-weight: 760;
}

.rp-observation {
  padding: 10px 12px;
  gap: 9px;
  color: #075985;
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 12px;
}

.rp-observation-copy {
  color: #334155;
  font-size: 0.76rem;
  line-height: 1.42;
}

.rp-monitor-criteria {
  gap: 6px;
}

.rp-monitor-criteria .q-icon {
  color: var(--rp-green);
  margin-top: 1px;
}

.rp-monitor-criterion {
  color: #475569;
  font-size: 0.73rem;
  line-height: 1.35;
}

.rp-monitor-warning {
  padding: 9px 11px;
  gap: 8px;
  color: #92400e;
  background: var(--rp-amber-soft);
  border: 1px solid #fde68a;
  border-radius: 11px;
  font-size: 0.72rem;
}

.rp-chat-panel {
  position: sticky;
  top: 0;
  display: flex !important;
  min-width: 0;
  min-height: 0;
  height: 100%;
  flex-direction: column;
  overflow: hidden;
  box-shadow: var(--rp-shadow) !important;
}

.rp-chat-header {
  width: 100%;
  min-height: 70px;
  padding: 13px 16px;
  gap: 11px;
  border-bottom: 1px solid var(--rp-line);
}

.rp-chat-avatar {
  width: 38px;
  height: 38px;
  color: #4338ca;
  background: var(--rp-primary-soft);
  border: 1px solid #dfe3ff;
  border-radius: 12px;
}

.rp-chat-title {
  color: var(--rp-text);
  font-size: 0.98rem;
  font-weight: 750;
  letter-spacing: -0.018em;
}

.rp-chat-subtitle {
  color: var(--rp-muted);
  font-size: 0.69rem;
}

.rp-new-chat {
  min-height: 35px;
  padding: 0 10px;
  color: #475569 !important;
  background: #f8fafc !important;
  border: 1px solid var(--rp-line);
  border-radius: 10px !important;
  font-size: 0.72rem;
  font-weight: 650;
}

.rp-context-slot {
  flex: 0 0 auto;
  padding: 0;
  gap: 0;
}

.rp-context-slot:empty {
  display: none;
}

.rp-context-banner {
  width: calc(100% - 24px);
  margin: 10px 12px 0;
  padding: 10px 12px;
  gap: 10px;
  color: #475569;
  background: #f8fafc;
  border: 1px solid var(--rp-line);
  border-radius: 13px;
}

.rp-context-banner.attention {
  color: #92400e;
  background: var(--rp-amber-soft);
  border-color: #fde68a;
}

.rp-context-banner.danger {
  color: #b91c1c;
  background: var(--rp-red-soft);
  border-color: #fecaca;
}

.rp-context-banner.complete {
  color: #047857;
  background: var(--rp-green-soft);
  border-color: #a7f3d0;
}

.rp-context-title {
  color: currentColor;
  font-size: 0.76rem;
  font-weight: 720;
}

.rp-context-copy {
  color: #64748b;
  font-size: 0.68rem;
  line-height: 1.4;
}

.rp-context-goal {
  color: var(--rp-text);
  font-size: 0.8rem;
  font-weight: 650;
  line-height: 1.38;
}

.rp-chat-scroll {
  flex: 1 1 auto;
  width: 100%;
  min-height: 0;
  height: auto;
  background:
    linear-gradient(rgba(248, 250, 252, 0.92), rgba(248, 250, 252, 0.92)),
    radial-gradient(circle at 20% 0%, rgba(79, 70, 229, 0.055), transparent 22rem);
}

.rp-chat-history,
.rp-live-slot {
  padding: 18px 18px 0;
  gap: 15px;
}

.rp-live-slot {
  padding-top: 15px;
  padding-bottom: 18px;
}

.rp-chat-empty {
  width: min(430px, calc(100% - 28px));
  margin: clamp(40px, 11vh, 100px) auto 30px;
  padding: 28px;
  gap: 10px;
  text-align: center;
  background: rgba(255, 255, 255, 0.68);
  border: 1px dashed var(--rp-line-strong);
  border-radius: 18px;
}

.rp-empty-orb {
  width: 52px;
  height: 52px;
  color: #fff;
  background: linear-gradient(145deg, #6366f1, #4f46e5);
  border-radius: 17px;
  box-shadow: 0 12px 28px rgba(79, 70, 229, 0.2);
}

.rp-empty-title {
  color: var(--rp-text);
  font-size: 1rem;
  font-weight: 750;
}

.rp-empty-copy {
  max-width: 44ch;
  color: var(--rp-muted);
  font-size: 0.76rem;
  line-height: 1.55;
}

.rp-message-row {
  gap: 9px;
}

.rp-message-stack {
  max-width: min(82%, 760px);
  gap: 5px;
}

.rp-message-stack.user {
  max-width: min(78%, 680px);
}

.rp-message-avatar {
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  color: #4338ca;
  background: var(--rp-primary-soft);
  border: 1px solid #dfe3ff;
  border-radius: 10px;
}

.rp-message-avatar.working {
  color: #fff;
  background: var(--rp-primary);
  border-color: transparent;
}

.rp-message-bubble {
  width: fit-content;
  max-width: 100%;
  padding: 10px 13px;
  font-size: 0.8rem;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.rp-message-bubble.user {
  color: #fff;
  background: linear-gradient(145deg, #5b52eb, #4f46e5);
  border-radius: 15px 15px 4px 15px;
  box-shadow: 0 7px 18px rgba(79, 70, 229, 0.17);
}

.rp-message-bubble.assistant {
  color: #263247;
  background: #fff;
  border: 1px solid #dfe5ee;
  border-radius: 4px 15px 15px 15px;
  box-shadow: 0 5px 16px rgba(30, 41, 59, 0.055);
}

.rp-message-bubble.assistant.error {
  color: #991b1b;
  background: var(--rp-red-soft);
  border-color: #fecaca;
}

.rp-message-bubble.streaming {
  min-width: 122px;
  color: #475569;
}

.rp-message-bubble .nicegui-markdown p:first-child {
  margin-top: 0;
}

.rp-message-bubble .nicegui-markdown p:last-child {
  margin-bottom: 0;
}

.rp-message-time {
  padding: 0 3px;
  color: var(--rp-faint);
  font-size: 0.6rem;
}

.rp-trace-expansion {
  width: min(100%, 660px);
  color: #475569;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 11px;
  overflow: hidden;
}

.rp-trace-expansion .q-item {
  min-height: 36px;
  padding: 6px 10px;
  font-size: 0.68rem;
  font-weight: 660;
}

.rp-trace-expansion .q-expansion-item__content {
  border-top: 1px solid #e2e8f0;
}

.rp-trace-list {
  max-height: 320px;
  padding: 9px;
  gap: 8px;
  overflow-y: auto;
}

.rp-trace-item {
  padding: 8px 9px;
  gap: 5px;
  background: #fff;
  border: 1px solid #e8edf3;
  border-radius: 9px;
}

.rp-trace-kind {
  color: #4338ca;
  font-size: 0.62rem;
  font-weight: 740;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.rp-trace-node {
  color: var(--rp-faint);
  font-size: 0.59rem;
}

.rp-trace-content {
  max-height: 180px;
  overflow: auto;
  color: #475569;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.66rem;
  line-height: 1.48;
  overflow-wrap: anywhere;
}

.rp-jump-button {
  position: absolute;
  z-index: 4;
  right: 24px;
  bottom: 84px;
  color: #4338ca !important;
  background: #fff !important;
  border: 1px solid #dfe3ff;
  border-radius: 999px !important;
  box-shadow: var(--rp-shadow-soft);
  font-size: 0.68rem;
}

.rp-composer {
  flex: 0 0 auto;
  width: calc(100% - 24px);
  min-height: 58px;
  margin: 10px 12px 12px;
  padding: 7px 8px 7px 14px;
  gap: 8px;
  background: #fff;
  border: 1px solid #d7dee9;
  border-radius: 16px;
  box-shadow: 0 8px 26px rgba(30, 41, 59, 0.08);
}

.rp-composer:focus-within {
  border-color: #a5b4fc;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.10), 0 8px 26px rgba(30, 41, 59, 0.08);
}

.rp-chat-input .q-field__control,
.rp-chat-input .q-field__native {
  min-height: 42px !important;
  color: var(--rp-text);
  font-size: 0.8rem;
}

.rp-send-button {
  width: 40px;
  height: 40px;
  color: #fff !important;
  background: linear-gradient(145deg, #625bf6, #4f46e5) !important;
  box-shadow: 0 7px 18px rgba(79, 70, 229, 0.22);
}

.rp-health-row {
  gap: 5px;
}

.rp-health-pill {
  min-height: 28px;
  padding: 4px 8px;
  gap: 6px;
  color: #64748b;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  font-size: 0.62rem;
  font-weight: 650;
}

.rp-health-pill.degraded {
  color: #b91c1c;
  background: var(--rp-red-soft);
  border-color: #fecaca;
}

.rp-runtime-chip {
  min-height: 31px;
  padding: 5px 10px;
  gap: 6px;
  color: #475569;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  font-size: 0.66rem;
  font-weight: 700;
}

.rp-runtime-chip.working,
.rp-runtime-chip.complete,
.rp-runtime-chip.ready {
  color: #047857;
  background: var(--rp-green-soft);
  border-color: #a7f3d0;
}

.rp-runtime-chip.attention {
  color: #92400e;
  background: var(--rp-amber-soft);
  border-color: #fde68a;
}

.rp-runtime-chip.danger {
  color: #b91c1c;
  background: var(--rp-red-soft);
  border-color: #fecaca;
}

.rp-header-stop {
  min-height: 34px;
  color: #b91c1c !important;
  background: var(--rp-red-soft) !important;
  border: 1px solid #fecaca;
  border-radius: 10px !important;
  font-size: 0.69rem;
  font-weight: 690;
  text-transform: none;
}

.rp-primary-action,
.rp-danger-action {
  min-height: 36px;
  border-radius: 10px !important;
  font-size: 0.72rem;
  font-weight: 680;
}

.rp-primary-action {
  color: #fff !important;
  background: var(--rp-primary) !important;
}

.rp-danger-action {
  color: #fff !important;
  background: var(--rp-red) !important;
}

.rp-dialog-card {
  width: min(520px, calc(100vw - 28px));
  padding: 21px;
  gap: 16px;
  color: var(--rp-text);
  background: #fff !important;
  border: 1px solid var(--rp-line);
  border-radius: 18px !important;
  box-shadow: 0 24px 70px rgba(30, 41, 59, 0.18) !important;
}

.rp-dialog-title {
  color: var(--rp-text);
  font-size: 1rem;
  font-weight: 750;
}

.rp-dialog-copy,
.rp-dialog-note,
.rp-safety-copy {
  color: var(--rp-muted);
  font-size: 0.75rem;
  line-height: 1.55;
}

.rp-dialog-note {
  padding: 9px 11px;
  color: #92400e;
  background: var(--rp-amber-soft);
  border: 1px solid #fde68a;
  border-radius: 10px;
}

.rp-safety-copy {
  color: #b91c1c;
}

.rp-danger-icon,
.rp-reset-icon {
  width: 42px;
  height: 42px;
  flex: 0 0 auto;
  border-radius: 13px;
}

.rp-danger-icon {
  color: #b91c1c;
  background: var(--rp-red-soft);
}

.rp-reset-icon {
  color: #4338ca;
  background: var(--rp-primary-soft);
}

@media (max-width: 1050px) {
  body { overflow-y: auto; }
  .rp-workspace {
    display: flex;
    flex-direction: column;
    height: auto;
    overflow: visible;
  }
  .rp-left-pane { display: contents; }
  .rp-camera-card { order: 1; }
  .rp-chat-panel { order: 2; }
  .rp-monitor-card { order: 3; }
  .rp-agent-section { order: 4; }
  .rp-camera-card { flex-basis: min(52dvh, 520px); }
  .rp-chat-panel { position: relative; min-height: 70dvh; height: 78dvh; }
  .rp-agent-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}

@media (max-width: 700px) {
  .rp-header { padding: 0 10px; }
  .rp-brand-subtitle, .rp-health-label { display: none; }
  .rp-health-pill { min-width: 27px; padding: 6px; gap: 0; }
  .rp-runtime-chip { padding: 5px 8px; }
  .rp-runtime-chip .nicegui-label { display: none; }
  .rp-header-stop { min-width: 34px; padding: 0 8px; }
  .rp-header-stop .q-btn__content > span:not(.q-icon) { display: none; }
  .rp-workspace { padding: 10px; gap: 10px; }
  .rp-left-pane { gap: 10px; }
  .rp-camera-card { min-height: 270px; flex-basis: min(44dvh, 520px); }
  .rp-monitor-card { min-height: 250px; }
  .rp-agent-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .rp-chat-panel { min-height: 76dvh; height: 82dvh; }
  .rp-chat-header { padding: 11px 12px; }
  .rp-chat-history, .rp-live-slot { padding-left: 11px; padding-right: 11px; }
  .rp-message-stack, .rp-message-stack.user { max-width: 88%; }
  .rp-context-banner { flex-wrap: wrap; }
  .rp-context-banner .q-space { display: none; }
  .rp-composer {
    width: auto;
    align-self: stretch;
    margin: 8px max(8px, env(safe-area-inset-right))
      max(8px, env(safe-area-inset-bottom))
      max(8px, env(safe-area-inset-left));
  }
}

@media (max-width: 520px) {
  .rp-agent-grid { grid-template-columns: minmax(0, 1fr); }
}

@media (max-width: 350px) {
  .rp-brand-copy { display: none; }
}
"""


__all__ = ["APP_CSS"]
