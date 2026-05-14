import dataclasses
import json
import mimetypes
import shutil
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import quote
from urllib.parse import urlparse

import imageio.v3 as iio
import tyro


_FRAME_PAD_WIDTH = 6
_FRAME_LOCKS_GUARD = threading.Lock()
_FRAME_LOCKS: dict[str, threading.Lock] = {}

_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spec Trace Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4eee1;
      --ink: #18212b;
      --muted: #6b7280;
      --border: rgba(148, 163, 184, 0.22);
      --panel: rgba(255, 252, 246, 0.90);
      --shadow: 0 18px 40px rgba(24, 33, 43, 0.08);
      --accent: #c76e2b;
      --draft: #0f766e;
      --vlm: #2563eb;
      --full: #b91c1c;
      --gutter-label: #8a5c36;
      --gutter-muted: #6f7281;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(199, 110, 43, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 24%),
        linear-gradient(180deg, #f9f4ea 0%, var(--bg) 48%, #e8e0d3 100%);
    }
    header {
      flex: 0 0 auto;
      padding: 24px 24px 14px;
    }
    .title {
      margin: 0;
      font-size: 28px;
      font-weight: 780;
      letter-spacing: -0.04em;
    }
    .viewer-shell {
      flex: 1 1 auto;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 300px;
      gap: 16px;
      min-height: 0;
      padding: 0 20px 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .sidebar {
      padding: 18px;
      position: sticky;
      top: 16px;
    }
    .sidebar h3, .stats-sidebar h3 {
      margin: 0 0 14px;
      font-size: 13px;
      font-weight: 780;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7b5b3f;
    }
    .stats-sidebar {
      padding: 18px;
      position: sticky;
      top: 16px;
      align-self: start;
    }
    .field {
      margin-bottom: 12px;
    }
    .field label {
      display: block;
      margin-bottom: 7px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    select, input, button {
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      background: #fffdf8;
      padding: 11px 12px;
      font: inherit;
      color: var(--ink);
    }
    button {
      cursor: pointer;
      border: none;
      color: #fffef9;
      background: linear-gradient(135deg, #ce7a33, #b55a28);
      font-weight: 700;
      letter-spacing: 0.01em;
      box-shadow: 0 12px 24px rgba(181, 90, 40, 0.22);
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .hint {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(148, 163, 184, 0.16);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .content {
      min-width: 0;
      min-height: 0;
      align-self: stretch;
      height: 100%;
    }
    .hero {
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 16px;
      overflow: hidden;
      min-height: 0;
      height: 100%;
    }
    .hero-head {
      display: block;
      min-height: 0;
      flex: 0 0 auto;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 20px;
      flex: 1 1 auto;
      min-height: 0;
      height: 100%;
      align-items: stretch;
    }
    .hero-grid > * {
      min-height: 0;
      height: 100%;
    }
    .stats-sections {
      display: grid;
      grid-template-rows: auto auto;
      gap: 16px;
      min-height: 0;
      align-content: start;
    }
    .info-card {
      padding: 18px;
      min-height: 0;
      overflow: auto;
    }
    .info-card h3 {
      margin: 0 0 14px;
      font-size: 13px;
      font-weight: 780;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7b5b3f;
    }
    .kv-list {
      display: grid;
      gap: 10px;
    }
    .kv-item {
      display: grid;
      grid-template-columns: minmax(104px, auto) minmax(0, 1fr);
      gap: 12px;
      align-items: baseline;
      font-size: 14px;
    }
    .kv-label {
      color: var(--gutter-muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
    }
    .kv-value {
      color: var(--ink);
      font-size: 14px;
      line-height: 1.4;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .draft {
      color: var(--draft);
    }
    .vlm {
      color: var(--vlm);
    }
    .full {
      color: var(--full);
    }
    .action-item {
      grid-template-columns: 1fr;
      gap: 6px;
      align-items: start;
    }
    .action-list {
      display: grid;
      gap: 3px;
      font-size: 13px;
      line-height: 1.4;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--ink);
    }
    .action-line {
      white-space: nowrap;
    }
    .action-line-placeholder {
      color: var(--muted);
    }
    .stage {
      min-width: 0;
      min-height: 0;
      height: 100%;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 12px;
      overflow: hidden;
    }
    .stage-title {
      margin: 0;
      font-size: 20px;
      font-weight: 760;
      letter-spacing: -0.03em;
    }
    .stage-subtitle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 4px;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
      overflow: hidden;
    }
    .frame-shell {
      min-width: 0;
      min-height: 0;
      width: 100%;
      height: 100%;
      max-height: 100%;
      aspect-ratio: 1 / 1;
      justify-self: center;
      align-self: center;
      overflow: hidden;
      border-radius: 24px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background:
        radial-gradient(circle at top, rgba(199, 110, 43, 0.14), transparent 44%),
        radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.08), transparent 34%),
        linear-gradient(180deg, rgba(255, 252, 246, 0.96), rgba(236, 227, 214, 0.94));
      display: grid;
      place-items: center;
      position: relative;
    }
    .frame-shell img {
      height: 100%;
      width: auto;
      max-width: 100%;
      max-height: 100%;
      display: block;
      object-fit: contain;
      background: transparent;
      justify-self: center;
      align-self: center;
    }
    .frame-empty {
      padding: 18px;
      text-align: center;
      color: #7b5b3f;
      font-size: 14px;
      line-height: 1.5;
      max-width: 320px;
    }
    .controls-panel {
      padding: 12px 14px 14px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.62), rgba(252, 247, 239, 0.64));
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
      min-height: 0;
    }
    .timeline-readout {
      margin-top: 0;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .frame-label {
      font-size: 13px;
      font-weight: 500;
      letter-spacing: 0.01em;
      color: var(--ink);
      line-height: 1.5;
    }
    .frame-meta-label {
      color: #6b7280;
      font-weight: 500;
    }
    .frame-meta-value {
      color: var(--ink);
      font-weight: 500;
    }
    .frame-meta-route {
      font-weight: 600;
    }
    .frame-meta-value.frame-meta-route.draft {
      color: var(--draft);
    }
    .frame-meta-value.frame-meta-route.vlm {
      color: var(--vlm);
    }
    .frame-meta-value.frame-meta-route.full {
      color: var(--full);
    }
    .timeline-readout .router-legend {
      margin-top: 0;
      justify-content: flex-end;
      flex: 0 0 auto;
    }
    .timeline-shell {
      margin-top: 10px;
    }
    .timeline-track {
      position: relative;
      height: 54px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(247, 239, 227, 0.86));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.62);
      cursor: pointer;
      user-select: none;
      touch-action: none;
    }
    .timeline-rail {
      position: absolute;
      left: 14px;
      right: 14px;
      top: 50%;
      height: 26px;
      transform: translateY(-50%);
    }
    .timeline-rail::before {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: 50%;
      height: 8px;
      border-radius: 999px;
      transform: translateY(-50%);
      background: rgba(199, 110, 43, 0.18);
    }
    .timeline-route-segment {
      position: absolute;
      top: 50%;
      height: 18px;
      border-radius: 999px;
      transform: translateY(-50%);
      opacity: 0;
      pointer-events: none;
      transition: opacity 120ms ease;
    }
    .timeline-route-segment.visible {
      opacity: 0.24;
    }
    .timeline-route-segment.draft {
      background: rgba(15, 118, 110, 0.88);
    }
    .timeline-route-segment.vlm {
      background: rgba(37, 99, 235, 0.82);
    }
    .timeline-route-segment.full {
      background: rgba(185, 28, 28, 0.82);
    }
    .router-points {
      position: absolute;
      inset: 0;
    }
    .route-tick {
      position: absolute;
      top: 50%;
      width: 14px;
      height: 26px;
      transform: translate(-50%, -50%);
      cursor: pointer;
    }
    .route-tick::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 2px;
      bottom: 2px;
      width: 4px;
      border-radius: 999px;
      transform: translateX(-50%);
      box-shadow: 0 0 0 2px rgba(255, 253, 248, 0.9);
    }
    .route-tick.draft::before {
      background: #2dd4bf;
    }
    .route-tick.vlm::before {
      background: #60a5fa;
    }
    .route-tick.full::before {
      background: #f87171;
    }
    .route-tick.active::before {
      width: 6px;
      top: 0;
      bottom: 0;
      box-shadow: 0 0 0 3px rgba(255, 253, 248, 0.96);
    }
    .timeline-playhead {
      position: absolute;
      top: 1px;
      bottom: 1px;
      width: 2px;
      background: rgba(24, 33, 43, 0.62);
      transform: translateX(-50%);
      pointer-events: none;
    }
    .timeline-thumb {
      position: absolute;
      top: 50%;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: #c76e2b;
      box-shadow: 0 0 0 3px rgba(255, 253, 248, 0.95);
      transform: translate(-50%, -50%);
      pointer-events: none;
    }
    .router-legend {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .router-legend.compact {
      gap: 6px 10px;
    }
    .legend-label {
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8a5c36;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--ink);
      font-weight: 600;
      white-space: nowrap;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.46);
    }
    .legend-swatch.draft {
      background: #2dd4bf;
    }
    .legend-swatch.vlm {
      background: #60a5fa;
    }
    .legend-swatch.full {
      background: #f87171;
    }
    .frame-controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(88px, 108px));
      gap: 8px;
      align-items: center;
      justify-content: center;
      margin-top: 12px;
    }
    .frame-controls button {
      width: 100%;
      min-width: 0;
      padding: 11px 12px;
    }
    @media (max-width: 1320px) {
      .viewer-shell {
        grid-template-columns: 284px minmax(0, 1fr) 284px;
      }
    }
    @media (max-width: 1180px) {
      body {
        height: auto;
        overflow: auto;
      }
      .viewer-shell {
        grid-template-columns: 1fr;
        min-height: auto;
      }
      .sidebar, .stats-sidebar {
        position: static;
        max-height: none;
      }
      .content {
        height: auto;
      }
      .hero-grid {
        grid-template-columns: 1fr;
        height: auto;
      }
      .stage {
        min-height: 500px;
      }
    }
    @media (max-width: 760px) {
      header {
        padding: 20px 16px 10px;
      }
      .viewer-shell {
        padding: 0 12px 18px;
      }
      .frame-controls {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
      .timeline-readout {
        flex-direction: column;
        align-items: stretch;
      }
      .timeline-readout .router-legend {
        justify-content: flex-start;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1 class="title">Spec Trace Viewer</h1>
  </header>
  <main class="viewer-shell">
    <aside class="panel sidebar">
      <h3>Navigator</h3>
      <div class="field">
        <label for="datasetSelect">Dataset</label>
        <select id="datasetSelect"></select>
      </div>
      <div class="field">
        <label for="taskSelect">Task</label>
        <select id="taskSelect"></select>
      </div>
      <div class="field">
        <label for="episodeSelect">Episode</label>
        <select id="episodeSelect"></select>
      </div>
      <button id="loadBtn" type="button">Load Episode</button>
      <div class="field" style="margin-top: 14px;">
        <label for="segmentStart">Segment Start Frame</label>
        <input id="segmentStart" type="number" min="0" value="0">
      </div>
      <div class="field">
        <label for="segmentEnd">Segment End Frame</label>
        <input id="segmentEnd" type="number" min="0" value="0">
      </div>
      <button id="applySegmentBtn" type="button">Apply Segment</button>
      <div class="hint" id="datasetMeta">No dataset loaded.</div>
      <div class="hint">Keyboard: <b>Left</b>/<b>Right</b> step one frame, <b>Space</b> plays or pauses, <b>Home</b>/<b>End</b> jump to start or end.</div>
    </aside>

    <section class="content">
      <section class="panel hero">
        <div class="hero-head">
          <div>
            <h2 class="stage-title">Episode Playback</h2>
            <div class="stage-subtitle" id="episodeMeta">Select a dataset, task, and episode to begin.</div>
          </div>
        </div>
        <div class="hero-grid">
          <section class="stage">
            <div class="frame-shell">
              <img id="frameImage" alt="" aria-hidden="true" hidden>
              <div class="frame-empty" id="frameEmpty">Load an episode to view exact cached frames.</div>
            </div>

            <div class="controls-panel">
              <div class="timeline-readout">
                <div class="frame-label" id="frameLabel">No frame loaded.</div>
                <div class="router-legend compact">
                  <span class="legend-item"><span class="legend-swatch draft"></span>Draft</span>
                  <span class="legend-item"><span class="legend-swatch vlm"></span>VLM</span>
                  <span class="legend-item"><span class="legend-swatch full"></span>Full</span>
                </div>
              </div>
              <div class="timeline-shell">
                <div class="timeline-track" id="timelineTrack">
                  <div class="timeline-rail" id="timelineRail">
                    <div class="timeline-route-segment" id="timelineRouteSegment"></div>
                    <div class="router-points" id="routerPoints"></div>
                    <div class="timeline-playhead" id="timelinePlayhead"></div>
                    <div class="timeline-thumb" id="timelineThumb"></div>
                  </div>
                </div>
              </div>
              <div class="frame-controls">
                <button id="prevBtn" type="button">Prev</button>
                <button id="togglePlayBtn" type="button">Play</button>
                <button id="nextBtn" type="button">Next</button>
              </div>
            </div>
          </section>
        </div>
      </section>
    </section>
    <aside class="panel stats-sidebar">
      <div class="stats-sections">
        <section class="info-card">
          <h3>Current Frame</h3>
          <div class="kv-list" id="frameDetails"></div>
        </section>

        <section class="info-card">
          <h3>Runtime Stats</h3>
          <div class="kv-list" id="statsDetails"></div>
        </section>
      </div>
    </aside>
  </main>
  <script>
    const state = {
      datasets: [],
      tasks: [],
      episodes: [],
      trace: [],
      infers: [],
      inferMap: new Map(),
      routeStarts: [],
      fps: 10,
      currentFrame: 0,
      segmentStart: 0,
      segmentEnd: 0,
      episode: null,
      frameSource: null,
      isPlaying: false,
      playTimer: null,
      timelinePointerId: null,
      lastImageUrl: '',
    };

    const datasetSelect = document.getElementById('datasetSelect');
    const taskSelect = document.getElementById('taskSelect');
    const episodeSelect = document.getElementById('episodeSelect');
    const loadBtn = document.getElementById('loadBtn');
    const applySegmentBtn = document.getElementById('applySegmentBtn');
    const frameImage = document.getElementById('frameImage');
    const frameEmpty = document.getElementById('frameEmpty');
    const frameLabel = document.getElementById('frameLabel');
    const frameDetails = document.getElementById('frameDetails');
    const statsDetails = document.getElementById('statsDetails');
    const routerPoints = document.getElementById('routerPoints');
    const datasetMeta = document.getElementById('datasetMeta');
    const episodeMeta = document.getElementById('episodeMeta');
    const segmentStart = document.getElementById('segmentStart');
    const segmentEnd = document.getElementById('segmentEnd');
    const timelineTrack = document.getElementById('timelineTrack');
    const timelineRail = document.getElementById('timelineRail');
    const timelineRouteSegment = document.getElementById('timelineRouteSegment');
    const timelinePlayhead = document.getElementById('timelinePlayhead');
    const timelineThumb = document.getElementById('timelineThumb');
    const togglePlayBtn = document.getElementById('togglePlayBtn');
    const ACTION_DIM = 7;

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return Number(value).toFixed(digits);
    }

    function routeLabel(route) {
      if (!route) return 'n/a';
      return String(route).toUpperCase();
    }

    function metaField(label, value, valueClass = '') {
      const classSuffix = valueClass ? ` ${valueClass}` : '';
      return `<span class="frame-meta-label">${label}</span> <span class="frame-meta-value${classSuffix}">${value}</span>`;
    }

    function kvItem(label, value) {
      return `<div class="kv-item"><div class="kv-label">${label}</div><div class="kv-value">${value}</div></div>`;
    }

    function actionItem(label, values) {
      const actionValues = Array.from({ length: ACTION_DIM }, (_, idx) => {
        if (Array.isArray(values) && idx < values.length && values[idx] !== null && values[idx] !== undefined) {
          return String(values[idx]);
        }
        return 'n/a';
      });
      const renderedValues = actionValues
        .map((value, idx) => {
          const className = value === 'n/a' ? 'action-line action-line-placeholder' : 'action-line';
          return `<div class="${className}">a${idx}: ${value}</div>`;
        })
        .join('');
      return `
        <div class="kv-item action-item">
          <div class="kv-label">${label}</div>
          <div class="action-list">${renderedValues}</div>
        </div>
      `;
    }

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(await res.text());
      }
      return await res.json();
    }

    function currentTrace() {
      return state.trace[state.currentFrame] || null;
    }

    function totalFrames() {
      return Math.max(0, state.trace.length);
    }

    function maxFrameIndex() {
      return Math.max(0, totalFrames() - 1);
    }

    function resolvedImageFrame(frame = state.currentFrame) {
      const available = Number(state.frameSource && state.frameSource.frame_count ? state.frameSource.frame_count : 0);
      if (available <= 0) return null;
      return clamp(Number(frame) || 0, 0, available - 1);
    }

    function frameUrl(frame) {
      if (!state.frameSource || !state.frameSource.url_template) return '';
      const resolved = resolvedImageFrame(frame);
      if (resolved === null) return '';
      const padded = String(resolved).padStart(Number(state.frameSource.pad_width || 6), '0');
      return state.frameSource.url_template.replace('__FRAME__', padded);
    }

    function prefetchNearbyFrames(frame) {
      if (!state.frameSource) return;
      const neighbors = [frame + 1, frame - 1];
      neighbors.forEach((candidate) => {
        const resolved = resolvedImageFrame(candidate);
        if (resolved === null) return;
        const img = new Image();
        img.src = frameUrl(resolved);
      });
    }

    function routeContextForFrame(frame = state.currentFrame) {
      if (state.routeStarts.length === 0) return null;
      let activeIndex = 0;
      for (let idx = 0; idx < state.routeStarts.length; idx += 1) {
        if (state.routeStarts[idx].frame <= frame) {
          activeIndex = idx;
        } else {
          break;
        }
      }
      const current = state.routeStarts[activeIndex];
      const next = state.routeStarts[activeIndex + 1] || null;
      return {
        frame: current.frame,
        route_type: current.route_type,
        infer_id: current.infer_id,
        endFrameExclusive: next ? next.frame : totalFrames(),
      };
    }

    function computeSegmentSummary() {
      const subset = state.trace.filter((rec) => rec.frame_idx >= state.segmentStart && rec.frame_idx <= state.segmentEnd);
      const actionSubset = subset.filter((rec) => rec.executed_action !== null);
      const inferIds = [...new Set(subset.map((rec) => Number(rec.infer_id)).filter((v) => Number.isFinite(v)))];
      const inferSubset = inferIds.map((id) => state.inferMap.get(id)).filter(Boolean);
      const routeCountsInfer = { draft: 0, vlm: 0, full: 0 };
      const routeCountsAction = { draft: 0, vlm: 0, full: 0 };
      inferSubset.forEach((rec) => { routeCountsInfer[rec.route_type] = (routeCountsInfer[rec.route_type] || 0) + 1; });
      actionSubset.forEach((rec) => { routeCountsAction[rec.route_type] = (routeCountsAction[rec.route_type] || 0) + 1; });
      const inferLatencySum = inferSubset.reduce((acc, rec) => acc + Number(rec.sample_actions_ms || 0), 0);
      const acceptedMean = inferSubset.length > 0
        ? inferSubset.reduce((acc, rec) => acc + Number(rec.accepted_prefix_len || 0), 0) / inferSubset.length
        : null;
      return {
        avgInferLatencyMs: inferSubset.length > 0 ? inferLatencySum / inferSubset.length : null,
        avgActionLatencyMs: actionSubset.length > 0 ? inferLatencySum / actionSubset.length : null,
        acceptedMean,
        draftInferRatio: inferSubset.length > 0 ? (routeCountsInfer.draft / inferSubset.length) * 100 : 0,
        vlmInferRatio: inferSubset.length > 0 ? (routeCountsInfer.vlm / inferSubset.length) * 100 : 0,
        fullInferRatio: inferSubset.length > 0 ? (routeCountsInfer.full / inferSubset.length) * 100 : 0,
        draftActionRatio: actionSubset.length > 0 ? (routeCountsAction.draft / actionSubset.length) * 100 : 0,
        vlmActionRatio: actionSubset.length > 0 ? (routeCountsAction.vlm / actionSubset.length) * 100 : 0,
        fullActionRatio: actionSubset.length > 0 ? (routeCountsAction.full / actionSubset.length) * 100 : 0,
      };
    }

    function renderFramePanel() {
      const trace = currentTrace();
      if (!trace) {
        frameDetails.innerHTML = [
          kvItem('Infer ID', 'n/a'),
          kvItem('Infer Start', 'n/a'),
          kvItem('Action Offset', 'n/a'),
          kvItem('Accepted Prefix', 'n/a'),
          kvItem('Image Frame', 'n/a'),
          actionItem('Executed Action', []),
        ].join('');
        return;
      }
      const resolved = resolvedImageFrame();
      frameDetails.innerHTML = [
        kvItem('Infer ID', trace.infer_id),
        kvItem('Infer Start', String(Boolean(trace.infer_start_frame))),
        kvItem('Action Offset', trace.action_offset_in_chunk ?? 'n/a'),
        kvItem('Accepted Prefix', trace.accepted_prefix_len ?? 'n/a'),
        kvItem('Image Frame', resolved === null ? 'n/a' : resolved),
        actionItem('Executed Action', trace.executed_action),
      ].join('');
    }

    function renderStatsPanel() {
      const summary = computeSegmentSummary();
      if (!state.episode) {
        statsDetails.innerHTML = [
          kvItem('Avg Infer Latency', 'n/a'),
          kvItem('Avg Action Latency', 'n/a'),
          kvItem('Accepted Mean', 'n/a'),
          kvItem('Draft Ratio', 'n/a'),
          kvItem('VLM Ratio', 'n/a'),
          kvItem('Full Ratio', 'n/a'),
          kvItem('Draft Action Ratio', 'n/a'),
          kvItem('VLM Action Ratio', 'n/a'),
          kvItem('Full Action Ratio', 'n/a'),
        ].join('');
        return;
      }
      statsDetails.innerHTML = [
        kvItem('Avg Infer Latency', `${fmt(summary.avgInferLatencyMs)} ms`),
        kvItem('Avg Action Latency', `${fmt(summary.avgActionLatencyMs)} ms`),
        kvItem('Accepted Mean', fmt(summary.acceptedMean)),
        kvItem('Draft Ratio', `${fmt(summary.draftInferRatio, 1)}%`),
        kvItem('VLM Ratio', `${fmt(summary.vlmInferRatio, 1)}%`),
        kvItem('Full Ratio', `${fmt(summary.fullInferRatio, 1)}%`),
        kvItem('Draft Action Ratio', `${fmt(summary.draftActionRatio, 1)}%`),
        kvItem('VLM Action Ratio', `${fmt(summary.vlmActionRatio, 1)}%`),
        kvItem('Full Action Ratio', `${fmt(summary.fullActionRatio, 1)}%`),
      ].join('');
    }

    function renderFrameMeta() {
      const trace = currentTrace();
      if (!trace) {
        frameLabel.textContent = 'No frame loaded.';
        return;
      }
      const inferRecord = state.inferMap.get(Number(trace.infer_id)) || null;
      const routeContext = routeContextForFrame();
      const routeStart = routeContext ? routeContext.frame : trace.frame_idx;
      const routeEnd = routeContext ? Math.max(routeContext.frame, routeContext.endFrameExclusive - 1) : trace.frame_idx;
      const routeType = routeContext ? routeContext.route_type : trace.route_type;
      const acceptedAction = inferRecord?.accepted_prefix_len ?? trace.accepted_prefix_len ?? 'n/a';
      const latencyMs = inferRecord?.sample_actions_ms ?? 'n/a';
      const latencyLabel = latencyMs === 'n/a' ? 'n/a' : `${fmt(latencyMs, 1)} ms`;
      frameLabel.innerHTML = [
        metaField('Frame', `${trace.frame_idx} / ${maxFrameIndex()}`),
        metaField('Infer', trace.infer_id),
        metaField('Route', routeLabel(routeType), routeType ? `frame-meta-route ${routeType}` : 'frame-meta-route'),
        `<span class="frame-meta-label">From</span> <span class="frame-meta-value">${routeStart}</span> <span class="frame-meta-label">to</span> <span class="frame-meta-value">${routeEnd}</span>`,
        metaField('Accepted', acceptedAction),
        metaField('Latency', latencyLabel),
      ].join(' | ');
    }

    function renderFrameImage() {
      const url = frameUrl(state.currentFrame);
      if (!url) {
        frameImage.hidden = true;
        frameEmpty.hidden = false;
        frameEmpty.textContent = state.episode ? 'No cached frame image is available for this episode.' : 'Load an episode to view exact cached frames.';
        frameImage.removeAttribute('src');
        state.lastImageUrl = '';
        return;
      }
      frameEmpty.hidden = true;
      frameImage.hidden = false;
      if (state.lastImageUrl !== url) {
        frameImage.src = url;
        state.lastImageUrl = url;
        prefetchNearbyFrames(state.currentFrame);
      }
    }

    function renderTimeline() {
      if (state.trace.length === 0) {
        routerPoints.innerHTML = '';
        timelinePlayhead.style.left = '0px';
        timelineThumb.style.left = '0px';
        timelineRouteSegment.className = 'timeline-route-segment';
        timelineRouteSegment.style.left = '0%';
        timelineRouteSegment.style.width = '0%';
        return;
      }

      const total = Math.max(1, totalFrames());
      const activeRoute = routeContextForFrame();
      routerPoints.innerHTML = state.routeStarts
        .map((rec) => {
          const left = total === 1 ? 0 : (rec.frame / Math.max(1, total - 1)) * 100;
          const active = activeRoute && activeRoute.frame === rec.frame ? ' active' : '';
          return `<span class="route-tick ${rec.route_type}${active}" style="left:${left}%" data-frame="${rec.frame}" title="${rec.route_type} @ frame ${rec.frame}"></span>`;
        })
        .join('');

      const playheadLeft = total === 1 ? 0 : (state.currentFrame / Math.max(1, total - 1)) * 100;
      timelinePlayhead.style.left = `${playheadLeft}%`;
      timelineThumb.style.left = `${playheadLeft}%`;

      if (!activeRoute) {
        timelineRouteSegment.className = 'timeline-route-segment';
        timelineRouteSegment.style.left = '0%';
        timelineRouteSegment.style.width = '0%';
        return;
      }

      const start = total === 1 ? 0 : (activeRoute.frame / Math.max(1, total - 1)) * 100;
      const endFrame = Math.max(activeRoute.frame, activeRoute.endFrameExclusive - 1);
      const end = total === 1 ? 100 : (endFrame / Math.max(1, total - 1)) * 100;
      timelineRouteSegment.className = `timeline-route-segment ${activeRoute.route_type} visible`;
      timelineRouteSegment.style.left = `${start}%`;
      timelineRouteSegment.style.width = `${Math.max(0.8, end - start)}%`;
    }

    function renderAll() {
      renderFrameMeta();
      renderFramePanel();
      renderStatsPanel();
      renderTimeline();
      renderFrameImage();
      togglePlayBtn.disabled = state.trace.length === 0;
      togglePlayBtn.textContent = state.isPlaying ? 'Pause' : 'Play';
    }

    function stopPlayback(render = true) {
      if (state.playTimer !== null) {
        window.clearInterval(state.playTimer);
        state.playTimer = null;
      }
      state.isPlaying = false;
      if (render) renderAll();
    }

    function startPlayback() {
      if (state.trace.length === 0 || state.isPlaying) return;
      const delayMs = Math.max(20, Math.round(1000 / Math.max(1, state.fps)));
      state.isPlaying = true;
      renderAll();
      state.playTimer = window.setInterval(() => {
        if (state.currentFrame >= maxFrameIndex()) {
          stopPlayback();
          return;
        }
        setCurrentFrame(state.currentFrame + 1);
      }, delayMs);
    }

    function setCurrentFrame(frame, options = {}) {
      if (state.trace.length === 0) return;
      const clamped = clamp(Number(frame) || 0, 0, maxFrameIndex());
      if (options.pausePlayback) stopPlayback(false);
      if (clamped === state.currentFrame && !options.forceRender) {
        renderFrameMeta();
        renderFramePanel();
        renderTimeline();
        renderFrameImage();
        togglePlayBtn.textContent = state.isPlaying ? 'Pause' : 'Play';
        return;
      }
      state.currentFrame = clamped;
      renderAll();
    }

    function applySegmentFromInputs() {
      const maxFrame = maxFrameIndex();
      state.segmentStart = clamp(Number(segmentStart.value) || 0, 0, maxFrame);
      state.segmentEnd = clamp(Number(segmentEnd.value) || 0, state.segmentStart, maxFrame);
      segmentStart.value = String(state.segmentStart);
      segmentEnd.value = String(state.segmentEnd);
      renderAll();
    }

    function resetEpisodeState() {
      stopPlayback(false);
      state.trace = [];
      state.infers = [];
      state.inferMap = new Map();
      state.routeStarts = [];
      state.currentFrame = 0;
      state.segmentStart = 0;
      state.segmentEnd = 0;
      state.episode = null;
      state.frameSource = null;
      state.lastImageUrl = '';
      frameImage.hidden = true;
      frameEmpty.hidden = false;
      frameEmpty.textContent = 'Load an episode to view exact cached frames.';
      frameImage.removeAttribute('src');
      segmentStart.value = '0';
      segmentEnd.value = '0';
      episodeMeta.textContent = 'Select a dataset, task, and episode to begin.';
      renderAll();
    }

    async function loadDatasets() {
      state.datasets = await fetchJson('/api/datasets');
      datasetSelect.innerHTML = state.datasets
        .map((dataset) => `<option value="${dataset.dataset_id}">${dataset.display_name}</option>`)
        .join('');
      if (state.datasets.length === 0) {
        datasetMeta.textContent = 'No datasets found under the current data_root.';
        resetEpisodeState();
        return;
      }
      await loadTasks();
    }

    async function loadTasks() {
      const datasetId = datasetSelect.value;
      const dataset = state.datasets.find((item) => item.dataset_id === datasetId) || null;
      state.tasks = await fetchJson(`/api/tasks?dataset=${encodeURIComponent(datasetId)}`);
      taskSelect.innerHTML = state.tasks
        .map((task) => `<option value="${task.task_name}">${task.task_name}${task.task_id !== null ? ` (id ${task.task_id})` : ''}</option>`)
        .join('');
      datasetMeta.textContent = dataset
        ? `dataset=${dataset.display_name} tasks=${dataset.task_count} episodes=${dataset.episode_count}`
        : 'Dataset loaded.';
      if (state.tasks.length === 0) {
        state.episodes = [];
        episodeSelect.innerHTML = '';
        resetEpisodeState();
        return;
      }
      await loadEpisodes();
    }

    async function loadEpisodes() {
      const datasetId = datasetSelect.value;
      const taskName = taskSelect.value;
      state.episodes = await fetchJson(`/api/episodes?dataset=${encodeURIComponent(datasetId)}&task=${encodeURIComponent(taskName)}`);
      episodeSelect.innerHTML = state.episodes
        .map((ep) => {
          const episodeRef = ep.episode_key || String(ep.episode_idx);
          const episodeLabel = ep.episode_key || `ep ${ep.episode_idx}`;
          return `<option value="${episodeRef}">${episodeLabel} ${ep.success ? 'success' : 'failure'}</option>`;
        })
        .join('');
      resetEpisodeState();
    }

    async function loadEpisode() {
      const datasetId = datasetSelect.value;
      const taskName = taskSelect.value;
      const episodeRef = episodeSelect.value;
      loadBtn.disabled = true;
      stopPlayback(false);
      episodeMeta.textContent = `Loading episode ${episodeRef} and preparing cached frames...`;
      try {
        const episodeQuery = /^\d+$/.test(episodeRef)
          ? `episode_idx=${encodeURIComponent(episodeRef)}`
          : `episode_key=${encodeURIComponent(episodeRef)}`;
        const data = await fetchJson(`/api/episode?dataset=${encodeURIComponent(datasetId)}&task=${encodeURIComponent(taskName)}&${episodeQuery}`);
        state.episode = data.episode;
        state.trace = data.trace;
        state.infers = data.infers;
        state.inferMap = new Map(state.infers.map((rec) => [Number(rec.infer_id), rec]));
        state.routeStarts = state.trace
          .map((rec, idx) => ({ frame: idx, route_type: rec.route_type, infer_id: rec.infer_id, infer_start_frame: Boolean(rec.infer_start_frame) }))
          .filter((rec) => rec.infer_start_frame);
        if (state.routeStarts.length === 0 && state.trace.length > 0) {
          state.routeStarts = [{ frame: 0, route_type: state.trace[0].route_type, infer_id: state.trace[0].infer_id }];
        }
        state.fps = Number(data.fps || 10);
        state.frameSource = data.frame_source || null;
        state.currentFrame = 0;
        state.segmentStart = 0;
        state.segmentEnd = maxFrameIndex();
        state.lastImageUrl = '';
        segmentStart.value = '0';
        segmentEnd.value = String(state.segmentEnd);
        const cacheCount = state.frameSource ? Number(state.frameSource.frame_count || 0) : 0;
        episodeMeta.textContent = `dataset=${datasetId} task="${data.episode.task_description}" success=${data.episode.success} trace_frames=${state.trace.length} cached_frames=${cacheCount} infer_count=${data.episode.infer_record_count}`;
        renderAll();
      } finally {
        loadBtn.disabled = false;
      }
    }

    function isTypingTarget(event) {
      const tag = event.target && event.target.tagName ? event.target.tagName.toLowerCase() : '';
      return tag === 'input' || tag === 'textarea' || tag === 'select' || tag === 'button';
    }

    function frameFromPointerEvent(event) {
      const rect = timelineRail.getBoundingClientRect();
      const x = clamp(event.clientX - rect.left, 0, rect.width);
      const ratio = rect.width <= 0 ? 0 : x / rect.width;
      return Math.round(ratio * maxFrameIndex());
    }

    function updateFrameFromPointer(event) {
      if (state.trace.length === 0) return;
      setCurrentFrame(frameFromPointerEvent(event), { pausePlayback: true });
    }

    datasetSelect.addEventListener('change', loadTasks);
    taskSelect.addEventListener('change', loadEpisodes);
    loadBtn.addEventListener('click', () => {
      loadEpisode().catch((error) => {
        episodeMeta.textContent = error.message;
      });
    });
    applySegmentBtn.addEventListener('click', applySegmentFromInputs);
    document.getElementById('prevBtn').addEventListener('click', () => setCurrentFrame(state.currentFrame - 1, { pausePlayback: true }));
    document.getElementById('nextBtn').addEventListener('click', () => setCurrentFrame(state.currentFrame + 1, { pausePlayback: true }));
    togglePlayBtn.addEventListener('click', () => {
      if (state.isPlaying) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    timelineTrack.addEventListener('pointerdown', (event) => {
      if (state.trace.length === 0) return;
      event.preventDefault();
      stopPlayback(false);
      const routeTick = event.target.closest('.route-tick');
      if (routeTick) {
        setCurrentFrame(Number(routeTick.dataset.frame || 0), { pausePlayback: true });
      } else {
        updateFrameFromPointer(event);
      }
      state.timelinePointerId = event.pointerId;
      timelineTrack.setPointerCapture(event.pointerId);
    });

    timelineTrack.addEventListener('pointermove', (event) => {
      if (state.timelinePointerId !== event.pointerId) return;
      updateFrameFromPointer(event);
    });

    timelineTrack.addEventListener('pointerup', (event) => {
      if (state.timelinePointerId !== event.pointerId) return;
      timelineTrack.releasePointerCapture(event.pointerId);
      state.timelinePointerId = null;
    });

    timelineTrack.addEventListener('pointercancel', (event) => {
      if (state.timelinePointerId !== event.pointerId) return;
      state.timelinePointerId = null;
    });

    window.addEventListener('keydown', (event) => {
      if (isTypingTarget(event) || state.trace.length === 0) return;
      if (event.code === 'ArrowLeft') {
        event.preventDefault();
        setCurrentFrame(state.currentFrame - 1, { pausePlayback: true });
        return;
      }
      if (event.code === 'ArrowRight') {
        event.preventDefault();
        setCurrentFrame(state.currentFrame + 1, { pausePlayback: true });
        return;
      }
      if (event.code === 'Home') {
        event.preventDefault();
        setCurrentFrame(0, { pausePlayback: true });
        return;
      }
      if (event.code === 'End') {
        event.preventDefault();
        setCurrentFrame(maxFrameIndex(), { pausePlayback: true });
        return;
      }
      if (event.code === 'Space') {
        event.preventDefault();
        if (state.isPlaying) {
          stopPlayback();
        } else {
          startPlayback();
        }
      }
    });

    renderAll();
    loadDatasets().catch((error) => {
      datasetMeta.textContent = error.message;
    });
  </script>
</body>
</html>
"""


@dataclasses.dataclass
class Args:
    data_root: str = "/path/to/data"
    host: str = "127.0.0.1"
    port: int = 8011


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_task_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists() and (path / "episode_log.json").exists()


def _dataset_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    children = sorted([child for child in data_root.iterdir() if child.is_dir()])
    if _is_task_run_dir(data_root):
        return [data_root]
    if any(_is_task_run_dir(child) for child in children):
        return [data_root]
    datasets = []
    for child in children:
        if any(_is_task_run_dir(grand) for grand in child.iterdir() if grand.is_dir()):
            datasets.append(child)
    return datasets


def _task_run_dirs(dataset_dir: Path) -> list[Path]:
    if _is_task_run_dir(dataset_dir):
        return [dataset_dir]
    return sorted([child for child in dataset_dir.iterdir() if _is_task_run_dir(child)])


def _dataset_id(data_root: Path, dataset_dir: Path) -> str:
    return data_root.name if dataset_dir.resolve() == data_root.resolve() else dataset_dir.name


def _list_datasets(data_root: Path) -> list[dict[str, Any]]:
    datasets = []
    for dataset_dir in _dataset_dirs(data_root):
        task_dirs = _task_run_dirs(dataset_dir)
        episode_count = 0
        for task_dir in task_dirs:
            records = _read_json(task_dir / "episode_log.json")
            if isinstance(records, list):
                episode_count += len(records)
        dataset_id = _dataset_id(data_root, dataset_dir)
        datasets.append(
            {
                "dataset_id": str(dataset_id),
                "display_name": str(dataset_dir.name),
                "task_count": int(len(task_dirs)),
                "episode_count": int(episode_count),
            }
        )
    return datasets


def _safe_dataset_dir(data_root: Path, dataset_id: str) -> Path:
    for dataset_dir in _dataset_dirs(data_root):
        if _dataset_id(data_root, dataset_dir) == str(dataset_id):
            return dataset_dir
    raise FileNotFoundError(dataset_id)


def _list_tasks(dataset_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for task_dir in _task_run_dirs(dataset_dir):
        records = _read_json(task_dir / "episode_log.json")
        first = records[0] if isinstance(records, list) and records else {}
        task_id = first.get("task_id", None)
        task_description = first.get("task_description", None)
        tasks.append(
            {
                "task_name": str(task_dir.name),
                "task_id": None if task_id is None else int(task_id),
                "task_description": None if task_description is None else str(task_description),
                "episode_count": int(len(records) if isinstance(records, list) else 0),
            }
        )
    tasks.sort(key=lambda item: (item["task_id"] is None, item["task_id"] if item["task_id"] is not None else 10**9, item["task_name"]))
    return tasks


def _safe_task_dir(dataset_dir: Path, task_name: str) -> Path:
    for task_dir in _task_run_dirs(dataset_dir):
        if task_dir.name == str(task_name):
            return task_dir
    raise FileNotFoundError(task_name)


def _episode_key(record: dict[str, Any]) -> str | None:
    try:
        task_id = int(record["task_id"])
        episode_idx = int(record["episode_idx"])
    except (KeyError, TypeError, ValueError):
        return None
    return f"task{task_id:02d}_ep{episode_idx:03d}"


def _episodes_payload(task_dir: Path) -> list[dict[str, Any]]:
    records = _read_json(task_dir / "episode_log.json")
    if not isinstance(records, list):
        return []
    payload = []
    for record in records:
        if not isinstance(record, dict):
            continue
        enriched = dict(record)
        episode_key = _episode_key(record)
        if episode_key is not None:
            enriched["episode_key"] = episode_key
        payload.append(enriched)
    return payload


def _episode_record(task_dir: Path, episode_idx: int | None = None, episode_key: str | None = None) -> dict[str, Any]:
    records = _read_json(task_dir / "episode_log.json")
    if episode_key:
        for record in records:
            if _episode_key(record) == str(episode_key):
                return record
    for record in records:
        if episode_idx is not None and int(record.get("episode_idx", -1)) == int(episode_idx):
            return record
    if episode_key:
        raise FileNotFoundError(f"episode_key={episode_key}")
    raise FileNotFoundError(f"episode_idx={episode_idx}")


def _safe_rel_file(root_dir: Path, rel_path: str) -> Path:
    path = (root_dir / rel_path).resolve()
    if not str(path).startswith(str(root_dir.resolve())):
        raise FileNotFoundError(rel_path)
    return path


def _frame_cache_dir(video_path: Path) -> Path:
    return video_path.parent / f"{video_path.stem}.frames"


def _frame_manifest_path(frame_dir: Path) -> Path:
    return frame_dir / "manifest.json"


def _frame_lock(frame_dir: Path) -> threading.Lock:
    key = str(frame_dir.resolve())
    with _FRAME_LOCKS_GUARD:
        if key not in _FRAME_LOCKS:
            _FRAME_LOCKS[key] = threading.Lock()
        return _FRAME_LOCKS[key]


def _video_signature(video_path: Path) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "source_name": video_path.name,
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _load_frame_manifest(frame_dir: Path) -> dict[str, Any] | None:
    path = _frame_manifest_path(frame_dir)
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _cached_frame_file_count(frame_dir: Path) -> int:
    return sum(1 for child in frame_dir.glob("*.jpg") if child.is_file())


def _usable_frame_manifest(frame_dir: Path, video_path: Path) -> dict[str, Any] | None:
    manifest = _load_frame_manifest(frame_dir)
    if not manifest or not manifest.get("complete", False):
        return None
    signature = _video_signature(video_path)
    if manifest.get("source_size") != signature["source_size"] or manifest.get("source_mtime_ns") != signature["source_mtime_ns"]:
        return None
    actual_frame_count = int(manifest.get("actual_frame_count", 0) or 0)
    if actual_frame_count <= 0:
        return None
    if _cached_frame_file_count(frame_dir) < actual_frame_count:
        return None
    return manifest


def _prepare_frame_cache(task_dir: Path, video_rel_path: str, expected_frame_count: int) -> dict[str, Any]:
    video_path = _safe_rel_file(task_dir, video_rel_path)
    frame_dir = _frame_cache_dir(video_path)
    lock = _frame_lock(frame_dir)
    with lock:
        manifest = _usable_frame_manifest(frame_dir, video_path)
        if manifest is None:
            shutil.rmtree(frame_dir, ignore_errors=True)
            frame_dir.mkdir(parents=True, exist_ok=True)
            actual_frame_count = 0
            for idx, frame in enumerate(iio.imiter(video_path)):
                iio.imwrite(frame_dir / f"{idx:0{_FRAME_PAD_WIDTH}d}.jpg", frame)
                actual_frame_count = idx + 1
            if actual_frame_count <= 0:
                raise RuntimeError(f"no frames decoded from {video_path}")
            manifest = {
                **_video_signature(video_path),
                "complete": True,
                "expected_frame_count": int(expected_frame_count),
                "actual_frame_count": int(actual_frame_count),
                "pad_width": int(_FRAME_PAD_WIDTH),
            }
            _write_json(_frame_manifest_path(frame_dir), manifest)

        return {
            **manifest,
            "frame_dir": frame_dir,
            "relative_dir": str(frame_dir.relative_to(task_dir)),
        }


def _make_handler(data_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            data = path.read_bytes()
            mime, _ = mimetypes.guess_type(str(path))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send_html(_HTML)
                    return
                if parsed.path == "/api/datasets":
                    self._send_json(_list_datasets(data_root))
                    return
                if parsed.path == "/api/tasks":
                    dataset_id = query.get("dataset", [""])[0]
                    dataset_dir = _safe_dataset_dir(data_root, dataset_id)
                    self._send_json(_list_tasks(dataset_dir))
                    return
                if parsed.path == "/api/episodes":
                    dataset_id = query.get("dataset", [""])[0]
                    task_name = query.get("task", [""])[0]
                    dataset_dir = _safe_dataset_dir(data_root, dataset_id)
                    task_dir = _safe_task_dir(dataset_dir, task_name)
                    self._send_json(_episodes_payload(task_dir))
                    return
                if parsed.path == "/api/episode":
                    dataset_id = query.get("dataset", [""])[0]
                    task_name = query.get("task", [""])[0]
                    episode_key = query.get("episode_key", [""])[0] or None
                    episode_idx = None
                    if episode_key is None:
                        episode_idx = int(query.get("episode_idx", ["0"])[0])
                    dataset_dir = _safe_dataset_dir(data_root, dataset_id)
                    task_dir = _safe_task_dir(dataset_dir, task_name)
                    episode = _episode_record(task_dir, episode_idx=episode_idx, episode_key=episode_key)
                    trace = _read_jsonl(_safe_rel_file(task_dir, str(episode["trace_path"])))
                    infers = _read_jsonl(_safe_rel_file(task_dir, str(episode["infer_path"])))
                    frame_cache = _prepare_frame_cache(
                        task_dir,
                        video_rel_path=str(episode["video_path"]),
                        expected_frame_count=max(int(episode.get("frame_count", 0) or 0), len(trace)),
                    )
                    frame_url_template = f"/files?dataset={quote(dataset_id)}&task={quote(task_name)}&path={quote(frame_cache['relative_dir'] + '/__FRAME__.jpg')}"
                    video_url = f"/files?dataset={quote(dataset_id)}&task={quote(task_name)}&path={quote(str(episode['video_path']))}"
                    self._send_json(
                        {
                            "episode": episode,
                            "trace": trace,
                            "infers": infers,
                            "fps": 10,
                            "video_url": video_url,
                            "frame_source": {
                                "url_template": frame_url_template,
                                "frame_count": int(frame_cache["actual_frame_count"]),
                                "pad_width": int(frame_cache["pad_width"]),
                                "expected_frame_count": int(frame_cache["expected_frame_count"]),
                                "cache_dir": str(frame_cache["relative_dir"]),
                            },
                        }
                    )
                    return
                if parsed.path == "/files":
                    dataset_id = query.get("dataset", [""])[0]
                    task_name = query.get("task", [""])[0]
                    rel_path = query.get("path", [""])[0]
                    dataset_dir = _safe_dataset_dir(data_root, dataset_id)
                    task_dir = _safe_task_dir(dataset_dir, task_name)
                    self._send_file(_safe_rel_file(task_dir, rel_path))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except FileNotFoundError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except Exception as exc:  # pragma: no cover
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main(args: Args) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, int(args.port)), _make_handler(data_root))
    print(f"Spec trace viewer at http://{args.host}:{int(args.port)} root={data_root}")
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
