/**
 * PathVisualizerPlugin
 *
 * Tracks and renders cumulative movement history for detected objects:
 *   - Fading path trails per entity (longer than the 10-pt backend window)
 *   - Density heatmap that accumulates world-space positions over time
 *   - Dwell markers at locations where entities stood still
 */

class PathVisualizerPlugin {
  /**
   * @param {object} worldBounds  { minX, maxX, minY, maxY }
   * @param {function} worldToCanvas  (wx, wy) => [cx, cy]
   */
  constructor(worldBounds, worldToCanvas) {
    this._bounds        = worldBounds;
    this._worldToCanvas = worldToCanvas;

    // ── Feature toggles ──
    this.showTrails   = true;
    this.showHeatmap  = false;
    this.showDwell    = false;

    // ── Trail settings ──
    this.maxTrailLength    = 300;   // points per entity
    this.trailFadeSecs     = 20;    // oldest visible trail age in seconds

    // ── Heatmap grid ──
    this.gridCols     = 70;
    this.gridRows     = 70;
    this.heatDecay    = 0.992;      // per-frame decay multiplier (<1 = fades)
    this.heatmapData  = new Float32Array(this.gridCols * this.gridRows);

    // ── Dwell detection ──
    this.dwellThresholdSecs    = 5;   // standing still for this long → marker
    this.dwellDistanceMeters   = 0.4; // max movement to be considered "still"
    this.dwellSpots = [];             // [{x, y, secs, anomaly}]

    // ── Per-entity trail history ──
    // Map<entityId, Array<{x, y, t, anomaly}>>
    this._trails = new Map();

    // ── Off-screen canvas for heatmap blending ──
    this._heatCanvas  = document.createElement("canvas");
    this._heatCtx     = this._heatCanvas.getContext("2d");
    this._heatDirty   = true;

    // ── Off-screen canvas for smooth heatmap blur ──
    this._blurCanvas  = document.createElement("canvas");
    this._blurCtx     = this._blurCanvas.getContext("2d");
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Call once per WebSocket frame to ingest new scene state.
   * @param {object} sceneState   raw scene JSON from the server
   * @param {object} anomalyState raw anomaly JSON from the server
   */
  update(sceneState, anomalyState) {
    const entities = sceneState.entities || [];
    const scores   = anomalyState.scores  || [];
    const scoreMap = {};
    scores.forEach(s => { scoreMap[s.entity_id] = s; });

    const now = performance.now() / 1000;

    entities.forEach(entity => {
      const pos = entity.position_2d;
      if (!pos) return;

      const score     = scoreMap[entity.id];
      const isAnomaly = !!(score && score.is_anomalous);
      const pt        = { x: pos.x, y: pos.y, t: now, anomaly: isAnomaly };

      // ── Trails ──
      if (!this._trails.has(entity.id)) {
        this._trails.set(entity.id, []);
      }
      const trail = this._trails.get(entity.id);
      trail.push(pt);
      if (trail.length > this.maxTrailLength) trail.shift();

      // ── Heatmap accumulation ──
      this._accumulateHeat(pos.x, pos.y);

      // ── Dwell detection ──
      this._detectDwell(entity, isAnomaly, now);
    });

    // Remove trails for entities that have disappeared (keep for fade-out)
    // We intentionally keep dead trails; they fade naturally via age.
  }

  /**
   * Call inside the main render loop, after all other drawing is done.
   * @param {CanvasRenderingContext2D} ctx   main canvas context
   * @param {HTMLCanvasElement}        canvas main canvas element
   */
  render(ctx, canvas) {
    this._resizeOffscreen(canvas);
    this._decayHeat();

    if (this.showHeatmap) this._renderHeatmap(ctx, canvas);
    if (this.showTrails)  this._renderTrails(ctx);
    if (this.showDwell)   this._renderDwellSpots(ctx);
  }

  /** Remove all accumulated history. */
  clear() {
    this._trails.clear();
    this.heatmapData.fill(0);
    this.dwellSpots = [];
  }

  // ── Internal helpers ────────────────────────────────────────────────────────

  _accumulateHeat(wx, wy) {
    const { minX, maxX, minY, maxY } = this._bounds;
    const col = Math.floor(
      ((wx - minX) / (maxX - minX)) * this.gridCols
    );
    const row = Math.floor(
      ((wy - minY) / (maxY - minY)) * this.gridRows
    );
    if (col < 0 || col >= this.gridCols) return;
    if (row < 0 || row >= this.gridRows) return;
    // Splat a small 3×3 kernel to smooth the grid
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const r = row + dr;
        const c = col + dc;
        if (r < 0 || r >= this.gridRows || c < 0 || c >= this.gridCols) continue;
        const weight = (dr === 0 && dc === 0) ? 1.0 : 0.35;
        this.heatmapData[r * this.gridCols + c] += weight;
      }
    }
    this._heatDirty = true;
  }

  _decayHeat() {
    for (let i = 0; i < this.heatmapData.length; i++) {
      this.heatmapData[i] *= this.heatDecay;
    }
  }

  _detectDwell(entity, isAnomaly, now) {
    const pos = entity.position_2d;
    if (!pos || !entity.velocity_ms) return;

    const isStill = entity.velocity_ms < this.dwellDistanceMeters;
    const longDwell = entity.dwell_time_seconds >= this.dwellThresholdSecs;
    if (!isStill || !longDwell) return;

    // Check if this spot is already recorded nearby
    const nearby = this.dwellSpots.find(d => {
      const dx = d.x - pos.x;
      const dy = d.y - pos.y;
      return Math.sqrt(dx * dx + dy * dy) < this.dwellDistanceMeters * 2;
    });

    if (nearby) {
      nearby.secs   = entity.dwell_time_seconds;
      nearby.anomaly = nearby.anomaly || isAnomaly;
    } else {
      this.dwellSpots.push({
        x: pos.x,
        y: pos.y,
        secs: entity.dwell_time_seconds,
        anomaly: isAnomaly,
      });
    }
  }

  _resizeOffscreen(canvas) {
    if (
      this._heatCanvas.width  !== canvas.width ||
      this._heatCanvas.height !== canvas.height
    ) {
      this._heatCanvas.width  = canvas.width;
      this._heatCanvas.height = canvas.height;
      this._blurCanvas.width  = canvas.width;
      this._blurCanvas.height = canvas.height;
      this._heatDirty = true;
    }
  }

  // ── Rendering sub-routines ─────────────────────────────────────────────────

  _renderHeatmap(ctx, canvas) {
    const W = canvas.width;
    const H = canvas.height;

    // Find max value for normalization
    let maxVal = 1e-6;
    for (let i = 0; i < this.heatmapData.length; i++) {
      if (this.heatmapData[i] > maxVal) maxVal = this.heatmapData[i];
    }

    // Paint heatmap cells onto off-screen canvas
    const hCtx     = this._heatCtx;
    const cellW     = W / this.gridCols;
    const cellH     = H / this.gridRows;

    hCtx.clearRect(0, 0, W, H);

    for (let row = 0; row < this.gridRows; row++) {
      for (let col = 0; col < this.gridCols; col++) {
        const v    = this.heatmapData[row * this.gridCols + col];
        if (v < 0.01) continue;

        const norm = Math.min(v / maxVal, 1);

        // Map world grid cell → canvas coords
        // Grid row 0 = world minY, increasing row = increasing Y,
        // but canvas Y is inverted (minY is at bottom of canvas).
        const { minX, maxX, minY, maxY } = this._bounds;
        const wx = minX + (col / this.gridCols) * (maxX - minX);
        const wy = minY + (row / this.gridRows) * (maxY - minY);
        const [cx, cy] = this._worldToCanvas(wx, wy);

        // Cool-warm colormap: blue → cyan → green → yellow → red
        const color = this._heatColor(norm);
        hCtx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${0.12 + norm * 0.5})`;
        hCtx.fillRect(cx - cellW / 2, cy - cellH / 2, cellW + 1, cellH + 1);
      }
    }

    // Composite heatmap canvas with blur onto main canvas
    ctx.save();
    ctx.filter = "blur(8px)";
    ctx.drawImage(this._heatCanvas, 0, 0);
    ctx.filter = "none";
    ctx.restore();
  }

  /**
   * Cool-warm colormap: t=0 → blue, t=0.5 → green/yellow, t=1 → red
   * Returns [r, g, b] in 0-255 range.
   */
  _heatColor(t) {
    // Key stops: blue(0) → cyan(0.25) → green(0.4) → yellow(0.65) → red(1)
    const stops = [
      [0,   [  0,  30, 180]],
      [0.25,[ 20, 160, 220]],
      [0.45,[ 50, 205, 100]],
      [0.65,[220, 200,  20]],
      [1.0, [248,  81,  73]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      const [t0, c0] = stops[i];
      const [t1, c1] = stops[i + 1];
      if (t >= t0 && t <= t1) {
        const f = (t - t0) / (t1 - t0);
        return c0.map((v, j) => Math.round(v + f * (c1[j] - v)));
      }
    }
    return stops[stops.length - 1][1];
  }

  _renderTrails(ctx) {
    const now     = performance.now() / 1000;
    const cutoff  = now - this.trailFadeSecs;

    this._trails.forEach((trail, entityId) => {
      // Filter to visible window
      const visible = trail.filter(p => p.t >= cutoff);
      if (visible.length < 2) return;

      // Draw segments with fading opacity
      for (let i = 1; i < visible.length; i++) {
        const prev = visible[i - 1];
        const curr = visible[i];

        const ageRatio  = (curr.t - cutoff) / this.trailFadeSecs; // 0=old, 1=new
        const alpha     = Math.max(0, ageRatio * 0.85);
        const isAnomaly = curr.anomaly || prev.anomaly;

        const [x0, y0] = this._worldToCanvas(prev.x, prev.y);
        const [x1, y1] = this._worldToCanvas(curr.x, curr.y);

        // Use a gradient along each segment for smooth fade
        const grad = ctx.createLinearGradient(x0, y0, x1, y1);
        const prevAlpha = Math.max(0, ((prev.t - cutoff) / this.trailFadeSecs) * 0.85);
        const baseColor = isAnomaly ? "248,81,73" : "63,185,80";

        grad.addColorStop(0, `rgba(${baseColor},${prevAlpha})`);
        grad.addColorStop(1, `rgba(${baseColor},${alpha})`);

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.strokeStyle = grad;
        ctx.lineWidth   = 2;
        ctx.setLineDash([]);
        ctx.stroke();
      }

      // Draw a small arrowhead at the most recent point
      if (visible.length >= 2) {
        const last = visible[visible.length - 1];
        const prev = visible[visible.length - 2];
        const [tx, ty] = this._worldToCanvas(last.x, last.y);
        const [px, py] = this._worldToCanvas(prev.x, prev.y);
        const angle  = Math.atan2(ty - py, tx - px);
        const color  = last.anomaly ? "#f85149" : "#3fb950";
        const ageRatio = (last.t - cutoff) / this.trailFadeSecs;
        const alpha    = Math.max(0, ageRatio * 0.9);

        ctx.save();
        ctx.translate(tx, ty);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(5, 0);
        ctx.lineTo(-5, -4);
        ctx.lineTo(-5, 4);
        ctx.closePath();
        ctx.fillStyle = `${color}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`;
        ctx.fill();
        ctx.restore();
      }
    });
  }

  _renderDwellSpots(ctx) {
    this.dwellSpots.forEach(spot => {
      const [cx, cy]  = this._worldToCanvas(spot.x, spot.y);
      const radius     = Math.min(6 + spot.secs * 0.3, 20);
      const color      = spot.anomaly ? "#f85149" : "#d29922";

      // Outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = color + "99";
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Inner fill
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = color + "55";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1;
      ctx.stroke();

      // Dwell time label
      ctx.fillStyle = color;
      ctx.font      = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`${spot.secs.toFixed(0)}s`, cx, cy - radius - 3);
    });
  }
}
