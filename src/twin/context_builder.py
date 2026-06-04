"""
Builds LLM prompt context from live scene state and alert history.
Keeps context concise — LLMs perform better with focused context.
"""

import json
import time
import redis
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger("context_builder")

# Maximum alerts to include in context
MAX_ALERT_HISTORY = 20
# Maximum recent scene states to summarize
SCENE_SUMMARY_WINDOW = 30   # last 30 frames = ~3 seconds at 10fps


class ContextBuilder:
    def __init__(self, r: redis.Redis, zone_config_file: str):
        self.r = r

        with open(zone_config_file) as f:
            self.zone_config = json.load(f)

        self.zone_labels = {
            z["id"]: z["label"]
            for z in self.zone_config.get("zones", [])
        }

    def _format_position(self, pos: dict | None) -> str:
        if not pos:
            return "unknown position"
        return f"({pos['x']:.1f}m, {pos['y']:.1f}m from door origin)"

    def _format_entity(self, entity: dict) -> str:
        zone     = entity.get("zone")
        zone_str = zone["label"] if zone else "outside monitored zones"
        pos_str  = self._format_position(entity.get("position_2d"))
        dwell    = entity.get("dwell_time_seconds", 0)
        vel      = entity.get("velocity_ms", 0)
        motion   = "stationary" if vel < 0.3 else f"moving at {vel:.1f}m/s"

        return (
            f"- {entity['id']} ({entity['class']}, "
            f"confidence {entity['confidence']:.0%}): "
            f"{zone_str}, {pos_str}, "
            f"dwell {dwell:.0f}s, {motion}"
        )

    def _get_alert_history(self) -> list[dict]:
        raw = self.r.lrange("alerts:history", 0, MAX_ALERT_HISTORY - 1)
        alerts = []
        for r in raw:
            try:
                alerts.append(json.loads(r))
            except Exception:
                continue
        return alerts

    def _format_alert(self, alert: dict) -> str:
        ts       = alert.get("timestamp", "unknown time")
        entity   = alert.get("entity_id", "unknown")
        zone     = alert.get("zone", {})
        zone_lbl = zone.get("label", "unknown zone") if zone else "unknown zone"
        score    = alert.get("fused_score", 0)
        dwell    = alert.get("dwell_time", 0)
        return (
            f"[{ts}] {entity} in {zone_lbl} "
            f"(score={score:.2f}, dwell={dwell:.0f}s)"
        )

    def build(self) -> str:
        """Build complete context string for LLM prompt"""

        # Current time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Latest scene state
        raw_scene = self.r.get("scene:latest")
        scene     = json.loads(raw_scene) if raw_scene else {}
        entities  = scene.get("entities", [])
        frame_id  = scene.get("frame_id", 0)

        # Latest anomaly state
        raw_anomaly = self.r.get("anomaly:latest")
        anomaly     = json.loads(raw_anomaly) if raw_anomaly else {}
        scores      = anomaly.get("scores", [])
        active_alerts = [s for s in scores if s.get("is_anomalous")]

        # Alert history
        alert_history = self._get_alert_history()

        # Zone definitions
        zones_str = "\n".join(
            f"  - {z['label']} (alert_level={z['alert_level']}): "
            f"polygon covering {z.get('description', 'monitored area')}"
            for z in self.zone_config.get("zones", [])
        )

        # Entity summary
        if entities:
            entities_str = "\n".join(
                self._format_entity(e) for e in entities
            )
        else:
            entities_str = "  No entities currently detected"

        # Active anomalies
        if active_alerts:
            anomaly_str = "\n".join(
                f"  - {a['entity_id']}: score={a['fused_score']:.2f} "
                f"dwell={a.get('dwell_time', 0):.0f}s "
                f"zone={a.get('zone', {}).get('label', 'unknown') if a.get('zone') else 'none'}"
                for a in active_alerts
            )
        else:
            anomaly_str = "  No active anomalies"

        # Alert history
        if alert_history:
            history_str = "\n".join(
                f"  {self._format_alert(a)}"
                for a in alert_history[:10]
            )
        else:
            history_str = "  No alerts recorded yet"

        context = f"""
SENSOROS DIGITAL TWIN — SCENE INTELLIGENCE CONTEXT
Current time: {now_str}
Frame: {frame_id}

MONITORED ZONES:
{zones_str}

COORDINATE SYSTEM:
- Origin: front door edge
- X axis: positive = right across scene
- Y axis: positive = away from camera toward road
- Units: meters

CURRENT SCENE STATE:
{entities_str}

ACTIVE ANOMALY SCORES:
{anomaly_str}

RECENT ALERT HISTORY (most recent first):
{history_str}
""".strip()

        return context