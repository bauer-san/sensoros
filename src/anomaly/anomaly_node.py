"""
Live anomaly scoring node.
Subscribes to scene state, scores each entity,
publishes alerts to Redis and ZeroMQ.
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np

sys.path.insert(0, '/app')
from shared.bus import Subscriber, Publisher
from anomaly.features import entity_to_feature_vector, FEATURE_DIM

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("anomaly")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
MODEL_DIR      = "/app/models/anomaly"

# Alert suppression — entity must exceed threshold for N
# consecutive windows before alert is issued
ACCUMULATION_THRESHOLD = 3

def load_anomaly_config(zone_config_file: str = "/app/configs/zone_config.json") -> dict:
    """
    Load anomaly scoring thresholds from zone_config.json.
    Falls back to sensible defaults if not present.
    """
    defaults = {
        "dwell_alert_seconds": 120.0,
        "min_moving_velocity": 0.3,
        "rule_score_weight":   0.3
    }

    try:
        with open(zone_config_file) as f:
            data = json.load(f)
        # Read from optional anomaly_config section
        cfg = data.get("anomaly_config", {})
        return {**defaults, **cfg}
    except Exception as e:
        logger.warning(f"Could not load anomaly config: {e} — using defaults")
        return defaults

# ── Model loader ──────────────────────────────────────────────────────────────
def load_models() -> dict | None:
    import torch
    from anomaly.train import LSTMAutoencoder

    lstm_path = f"{MODEL_DIR}/lstm_weights.pt"
    lstm_meta = f"{MODEL_DIR}/lstm_meta.pkl"

    if not os.path.exists(lstm_path):
        logger.warning(
            "No LSTM model found — running rules-only mode. "
            "Run train.py to enable LSTM scoring."
        )
        return {"iforest": None, "lstm": None}

    logger.info("Loading LSTM model...")
    with open(lstm_meta, "rb") as f:
        meta = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMAutoencoder(FEATURE_DIM).to(device)
    model.load_state_dict(
        torch.load(lstm_path, map_location=device)
    )
    model.eval()
    meta["model"]  = model
    meta["device"] = device

    logger.info(
        f"LSTM loaded — threshold={meta['threshold']:.4f} "
        f"device={device}"
    )
    return {"iforest": None, "lstm": meta}

# ── Scorer ────────────────────────────────────────────────────────────────────

class AnomalyScorer:
    def __init__(self, models: dict, cfg: dict):
        self.iforest = models["iforest"]
        self.lstm    = models.get("lstm")
        self.cfg     = cfg        

        # Per-entity state
        self.sequence_buffer: dict[str, list] = {}
        self.alert_accumulator: dict[str, int] = {}
        self.active_alerts: set = set()

    # Configuration — derive from zone config, not hardcoded
    DWELL_ALERT_SECONDS  = 120.0   # flag if in zone longer than this
    MIN_MOVING_VELOCITY  = 0.3     # m/s — below this = stopped
    RULE_SCORE_WEIGHT    = 0.3     # blend: 30% rules, 70% LSTM

    def score_entity(
        self,
        entity: dict,
        timestamp: str
    ) -> dict:
        entity_id  = entity["id"]
        vec        = entity_to_feature_vector(entity, timestamp)
        if vec is None:
            return None

        # ── Rule-based score (always available, no training needed) ───────────
        dwell_time = float(entity.get("dwell_time_seconds", 0.0))
        velocity   = float(entity.get("velocity_ms", 0.0))
        zone       = entity.get("zone")
        alert_lvl  = zone.get("alert_level") if zone else None

        rule_score = 0.0

        dwell_threshold  = self.cfg["dwell_alert_seconds"]
        min_velocity     = self.cfg["min_moving_velocity"]
        rule_weight      = self.cfg["rule_score_weight"]

        # Rule 1 — loitering: in zone longer than threshold
        if dwell_time > dwell_threshold:
            rule_score = min(1.0,
                rule_score + (dwell_time - dwell_threshold)
                / dwell_threshold
            )

        # Rule 2 — stopped in monitored zone
        if zone and velocity < min_velocity and dwell_time > 10.0:
            rule_score = min(1.0, rule_score + 0.5)

        # Rule 3 — critical zone entry (immediate flag)
        if alert_lvl == "critical":
            rule_score = 1.0

        # ── LSTM sequence score ───────────────────────────────────────────────
        lstm_score = 0.0

        if self.lstm:
            import torch
            lstm   = self.lstm
            window = lstm["window_size"]

            if entity_id not in self.sequence_buffer:
                self.sequence_buffer[entity_id] = []

            vec_scaled = lstm["scaler"].transform(
                vec.reshape(1, -1)
            )[0]
            self.sequence_buffer[entity_id].append(vec_scaled)

            if len(self.sequence_buffer[entity_id]) >= window:
                seq = np.array(
                    self.sequence_buffer[entity_id][-window:]
                )
                seq_tensor = torch.FloatTensor(seq)\
                    .unsqueeze(0).to(lstm["device"])
                error      = float(
                    lstm["model"].reconstruction_error(
                        seq_tensor
                    ).item()
                )
                threshold  = lstm["threshold"]
                lstm_score = float(
                    np.clip(error / (threshold + 1e-6), 0, 2)
                )
                self.sequence_buffer[entity_id] = \
                    self.sequence_buffer[entity_id][-window:]

        # ── Fused score ───────────────────────────────────────────────────────
        if self.lstm and lstm_score > 0:
            # LSTM available — blend with rules
            fused = (
                (1.0 - rule_weight) * lstm_score
                + rule_weight * rule_score
            )
        else:
            # No LSTM sequence yet — rules only
            fused = rule_score

        # ── Alert accumulation ────────────────────────────────────────────────
        is_anomalous = fused > 1.0
        if is_anomalous:
            self.alert_accumulator[entity_id] = \
                self.alert_accumulator.get(entity_id, 0) + 1
        else:
            self.alert_accumulator[entity_id] = 0
            self.active_alerts.discard(entity_id)

        accumulated = self.alert_accumulator.get(entity_id, 0)
        emit_alert  = (
            accumulated >= ACCUMULATION_THRESHOLD
            and entity_id not in self.active_alerts
        )

        if emit_alert:
            self.active_alerts.add(entity_id)

        return {
            "entity_id":    entity_id,
            "rule_score":   round(rule_score,  3),
            "lstm_score":   round(lstm_score,  3),
            "fused_score":  round(fused,       3),
            "is_anomalous": is_anomalous,
            "accumulated":  accumulated,
            "emit_alert":   emit_alert,
            "zone":         zone,
            "position_2d":  entity.get("position_2d"),
            "dwell_time":   dwell_time,
            "velocity":     velocity
        }

    def prune(self, active_ids: set):
        for eid in list(self.sequence_buffer.keys()):
            if eid not in active_ids:
                del self.sequence_buffer[eid]
        for eid in list(self.alert_accumulator.keys()):
            if eid not in active_ids:
                del self.alert_accumulator[eid]
        self.active_alerts &= active_ids

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import redis

    # Wait for perception to bind
    time.sleep(3)

    r = redis.Redis(
        host=REDIS_HOST,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    r.ping()
    logger.info("Redis connected")

    sub = Subscriber(host="localhost", port=5555, topic="scene_state")
    pub = Publisher(port=5556)

    cfg = load_anomaly_config()
    logger.info(
        f"Anomaly config: dwell_threshold={cfg['dwell_alert_seconds']}s "
        f"min_velocity={cfg['min_moving_velocity']}m/s "
        f"rule_weight={cfg['rule_score_weight']}"
    )

    models = load_models()
    scorer = AnomalyScorer(models, cfg) if models else None

    if scorer:
        logger.info("Anomaly scoring active")
    else:
        logger.info("Observe-only mode — collecting data, not scoring")

    count     = 0
    fps_timer = time.time()

    while True:
        state = sub.receive()
        if not state:
            continue

        timestamp = state.get("timestamp", "")
        entities  = state.get("entities", [])
        active_ids = {e["id"] for e in entities}

        scores  = []
        alerts  = []

        for entity in entities:
            # Always log entity presence for data collection
            if scorer:
                result = scorer.score_entity(entity, timestamp)
                if result:
                    scores.append(result)
                    if result["emit_alert"]:
                        alerts.append(result)

        # Prune stale entities
        if scorer:
            scorer.prune(active_ids)

        # Publish scores
        insight = {
            "timestamp":    timestamp,
            "frame_id":     state.get("frame_id"),
            "entity_count": len(entities),
            "scores":       scores,
            "alerts":       alerts
        }

        pub.publish("anomaly_scores", insight)
        r.set("anomaly:latest", json.dumps(insight))

        # Log alerts immediately
        for alert in alerts:
            zone_label = alert["zone"]["label"] \
                if alert.get("zone") else "unknown zone"
            logger.warning(
                f"🚨 ALERT | {alert['entity_id']} | "
                f"zone={zone_label} | "
                f"score={alert['fused_score']:.3f} | "
                f"dwell={alert['dwell_time']:.0f}s"
            )

            # Persist to alert history for LLM context
            r.lpush("alerts:history", json.dumps({
                "timestamp":  timestamp,
                "entity_id":  alert["entity_id"],
                "zone":       alert.get("zone"),
                "fused_score": alert["fused_score"],
                "dwell_time": alert.get("dwell_time", 0),
                "position_2d": alert.get("position_2d")
            }))
            r.ltrim("alerts:history", 0, 99)   # keep last 100 alerts

        # FPS logging
        count += 1
        now    = time.time()
        if now - fps_timer >= 10.0:
            fps = count / (now - fps_timer)
            logger.info(
                f"{fps:.1f}fps | entities={len(entities)} | "
                f"alerts_active="
                f"{len(scorer.active_alerts) if scorer else 0}"
            )
            count     = 0
            fps_timer = now

if __name__ == "__main__":
    main()