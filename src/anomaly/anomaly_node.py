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

REDIS_HOST     = os.getenv("REDIS_HOST",    "redis")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
MODEL_DIR      = "/app/models/anomaly"

# Alert suppression — entity must exceed threshold for N
# consecutive windows before alert is issued
ACCUMULATION_THRESHOLD = 3

# ── Model loader ──────────────────────────────────────────────────────────────

def load_models() -> dict | None:
    import torch

    iforest_path = f"{MODEL_DIR}/iforest.pkl"
    lstm_path    = f"{MODEL_DIR}/lstm_weights.pt"
    lstm_meta    = f"{MODEL_DIR}/lstm_meta.pkl"

    if not os.path.exists(iforest_path):
        logger.warning(f"No trained models found at {MODEL_DIR} "
                       f"— run train.py first. Running in observe-only mode.")
        return None

    logger.info("Loading anomaly models...")

    with open(iforest_path, "rb") as f:
        iforest_bundle = pickle.load(f)

    lstm_bundle = None
    if os.path.exists(lstm_path) and os.path.exists(lstm_meta):
        from anomaly.train import LSTMAutoencoder  # reuse class definition

        with open(lstm_meta, "rb") as f:
            meta = pickle.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = LSTMAutoencoder(FEATURE_DIM).to(device)
        model.load_state_dict(torch.load(lstm_path, map_location=device))
        model.eval()
        meta["model"]  = model
        meta["device"] = device
        lstm_bundle    = meta

    logger.info("Models loaded")
    return {"iforest": iforest_bundle, "lstm": lstm_bundle}

# ── Scorer ────────────────────────────────────────────────────────────────────

class AnomalyScorer:
    def __init__(self, models: dict):
        self.iforest = models["iforest"]
        self.lstm    = models.get("lstm")

        # Per-entity state
        self.sequence_buffer: dict[str, list] = {}
        self.alert_accumulator: dict[str, int] = {}
        self.active_alerts: set = set()

    def score_entity(
        self,
        entity: dict,
        timestamp: str
    ) -> dict:
        entity_id = entity["id"]
        vec = entity_to_feature_vector(entity, timestamp)
        if vec is None:
            return None

        # ── Isolation Forest score ────────────────────────────────────────────
        iforest    = self.iforest
        vec_scaled = iforest["scaler"].transform(vec.reshape(1, -1))
        raw_score  = -iforest["model"].decision_function(vec_scaled)[0]
        threshold  = iforest["threshold"]
        if_score   = float(np.clip(raw_score / (threshold + 1e-6), 0, 2))

        # ── LSTM score ────────────────────────────────────────────────────────
        lstm_score = 0.0
        if self.lstm:
            import torch
            lstm       = self.lstm
            window     = lstm["window_size"]

            # Maintain sequence buffer per entity
            if entity_id not in self.sequence_buffer:
                self.sequence_buffer[entity_id] = []

            vec_scaled_lstm = lstm["scaler"].transform(
                vec.reshape(1, -1)
            )[0]
            self.sequence_buffer[entity_id].append(vec_scaled_lstm)

            # Only score when we have a full window
            if len(self.sequence_buffer[entity_id]) >= window:
                seq = np.array(
                    self.sequence_buffer[entity_id][-window:]
                )
                seq_tensor = torch.FloatTensor(seq)\
                    .unsqueeze(0).to(lstm["device"])
                error      = float(
                    lstm["model"].reconstruction_error(seq_tensor).item()
                )
                lstm_threshold = lstm["threshold"]
                lstm_score = float(
                    np.clip(error / (lstm_threshold + 1e-6), 0, 2)
                )
                # Trim buffer
                self.sequence_buffer[entity_id] = \
                    self.sequence_buffer[entity_id][-window:]

        # ── Fused score ───────────────────────────────────────────────────────
        # Weight LSTM higher if available — it catches temporal patterns
        if self.lstm and lstm_score > 0:
            fused = 0.4 * if_score + 0.6 * lstm_score
        else:
            fused = if_score

        # ── Alert accumulation ────────────────────────────────────────────────
        # Must exceed threshold for N consecutive windows
        is_anomalous = fused > 1.0   # normalized threshold
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
            "if_score":     round(if_score,   3),
            "lstm_score":   round(lstm_score,  3),
            "fused_score":  round(fused,       3),
            "is_anomalous": is_anomalous,
            "accumulated":  accumulated,
            "emit_alert":   emit_alert,
            "zone":         entity.get("zone"),
            "position_2d":  entity.get("position_2d"),
            "dwell_time":   entity.get("dwell_time_seconds", 0)
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

    sub = Subscriber(host="perception", port=5555, topic="scene_state")
    pub = Publisher(port=5556)

    models = load_models()
    scorer = AnomalyScorer(models) if models else None

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