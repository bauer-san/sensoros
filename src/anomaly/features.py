import numpy as np
import json
from typing import Optional

# ── Cyclical time encoding ────────────────────────────────────────────────────

def encode_cyclical(value: float, max_value: float) -> tuple[float, float]:
    """Encode a cyclical value as sin/cos pair so 23:59 and 00:01 are adjacent"""
    angle = 2 * np.pi * value / max_value
    return float(np.sin(angle)), float(np.cos(angle))

# ── Feature vector ────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "hour_sin",           # time of day — cyclical
    "hour_cos",
    "dow_sin",            # day of week — cyclical
    "dow_cos",
    "in_approach_zone",   # binary — entity in monitored zone
    "pos_x",              # world position
    "pos_y",
    "velocity",           # movement speed m/s
    "dwell_time",         # seconds in current zone
    "trajectory_len",     # how many trajectory points accumulated
    "linearity",          # 1.0=straight path, 0.0=erratic/loitering
    "zone_alert_level",   # 0=none, 1=low, 2=medium, 3=high, 4=critical
]

FEATURE_DIM = len(FEATURE_NAMES)

ALERT_LEVELS = {
    None:       0,
    "low":      1,
    "medium":   2,
    "high":     3,
    "critical": 4
}

def trajectory_linearity(trajectory: list) -> float:
    """
    Measure how straight the trajectory is.
    1.0 = perfectly straight (walking through)
    0.0 = random/loitering
    """
    if len(trajectory) < 3:
        return 1.0

    pts  = np.array(trajectory)
    # Vector from first to last point
    total_vec  = pts[-1] - pts[0]
    total_dist = np.linalg.norm(total_vec)

    if total_dist < 0.01:
        return 0.0  # barely moved — loitering

    # Sum of step distances
    step_dists = np.sum(
        np.linalg.norm(np.diff(pts, axis=0), axis=1)
    )

    # Linearity = straight-line distance / total path length
    # High = straight, Low = wandering
    return float(min(total_dist / (step_dists + 1e-6), 1.0))

def entity_to_feature_vector(
    entity: dict,
    timestamp_str: str
) -> Optional[np.ndarray]:
    """
    Convert a single entity from scene state to a feature vector.
    Returns None if entity lacks required fields.
    """
    import datetime

    # Parse timestamp
    try:
        ts = datetime.datetime.fromisoformat(
            timestamp_str.replace("Z", "+00:00")
        )
        hour = ts.hour + ts.minute / 60.0
        dow  = ts.weekday()
    except Exception:
        hour = 12.0
        dow  = 0

    hour_sin, hour_cos = encode_cyclical(hour, 24)
    dow_sin,  dow_cos  = encode_cyclical(dow,  7)

    # Zone
    zone       = entity.get("zone")
    zone_id    = zone.get("id") if zone else None
    alert_lvl  = zone.get("alert_level") if zone else None
    in_zone    = 1.0 if zone_id == "approach_zone" else 0.0
    alert_num  = float(ALERT_LEVELS.get(alert_lvl, 0))

    # Position
    pos        = entity.get("position_2d", {})
    pos_x      = float(pos.get("x", 0.0)) if pos else 0.0
    pos_y      = float(pos.get("y", 0.0)) if pos else 0.0

    # Motion
    velocity   = float(entity.get("velocity_ms",   0.0))
    dwell      = float(entity.get("dwell_time_seconds", 0.0))
    trajectory = entity.get("trajectory", [])
    traj_len   = float(len(trajectory))
    linearity  = trajectory_linearity(trajectory)

    vec = np.array([
        hour_sin, hour_cos,
        dow_sin,  dow_cos,
        in_zone,
        pos_x,    pos_y,
        velocity,
        dwell,
        traj_len,
        linearity,
        alert_num,
    ], dtype=np.float32)

    return vec

def scene_states_to_feature_matrix(
    scene_states: list[dict]
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a list of scene states to a feature matrix.
    Returns (X, entity_ids) where X is (N, FEATURE_DIM).
    Only includes frames with at least one entity.
    """
    vectors    = []
    entity_ids = []

    for state in scene_states:
        ts       = state.get("timestamp", "")
        entities = state.get("entities", [])

        for entity in entities:
            vec = entity_to_feature_vector(entity, ts)
            if vec is not None:
                vectors.append(vec)
                entity_ids.append(entity.get("id", "unknown"))

    if not vectors:
        return np.zeros((0, FEATURE_DIM)), []

    return np.array(vectors, dtype=np.float32), entity_ids