import time
import json
import logging
import os
import sys
import cv2
import redis
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

sys.path.insert(0, '/app')
from shared.bus import Publisher

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("perception")

REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PASSWORD  = os.getenv("REDIS_PASSWORD")
RTSP_URL        = os.getenv("CAMERA_RTSP_URL")
SCENE_STATE_FPS = int(os.getenv("SCENE_STATE_FPS", 10))
MODEL_PATH      = os.getenv("MODEL_PATH", "/app/models/yolov8n.engine")

# Classes we care about for security
TRACKED_CLASSES = {0: "person", 2: "car", 3: "motorcycle",
                   7: "truck",  15: "cat", 16: "dog"}

# ── Zone loader ──────────────────────────────────────────────────────────────

def load_zones(zone_file: str) -> list:
    if not os.path.exists(zone_file):
        logger.warning(f"No zone file found at {zone_file} — zone assignment disabled")
        return []
    with open(zone_file) as f:
        data = json.load(f)
    zones = []
    for z in data.get("zones", []):
        zones.append({
            "id":          z["id"],
            "label":       z["label"],
            "alert_level": z["alert_level"],
            "polygon":     Polygon(z["polygon"])
        })
    # Sort by area — assign to most specific (smallest) containing zone
    zones.sort(key=lambda z: z["polygon"].area)
    logger.info(f"Loaded {len(zones)} zones")
    return zones

def assign_zone(zones: list, world_x: float, world_y: float) -> dict | None:
    pt = Point(world_x, world_y)
    for zone in zones:
        if zone["polygon"].contains(pt):
            return {
                "id":          zone["id"],
                "label":       zone["label"],
                "alert_level": zone["alert_level"]
            }
    return None

def compute_scene_bounds(zones: list) -> dict:
    """
    Compute the bounding box of all defined zone polygons.
    Used to filter detections clearly outside the monitored area.
    Adds a configurable buffer beyond the zone boundary.
    """
    BUFFER_M = 2.0   # meters beyond zone edge before filtering

    if not zones:
        # No zones defined — use generous defaults
        return {
            "min_x": -50.0, "max_x": 50.0,
            "min_y":   0.0, "max_y": 50.0
        }

    all_x = []
    all_y = []
    for zone in zones:
        coords = list(zone["polygon"].exterior.coords)
        all_x.extend(c[0] for c in coords)
        all_y.extend(c[1] for c in coords)

    return {
        "min_x": min(all_x) - BUFFER_M,
        "max_x": max(all_x) + BUFFER_M,
        "min_y": min(all_y) - BUFFER_M,
        "max_y": max(all_y) + BUFFER_M
    }

# ── Homography loader ─────────────────────────────────────────────────────────

def load_homography(cal_file: str):
    if not os.path.exists(cal_file):
        logger.warning(f"No calibration file at {cal_file} — world coords disabled")
        return None
    with open(cal_file) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    logger.info("Homography loaded")
    return H

def image_to_world(H, u: float, v: float) -> tuple[float, float] | None:
    if H is None:
        return None
    pt = np.array([[[u, v]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0][0][0]), float(world[0][0][1])

# ── Trajectory store ──────────────────────────────────────────────────────────

class TrajectoryStore:
    """Keeps recent world-coordinate trajectory per tracked entity"""
    def __init__(self, maxlen: int = 30):
        self.maxlen  = maxlen
        self.store: dict[str, list] = {}
        self.dwell:  dict[str, float] = {}
        self.first_seen: dict[str, float] = {}

    def update(self, entity_id: str,
               world_pos: tuple[float, float] | None) -> list:
        now = time.time()
        if entity_id not in self.first_seen:
            self.first_seen[entity_id] = now

        if world_pos:
            if entity_id not in self.store:
                self.store[entity_id] = []
            self.store[entity_id].append(list(world_pos))
            if len(self.store[entity_id]) > self.maxlen:
                self.store[entity_id].pop(0)

        self.dwell[entity_id] = now - self.first_seen[entity_id]
        return self.store.get(entity_id, [])

    def dwell_time(self, entity_id: str) -> float:
        return round(self.dwell.get(entity_id, 0.0), 1)

    def prune(self, active_ids: set):
        for eid in list(self.store.keys()):
            if eid not in active_ids:
                del self.store[eid]
                del self.dwell[eid]
                del self.first_seen[eid]

# ── Camera ────────────────────────────────────────────────────────────────────

def open_capture(url: str) -> cv2.VideoCapture:
    for attempt in range(10):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera opened: {w}x{h}")
            return cap
        logger.warning(f"Camera open failed ({attempt+1}/10) — retrying in 3s")
        cap.release()
        time.sleep(3)
    raise RuntimeError(f"Cannot open camera after 10 attempts: {url}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Redis
    r = None
    for attempt in range(10):
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            r.ping()
            logger.info("Redis connected")
            break
        except Exception as e:
            logger.warning(f"Redis not ready ({attempt+1}/10): {e}")
            time.sleep(2)
    if r is None:
        logger.error("Redis unavailable — exiting")
        sys.exit(1)

    # Model — try TensorRT engine first, fall back to PyTorch
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading TensorRT engine: {MODEL_PATH}")
        model = YOLO(MODEL_PATH, task='detect')
    else:
        logger.warning("TensorRT engine not found — loading PyTorch model")
        model = YOLO("/app/models/yolov8n.pt")

    # Warm up GPU
    logger.info("Warming up model...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, verbose=False)
    # Keep GPU context alive between inference calls
    import torch
    _gpu_keepalive = torch.zeros(1, device='cuda')        
    logger.info("Model ready")

    # Calibration and zones
    H     = load_homography("/app/calibration/calibration.json")
    zones = load_zones("/app/configs/zone_config.json")

    # Compute scene bounds from zone polygons
    # Used to filter detections that are clearly outside the monitored area
    scene_bounds = compute_scene_bounds(zones)
    logger.info(
        f"Scene bounds: x=[{scene_bounds['min_x']:.1f}, "
        f"{scene_bounds['max_x']:.1f}] "
        f"y=[{scene_bounds['min_y']:.1f}, "
        f"{scene_bounds['max_y']:.1f}]"
    )

    # Camera
    cap = open_capture(RTSP_URL)

    # Pipeline components
    pub        = Publisher(port=5555)
    traj_store = TrajectoryStore(maxlen=30)
    interval   = 1.0 / SCENE_STATE_FPS
    frame_id   = 0

    # FPS + latency tracking
    fps_count  = 0
    fps_timer  = time.time()
    prev_ids   = set()

    logger.info("Live perception with YOLOv8 started")

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame read failed — reconnecting")
            cap.release()
            time.sleep(2)
            cap = open_capture(RTSP_URL)
            continue

        # ── Inference ────────────────────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,       # ByteTrack — maintains IDs across frames
            classes=list(TRACKED_CLASSES.keys()),
            conf=0.6,           # minimum detection confidence
            iou=0.5,            # NMS IoU threshold
            verbose=False,
            device=0            # GPU
        )

        # ── Entity construction ───────────────────────────────────────────────
        entities   = []
        active_ids = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Skip if tracker hasn't assigned an ID yet
                if box.id is None:
                    continue

                track_id  = int(box.id.item())
                class_id  = int(box.cls.item())
                conf      = float(box.conf.item())
                entity_id = f"{TRACKED_CLASSES.get(class_id, 'object')}_{track_id:03d}"

                # Bounding box — xyxy format
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Feet position — bottom-center of bounding box
                feet_u = (x1 + x2) / 2
                feet_v = y2

                # World coordinates via homography
                world_pos = image_to_world(H, feet_u, feet_v)

                # Early velocity estimate from trajectory store
                # (before update — uses last known position)
                prev_traj = traj_store.store.get(entity_id, [])
                if world_pos and len(prev_traj) >= 2:
                    dx = world_pos[0] - prev_traj[-1][0]
                    dy = world_pos[1] - prev_traj[-1][1]
                    velocity_raw = (dx**2 + dy**2) ** 0.5 / interval
                else:
                    velocity_raw = 0.0

                # Filter stationary far-field detections
                # Uses scene bounds derived from zone_config.json
                # plus a 2m buffer — avoids hardcoded distances
                if world_pos \
                        and world_pos[1] > scene_bounds["max_y"] \
                        and velocity_raw < 0.05:
                    logger.debug(
                        f"Filtered far-field detection: "
                        f"entity={entity_id} "
                        f"pos=({world_pos[0]:.2f}, {world_pos[1]:.2f}) "
                        f"vel={velocity_raw:.3f} "
                        f"conf={conf:.3f}"
                    )
                    continue

                # Trajectory and dwell time
                trajectory = traj_store.update(entity_id, world_pos)
                dwell_time = traj_store.dwell_time(entity_id)

                # Velocity — from last two trajectory points
                velocity = 0.0
                if len(trajectory) >= 2:
                    dx = trajectory[-1][0] - trajectory[-2][0]
                    dy = trajectory[-1][1] - trajectory[-2][1]
                    velocity = round(
                        (dx**2 + dy**2) ** 0.5 / interval, 2
                    )

                # Zone assignment
                zone = None
                if world_pos:
                    zone = assign_zone(zones, world_pos[0], world_pos[1])

                active_ids.add(entity_id)

                entity = {
                    "id":           entity_id,
                    "class":        TRACKED_CLASSES.get(class_id, "object"),
                    "confidence":   round(conf, 3),
                    "track_id":     track_id,
                    "bbox_image":   [round(x1), round(y1),
                                     round(x2), round(y2)],
                    "feet_image":   [round(feet_u), round(feet_v)],
                    "dwell_time_seconds": dwell_time,
                    "velocity_ms":  velocity,
                    "trajectory":   trajectory[-10:],  # last 10 points
                    "zone":         zone
                }

                if world_pos:
                    entity["position_2d"] = {
                        "x": round(world_pos[0], 2),
                        "y": round(world_pos[1], 2)
                    }

                entities.append(entity)

        # Prune stale trajectories
        traj_store.prune(active_ids)

        # ── Scene delta ───────────────────────────────────────────────────────
        current_ids    = active_ids
        new_entities   = list(current_ids - prev_ids)
        removed        = list(prev_ids - current_ids)
        changed        = list(current_ids & prev_ids)
        prev_ids       = current_ids

        # ── Zone summary ──────────────────────────────────────────────────────
        zone_counts: dict[str, int] = {}
        for entity in entities:
            if entity["zone"]:
                zid = entity["zone"]["id"]
                zone_counts[zid] = zone_counts.get(zid, 0) + 1

        # ── Scene state ───────────────────────────────────────────────────────
        t   = time.time()
        state = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S.") +
                          f"{int((t % 1) * 1000):03d}Z",
            "frame_id":   frame_id,
            "source":     "live_yolov8",
            "frame_shape": {
                "width":  frame.shape[1],
                "height": frame.shape[0]
            },
            "entity_count": len(entities),
            "entities":   entities,
            "zone_counts": zone_counts,
            "scene_delta": {
                "new_entities":     new_entities,
                "removed_entities": removed,
                "changed_entities": changed
            },
            "inference_ms": round((time.time() - loop_start) * 1000, 1)
        }

        state_json = json.dumps(state)
        pub.publish("scene_state", state)
        r.set("scene:latest", state_json)
        r.lpush("scene:replay_buffer", state_json)
        r.ltrim("scene:replay_buffer", 0, 99999)

        # ── FPS logging ───────────────────────────────────────────────────────
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 5.0:
            fps = fps_count / (now - fps_timer)
            logger.info(
                f"{fps:.1f}fps | frame={frame_id} | "
                f"entities={len(entities)} | "
                f"inference={state['inference_ms']}ms"
            )
            fps_count = 0
            fps_timer = now

        frame_id += 1

        elapsed    = time.time() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()

if __name__ == "__main__":
    main()