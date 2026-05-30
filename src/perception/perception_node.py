import time
import json
import logging
import os
import sys
import cv2
import redis

sys.path.insert(0, '/app')
from shared.bus import Publisher

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("perception")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PASSWORD  = os.getenv("REDIS_PASSWORD")
RTSP_URL        = os.getenv("CAMERA_RTSP_URL")
SCENE_STATE_FPS = int(os.getenv("SCENE_STATE_FPS", 30))

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

        logger.warning(f"Camera open failed (attempt {attempt+1}/10) "
                       f"— retrying in 3s")
        cap.release()
        time.sleep(3)

    raise RuntimeError(f"Cannot open camera after 10 attempts: {url}")

def make_scene_state(frame_id: int, frame_shape: tuple) -> dict:
    """
    Minimal scene state from live frame — no detection yet.
    Confirms camera → Redis → anomaly pipeline is working
    with real frames before adding YOLOv8.
    """
    h, w = frame_shape[:2]
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.") +
                     f"{int((time.time() % 1) * 1000):03d}Z",
        "frame_id": frame_id,
        "source":   "live_camera",
        "frame_shape": {"width": w, "height": h},
        "entities": [],          # populated once YOLOv8 is added
        "zones":    {},
        "scene_delta": {
            "changed_entities": [],
            "new_entities":     [],
            "removed_entities": []
        }
    }

def main():
    # Connect to Redis
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

    # Open camera
    cap = open_capture(RTSP_URL)

    pub      = Publisher(port=5555)
    interval = 1.0 / SCENE_STATE_FPS
    frame_id = 0

    # FPS tracking
    fps_count = 0
    fps_timer = time.time()

    logger.info("Live perception started")

    while True:
        loop_start = time.time()

        ret, frame = cap.read()

        if not ret:
            logger.warning("Frame read failed — attempting reconnect")
            cap.release()
            time.sleep(2)
            cap = open_capture(RTSP_URL)
            continue

        # Build and publish scene state
        state      = make_scene_state(frame_id, frame.shape)
        state_json = json.dumps(state)

        pub.publish("scene_state", state)
        r.set("scene:latest", state_json)
        r.lpush("scene:replay_buffer", state_json)
        r.ltrim("scene:replay_buffer", 0, 9999)

        # Log FPS every 5 seconds
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 5.0:
            fps = fps_count / (now - fps_timer)
            logger.info(
                f"Live capture: {fps:.1f}fps | "
                f"frame_id={frame_id} | "
                f"shape={frame.shape[1]}x{frame.shape[0]}"
            )
            fps_count = 0
            fps_timer = now

        frame_id += 1

        # Throttle to target FPS
        elapsed    = time.time() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()

if __name__ == "__main__":
    main()