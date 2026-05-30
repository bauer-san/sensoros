# src/anomaly/anomaly_node.py
# Stub that subscribes to scene state and logs receipt

import time
import logging
import os

from shared.bus import Subscriber

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("anomaly")

def main():
    sub = Subscriber(
        host="perception",
        port=5555,
        topic="scene_state"
    )
    logger.info("Anomaly stub started — listening for scene states")

    count = 0
    last_log = time.time()

    while True:
        state = sub.receive()
        if state:
            count += 1
            now = time.time()
            if now - last_log >= 5.0:
                fps = count / (now - last_log)
                logger.info(
                    f"Receiving scene states at {fps:.1f}fps | "
                    f"frame_id={state['frame_id']} | "
                    f"entities={len(state['entities'])}"
                )
                count = 0
                last_log = now

if __name__ == "__main__":
    main()