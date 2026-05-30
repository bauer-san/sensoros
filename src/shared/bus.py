# src/shared/bus.py
import zmq
import json
import logging

logger = logging.getLogger(__name__)

class Publisher:
    def __init__(self, port: int):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        logger.info(f"Publisher bound to port {port}")

    def publish(self, topic: str, data: dict):
        try:
            self.sock.send_string(
                f"{topic} {json.dumps(data)}",
                zmq.NOBLOCK
            )
        except zmq.Again:
            logger.warning(f"Publisher queue full on topic {topic}")

class Subscriber:
    def __init__(self, host: str, port: int, topic: str):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{host}:{port}")
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
        logger.info(f"Subscriber connected to {host}:{port} topic={topic}")

    def receive(self) -> dict | None:
        try:
            msg = self.sock.recv_string()
            _, payload = msg.split(" ", 1)
            return json.loads(payload)
        except zmq.Again:
            return None