# src/twin/main.py
# Minimal FastAPI service — health check + latest scene state

import json
import os
import redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import anthropic
from context_builder import ContextBuilder
import logging
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logger = logging.getLogger("twin")

app = FastAPI(title="SensorOS Twin")

r = redis.Redis(
    host="redis",
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

# Serve static files
app.mount(
    "/static",
    StaticFiles(directory="/app/static"),
    name="static"
)

@app.get("/")
def dashboard():
    return FileResponse("/app/static/index.html")

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WebSocket connected — {len(self.active)} clients")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info(f"WebSocket disconnected — {len(self.active)} clients")

    async def broadcast(self, data: dict):
        if not self.active:
            return
        message = json.dumps(data)
        dead    = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.active -= dead

manager = ConnectionManager()


# ── Background task — push scene + anomaly state to WebSocket clients ─────────

async def broadcast_loop():
    """
    Reads latest state from Redis at 10fps and pushes to
    all connected WebSocket clients.
    """
    while True:
        try:
            raw_scene   = r.get("scene:latest")
            raw_anomaly = r.get("anomaly:latest")
            raw_alerts  = r.lrange("alerts:history", 0, 9)

            scene   = json.loads(raw_scene)   if raw_scene   else {}
            anomaly = json.loads(raw_anomaly) if raw_anomaly else {}
            alerts  = [json.loads(a) for a in raw_alerts]

            payload = {
                "type":    "update",
                "scene":   scene,
                "anomaly": anomaly,
                "alerts":  alerts
            }

            await manager.broadcast(payload)

        except Exception as e:
            logger.error(f"Broadcast error: {e}")

        await asyncio.sleep(0.1)   # 10fps


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive — client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# Initialize context builder
ZONE_CONFIG = "/app/configs/zone_config.json"
context_builder = ContextBuilder(r, ZONE_CONFIG)

# Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "")
)

SYSTEM_PROMPT = """You are SensorOS, an intelligent security monitoring assistant.
You have access to real-time data from a digital twin of a monitored physical space.
Answer questions accurately based only on the provided scene context.
Be concise and direct. If something is not in the context, say so clearly.
When describing positions, use natural language (e.g. "near the entrance", "5 meters from the door").
Flag anything that seems security-relevant even if not directly asked."""

@app.get("/health")
def health():
    try:
        r.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/scene/latest")
def latest_scene():
    raw = r.get("scene:latest")
    if not raw:
        return JSONResponse(
            status_code=404,
            content={"detail": "No scene state available yet"}
        )
    return json.loads(raw)

@app.get("/scene/replay")
def replay_buffer(limit: int = 100):
    raw = r.lrange("scene:replay_buffer", 0, limit - 1)
    return [json.loads(s) for s in raw]

@app.get("/anomaly/latest")
def latest_anomaly():
    raw = r.get("anomaly:latest")
    if not raw:
        return JSONResponse(
            status_code=404,
            content={"detail": "No anomaly data available yet"}
        )
    return json.loads(raw)

@app.get("/query")
async def query(q: str):
    """Natural language query about the current scene state"""
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "Query parameter 'q' is required"}
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        return JSONResponse(
            status_code=503,
            content={"error": "ANTHROPIC_API_KEY not configured"}
        )

    try:
        # Build context from live scene state
        context = context_builder.build()

        # Query Claude
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Scene context:\n{context}\n\nQuestion: {q}"
                }
            ]
        )

        answer = response.content[0].text

        return {
            "question": q,
            "answer":   answer,
            "context_frame": json.loads(
                r.get("scene:latest") or "{}"
            ).get("frame_id"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/alerts/history")
def alert_history(limit: int = 20):
    """Recent alert history"""
    raw = r.lrange("alerts:history", 0, limit - 1)
    return [json.loads(a) for a in raw]