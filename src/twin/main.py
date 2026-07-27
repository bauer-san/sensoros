# src/twin/main.py

import json
import os
import time
import logging
import asyncio
from typing import Set

import redis
import anthropic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from context_builder import ContextBuilder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("twin")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SensorOS Twin")

# ── Redis ─────────────────────────────────────────────────────────────────────
r = redis.Redis(
    host="redis",
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount(
    "/static",
    StaticFiles(directory="/app/static"),
    name="static"
)

# ── Context builder ───────────────────────────────────────────────────────────
context_builder = ContextBuilder(r, "/app/configs/zone_config.json")

# ── Anthropic client ──────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "")
)

SYSTEM_PROMPT = """You are SensorOS, an intelligent security monitoring assistant.
You have access to real-time data from a digital twin of a monitored physical space.
Answer questions accurately based only on the provided scene context.
Be concise and direct. If something is not in the context, say so clearly.
When describing positions, use natural language (e.g. "near the entrance", "5 meters from the door").
Flag anything that seems security-relevant even if not directly asked."""

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

# ── Background broadcast task ─────────────────────────────────────────────────

async def broadcast_loop():
    while True:
        try:
            raw_scene   = r.get("scene:latest")
            raw_anomaly = r.get("anomaly:latest")
            raw_alerts  = r.lrange("alerts:history", 0, 9)

            scene   = json.loads(raw_scene)   if raw_scene   else {}
            anomaly = json.loads(raw_anomaly) if raw_anomaly else {}
            alerts  = [json.loads(a) for a in raw_alerts]

            await manager.broadcast({
                "type":    "update",
                "scene":   scene,
                "anomaly": anomaly,
                "alerts":  alerts
            })
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

        await asyncio.sleep(0.1)

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def dashboard():
    return FileResponse("/app/static/index.html")

@app.get("/path-viz")
def path_visualizer():
    return FileResponse("/app/static/path_visualizer.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

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
        context  = context_builder.build()
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
        return {
            "question":      q,
            "answer":        response.content[0].text,
            "context_frame": json.loads(
                r.get("scene:latest") or "{}"
            ).get("frame_id"),
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/alerts/history")
def alert_history(limit: int = 20):
    raw = r.lrange("alerts:history", 0, limit - 1)
    return [json.loads(a) for a in raw]