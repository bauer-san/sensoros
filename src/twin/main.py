# src/twin/main.py
# Minimal FastAPI service — health check + latest scene state

import json
import os
import redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="SensorOS Twin")

r = redis.Redis(
    host="redis",
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

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