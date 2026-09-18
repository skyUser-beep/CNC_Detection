import copy # copies conf. values
import asyncio # Handles async streaming
import shutil # Delete camera directories
import threading # Protects shared worker state
import time 
from types import SimpleNamespace # Creates a conf. object dynamically
from types import ModuleType # helps executes modules while copying conf

import cv2 # encodes video frames as jpeg images
import torch # checks for GPU

# web server,API error responses, streaming response, static file hosting, and request validation
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


import config
from camera.worker import CameraWorker
from detection.phone_detector import PhoneDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager

app = FastAPI(title="CNC Detection API")
workers: dict[str, CameraWorker] = {}
workers_lock = threading.Lock()
events = EventManager(config.outputs_dir)
zones = ZoneManager(config.outputs_dir, config.zone_margin_px)
app.mount("/outputs", StaticFiles(directory=config.outputs_dir), name="outputs")  # Security consideration only for trusted local network

class StartRequest(BaseModel): # validate incoming JSON (Pydantic)
    camera_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    max_persons: int = Field(default=1, ge=1, le=50)
    zone_limits: dict[str, int] = Field(default_factory=dict)
    multiple_limit_seconds: int = Field(default=120, ge=1)
    absence_limit_seconds: int = Field(default=300, ge=1)

@app.on_event("shutdown") 
def shutdown_workers():
    with workers_lock:
        active_workers = list(workers.values())
    for worker in active_workers:
        worker.running = False
        if worker.reader is not None:
            worker.reader.stop()

def choose_device():
    if torch.cuda.is_available():
        return 0, True
    return "cpu", False

def make_worker(request: StartRequest):
    device, half = choose_device()
    phone_detector = PhoneDetector(
        config.phone_model_path,
        device,half,
        config.phone_confidence,
        config.phone_image_size,
        config.phone_class_id,
    )
    worker_config = SimpleNamespace()
    for name in dir(config):
        if not name.startswith("_"):
            value = getattr(config, name)
            if not callable(value) and not isinstance(value, ModuleType):
                setattr(worker_config, name, copy.deepcopy(value))

    worker_config.multiple_limit_seconds = request.multiple_limit_seconds
    worker_config.multiple_person_limit_seconds = request.multiple_limit_seconds
    worker_config.absence_limit_seconds = request.absence_limit_seconds
    worker_config.allowed_people_per_zone = request.zone_limits or {
        "zone_1": request.max_persons,
    }

    worker = CameraWorker(
        request.camera_id,
        request.source,
        worker_config,
        device,
        half,
        phone_detector,
        events,
        zones,
    )
    if not worker.prepare():
        raise RuntimeError("Camera worker could not prepare the camera or zones")
    worker.startup_ready.set()
    worker.start()
    return worker

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/events")
def list_events():
    records = events.list_events()
    for record in records:
        screenshot = record.pop("screenshot")
        record["screenshot_url"] = (
            f"/outputs/cameras/{record['camera_id']}/screenshots/{screenshot}"
            if screenshot
            else None
        )
    return records

@app.delete("/events/{event_id}")
def delete_event(event_id: str):
    if not events.delete_event(event_id):
        raise HTTPException(404, "Event not found")
    return {"deleted": True}

@app.post("/detection/start")
def start_detection(request: StartRequest):
    with workers_lock:
        existing = workers.get(request.camera_id)
        if existing and existing.is_alive() and existing.running:
            return {"camera_id": request.camera_id, "running": True}
        try:
            workers[request.camera_id] = make_worker(request)
        except Exception as exc:
            raise HTTPException(500, f"Failed to start detection: {exc}") from exc
    return {"camera_id": request.camera_id, "running": True}


@app.post("/detection/{camera_id}/stop")
def stop_detection(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)
        if worker is not None:
            worker.running = False
            if worker.reader is not None:
                worker.reader.stop()
    return {"camera_id": camera_id, "running": False}

@app.delete("/detection/{camera_id}")
def delete_detection(camera_id: str):
    stop_detection(camera_id)
    camera_dir = events.base / camera_id
    if camera_dir.exists():
        shutil.rmtree(camera_dir)
    return {"camera_id": camera_id, "deleted": True}

@app.get("/detection/{camera_id}/status")
def detection_status(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)
        if worker is None:
            return {"camera_id": camera_id, "running": False, "people_inside": 0}
        safety = worker.safety
        zones = [
            {
                "zone": index + 1,
                "present": int(safety.current_zone_counts[index]),
                "allowed": safety.allowed_people_per_zone[index],
            }
            for index in range(len(safety.current_zone_counts))
        ]
        return {
            "camera_id": camera_id,
            "running": bool(worker.is_alive() and worker.running),
            "people_inside": sum(zone["present"] for zone in zones),
            "zones": zones,
            "last_error": str(worker.error) if worker.error else None,
        }

@app.get("/detection/{camera_id}/stream")
async def detection_stream(camera_id: str, request: Request):
    with workers_lock:
        worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(404, "Detection camera is not running")

    async def frames():
        while worker.running and worker.reader is not None:
            if await request.is_disconnected():
                break
            frame = worker.get_display()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"Could not encode stream frame for {camera_id}")
            
            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
            await asyncio.sleep(0.04)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )