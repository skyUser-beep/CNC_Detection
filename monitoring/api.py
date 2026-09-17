import copy
import threading
from types import SimpleNamespace

import torch
from fastapi import FastAPI, HTTPException
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

class StartRequest(BaseModel):
    camera_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    max_persons: int = Field(default=1, ge=1, le=50)
    multiple_limit_seconds: int = Field(default=120, ge=1)
    absence_limit_seconds: int = Field(default=300, ge=1)

def choose_device():
    if torch.cuda.is_available():
        return 0, True
    return "cpu", False

def make_worker(request: StartRequest):
    device, half = choose_device()
    phone_detector = PhoneDetector(
        config.phone_model_path,
        device,
        half,
        config.phone_confidence,
        config.phone_image_size,
        config.phone_class_id,
    )
    worker_config = SimpleNamespace()
    for name in dir(config):
        if not name.startswith("_"):
            value = getattr(config, name)
            if not callable(value):
                setattr(worker_config, name, copy.deepcopy(value))

    worker_config.multiple_limit_seconds = request.multiple_limit_seconds
    worker_config.multiple_person_limit_seconds = request.multiple_limit_seconds
    worker_config.absence_limit_seconds = request.absence_limit_seconds

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


@app.get("/detection/{camera_id}/status")
def detection_status(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)
        if worker is None:
            return {"camera_id": camera_id, "running": False, "people_inside": 0}
        safety = worker.safety
        return {
            "camera_id": camera_id,
            "running": bool(worker.is_alive() and worker.running),
            "people_inside": int(getattr(safety, "number_inside", 0)),
            "last_error": str(worker.error) if worker.error else None,
        }
