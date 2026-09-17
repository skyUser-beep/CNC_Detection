import copy
import traceback
import threading
import time
from types import SimpleNamespace
from types import ModuleType

import cv2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from camera.worker import CameraWorker
from detection.phone_detector import PhoneDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager


app = FastAPI(title="CNC Detection API")
workers: dict[str, CameraWorker] = {}
workers_lock = threading.Lock()
starting: set[str] = set()
startup_errors: dict[str, str] = {}
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
                if isinstance(value, ModuleType):
                    continue
                try:
                    value = copy.deepcopy(value)
                except (TypeError, copy.Error):
                    value = value
                setattr(worker_config, name, value)

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


def start_worker(request: StartRequest):
    try:
        worker = make_worker(request)
        with workers_lock:
            workers[request.camera_id] = worker
            startup_errors.pop(request.camera_id, None)
    except Exception as exc:
        traceback.print_exc()
        with workers_lock:
            startup_errors[request.camera_id] = f"{type(exc).__name__}: {exc}"
    finally:
        with workers_lock:
            starting.discard(request.camera_id)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def detection_events():
    records = events.list_events()
    for record in records:
        if record["screenshot"]:
            record["screenshot_url"] = (
                f"/events/{record['camera_id']}/screenshots/"
                f"{record['screenshot']}"
            )
    return records


@app.get("/events/{camera_id}/screenshots/{filename}")
def event_screenshot(camera_id: str, filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid screenshot name")
    path = events.camera_dir(camera_id) / "screenshots" / filename
    if not path.is_file():
        raise HTTPException(404, "Screenshot not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/detection/start")
def start_detection(request: StartRequest):
    with workers_lock:
        existing = workers.get(request.camera_id)
        if existing and existing.is_alive() and existing.running:
            return {"camera_id": request.camera_id, "running": True}
        if request.camera_id in starting:
            return {"camera_id": request.camera_id, "running": False, "starting": True}
        starting.add(request.camera_id)
        startup_errors.pop(request.camera_id, None)
        threading.Thread(
            target=start_worker,
            args=(request,),
            daemon=True,
            name=f"api-start-{request.camera_id}",
        ).start()
    return {"camera_id": request.camera_id, "running": False, "starting": True}


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
        error = startup_errors.get(camera_id)
        is_starting = camera_id in starting
        if worker is None:
            return {
                "camera_id": camera_id,
                "running": False,
                "starting": is_starting,
                "people_inside": 0,
                "last_error": error,
            }
        safety = worker.safety
        return {
            "camera_id": camera_id,
            "running": bool(worker.is_alive() and worker.running),
            "starting": is_starting,
            "people_inside": int(getattr(safety, "number_inside", 0)),
            "last_error": error or (str(worker.error) if worker.error else None),
        }


def mjpeg_frames(camera_id: str):
    while True:
        with workers_lock:
            worker = workers.get(camera_id)

        if worker is None:
            break

        frame = worker.get_display()
        if frame is None:
            if not worker.is_alive() and not worker.running:
                break
            time.sleep(0.05)
            continue

        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
        time.sleep(0.05)


@app.get("/detection/{camera_id}/stream")
def detection_stream(camera_id: str):
    with workers_lock:
        worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(404, "Detection is not running for this camera")
    return StreamingResponse(
        mjpeg_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
