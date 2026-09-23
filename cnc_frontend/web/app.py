from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database.db import (init_db, list_machines, get_machine, create_machine, set_active,
    update_machine_settings, recent_events, add_event,
    delete_machine as delete_machine_record,
)

from monitoring.adapter import MonitoringManager

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="CNC Monitoring Dashboard")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
manager = MonitoringManager()

def camera_host(rtsp_url: str):
    from urllib.parse import urlsplit
    parsed = urlsplit(rtsp_url.strip())
    if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname:
        raise HTTPException(400, "Camera URL must contain a valid RTSP host")
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return host

def backend_stream(path: str):
    request = UrlRequest(f"{manager.backend_url}{quote(path, safe='/%')}")
    try:
        response = urlopen(request, timeout=10)
    except (HTTPError, URLError) as exc:
        raise HTTPException(502, f"CNC backend stream unavailable: {exc}") from exc
    try:
        while chunk := response.read(64 * 1024):
            yield chunk
    finally:
        response.close()

@app.on_event("startup")
def startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    machines = []
    for machine in list_machines():
        machine["runtime"] = manager.status(machine["id"])
        machines.append(machine)
    try:
        events = manager.events()
    except RuntimeError as exc:
        raise HTTPException(502, f"CNC backend events unavailable: {exc}") from exc
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "machines": machines,
        "events": events,
    })

@app.post("/machines")
def add_machine(name: str = Form(...),
    rtsp_url: str = Form(...),
    max_persons: int = Form(...),
    multiple_limit_seconds: int = Form(120),
    absence_limit_seconds: int = Form(300),
):
    if not rtsp_url.lower().startswith(("rtsp://", "rtsps://")):
        raise HTTPException(400, "Camera URL must start with rtsp:// or rtsps://")
    if max_persons < 1 or max_persons > 50:
        raise HTTPException(400, "Maximum people must be between 1 and 50")
    machine_id = create_machine(name.strip(), rtsp_url.strip(), max_persons, multiple_limit_seconds, absence_limit_seconds)
    return RedirectResponse(url=f"/", status_code=303)

@app.post("/machines/{machine_id}/start")
def start_machine(machine_id: int):
    machine = get_machine(machine_id)
    if not machine:
        raise HTTPException(404, "Machine not found")
    try:
        manager.start(machine)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    set_active(machine_id, True)
    return RedirectResponse(url="/", status_code=303)

@app.post("/machines/{machine_id}/settings")
async def update_settings( request: Request,
    machine_id: int,
    max_persons: int = Form(...),
    multiple_limit_seconds: int = Form(...),
    absence_limit_seconds: int = Form(...),
):
    machine = get_machine(machine_id)
    if not machine:
        raise HTTPException(404, "Machine not found")
    if not 1 <= max_persons <= 50:
        raise HTTPException(400, "Maximum people must be between 1 and 50")
    if multiple_limit_seconds < 1 or absence_limit_seconds < 1:
        raise HTTPException(400, "Delays must be at least 1 second")
    form = await request.form()
    zone_limits = {}
    for key, value in form.items():
        if key.startswith("zone_limit_"):
            zone_number = key.removeprefix("zone_limit_")
            try:
                zone_limits[f"zone_{int(zone_number)}"] = max(1, min(50, int(value)))
            except ValueError as exc:
                raise HTTPException(400, f"Invalid limit for zone {zone_number}") from exc
    if not zone_limits:
        zone_limits = {"zone_1": max_persons}
    update_machine_settings(machine_id,
        max_persons,
        multiple_limit_seconds,
        absence_limit_seconds,
        zone_limits,
    )
    
    if manager.status(machine_id)["running"]:
        manager.stop(machine_id)
        manager.start(get_machine(machine_id))
    return RedirectResponse(url="/", status_code=303)

@app.post("/machines/{machine_id}/stop")
def stop_machine(machine_id: int):
    if not get_machine(machine_id):
        raise HTTPException(404, "Machine not found")
    manager.stop(machine_id)
    set_active(machine_id, False)
    return RedirectResponse(url="/", status_code=303)

@app.post("/machines/{machine_id}/delete")
def delete_machine(machine_id: int):
    if not get_machine(machine_id):
        raise HTTPException(404, "Machine not found")
    try:
        manager.delete_machine(machine_id)
    except RuntimeError as exc:
        raise HTTPException(502, f"CNC backend could not delete machine: {exc}") from exc
    if not delete_machine_record(machine_id):
        raise HTTPException(404, "Machine not found")
    return RedirectResponse(url="/", status_code=303)

@app.get("/api/machines/{machine_id}/status")
def machine_status(machine_id: int):
    if not get_machine(machine_id):
        raise HTTPException(404, "Machine not found")
    return manager.status(machine_id)

@app.post("/api/camera-credentials")
def save_credentials(rtsp_url: str = Form(...), username: str = Form(...), password: str = Form(...)):
    from database.db import save_camera_credentials
    host = camera_host(rtsp_url)
    username = username.strip()
    if not username or not password:
        raise HTTPException(400, "Camera username and password are required")
    save_camera_credentials(host, username, password)
    return {"saved": True}

@app.get("/backend/detection/{camera_id}/stream")
def machine_stream(camera_id: str):
    return StreamingResponse(
        backend_stream(f"/detection/{camera_id}/stream"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/backend/outputs/{file_path:path}")
def output_file(file_path: str):
    return StreamingResponse(
        backend_stream(f"/outputs/{file_path}"),
        media_type="image/jpeg",
    )

@app.get("/api/events")
def events():
    try:
        return manager.events()
    except RuntimeError as exc:
        raise HTTPException(502, f"CNC backend events unavailable: {exc}") from exc

@app.post("/api/events/{event_id}/delete")
def delete_event(event_id: str):
    try:
        manager.delete_event(event_id)
    except RuntimeError as exc:
        raise HTTPException(502, f"CNC backend event deletion failed: {exc}") from exc
    return RedirectResponse(url="/", status_code=303)
