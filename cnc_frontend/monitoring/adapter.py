import json
import os
import threading
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BACKEND_URL = os.getenv("CNC_BACKEND_URL", "http://127.0.0.1:9000").rstrip("/")

@dataclass
class Session:
    machine_id: int
    running: bool = False
    started_at: float | None = None
    people_inside: int = 0
    last_error: str | None = None

class MonitoringManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[int, Session] = {}

    def _request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{BACKEND_URL}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("detail", str(exc))
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = str(exc)
            raise RuntimeError(f"CNC backend request failed: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"CNC backend request failed: {exc}") from exc

    @staticmethod
    def _camera_id(machine_id: int) -> str:
        return f"camera_{machine_id:02d}"

    def start(self, machine: dict):
        machine_id = int(machine["id"])
        camera_id = self._camera_id(machine_id)
        result = self._request(
            "POST",
            "/detection/start",
            {
                "camera_id": camera_id,
                "source": machine["rtsp_url"],
                "max_persons": machine["max_persons"],
                "zone_limits": machine.get("zone_limits", {}),
                "multiple_limit_seconds": machine["multiple_limit_seconds"],
                "absence_limit_seconds": machine["absence_limit_seconds"],
            },
        )
        session = Session(
            machine_id=machine_id,
            running=bool(result.get("running")),
            started_at=time.time(),
        )
        with self._lock:
            self._sessions[machine_id] = session
        return session

    def stop(self, machine_id: int):
        try:
          result = self._request("POST", f"/detection/{self._camera_id(machine_id)}/stop")
        except RuntimeError as exc:
            with self._lock:
                session=self._sessions.get(machine_id)
                if session:
                    session.last_error=str(exc)
            raise
        with self._lock:
            session = self._sessions.get(machine_id)
            if session:
                session.running = bool(result.get("running"))
                session.last_error= result.get("last_error")
            return session

    def delete_machine(self, machine_id: int):
        return self._request("DELETE", f"/detection/{self._camera_id(machine_id)}")

    def status(self, machine_id: int):
        with self._lock:
            session = self._sessions.get(machine_id)
            started_at = session.started_at if session else None

        try:
            result = self._request(
                "GET", f"/detection/{self._camera_id(machine_id)}/status"
            )
        except RuntimeError as exc:
            with self._lock:
                session = self._sessions.get(machine_id)
                if session:
                    session.last_error = str(exc)
            return {
                "running": False,
                "people_inside": 0,
                "zones": [],
                "last_error": str(exc),
                "started_at": started_at,
            }

        zones = result.get("zones", [])
        people_inside = sum(
            int(zone.get("present", 0))
            for zone in zones
        )

        with self._lock:
            session = self._sessions.get(machine_id)
            if session:
                session.running = bool(result.get("running"))
                session.people_inside = people_inside
                session.last_error = result.get("last_error")

        return {
            "running": bool(result.get("running")),
            "people_inside": people_inside,
            "zones": zones,
            "last_error": result.get("last_error"),
            "started_at": started_at,
        }

    def events(self):
        return self._request("GET", "/events")

    def delete_event(self, event_id):
        return self._request("DELETE", f"/events/{event_id}")
