# CNC Detection and Monitoring Dashboard

Multi-camera CNC safety monitoring with YOLO person/phone detection, tracking,
work-zone rules, event screenshots, and a web dashboard for machine
configuration.

The application has two services:

- **Backend API**: reads RTSP/video sources, runs detection, tracks people,
  applies safety rules, stores events, and serves MJPEG live streams.
- **Frontend dashboard**: registers machines, starts/stops cameras, displays
  live video and counts, configures zone limits, and manages events/machines.

## Repository layout

```text
CNC_Detection/
├── cnc_backend/
│   ├── camera/                 # Camera readers and detection workers
│   ├── detection/              # Person and phone detectors
│   ├── display/               # Rendered stream overlays
│   ├── events/                # CSV event logs and screenshots
│   ├── monitoring/api.py       # FastAPI detection backend
│   ├── safety/                # Occupancy and absence rules
│   ├── tracking/              # Tracker matching and deduplication
│   ├── zones/                 # Zone storage and geometry
│   ├── config.py              # Backend configuration
│   ├── models/                # YOLO model files
│   └── data/outputs/          # Generated events and screenshots
├── cnc_frontend/
│   ├── database/              # SQLite machine configuration
│   ├── monitoring/adapter.py  # Backend API client
│   ├── templates/             # Dashboard HTML
│   ├── static/                # Dashboard CSS
│   └── web/app.py             # FastAPI dashboard
├── requirements.txt
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10-3.12 recommended
- OpenCV-compatible camera/video source
- YOLO model files in `cnc_backend/models/`
- NVIDIA GPU and compatible PyTorch installation are optional

Python 3.14 is not recommended for the computer-vision dependencies. Use a
supported Python version if package installation or OpenCV/PyTorch imports
fail.

## Installation on Windows

From the repository root:

```powershell
cd C:\Users\Sahil\Downloads\CNC_Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands with the virtual
environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The frontend also contains a local environment at
`cnc_frontend\.venv`. Do not mix environments accidentally: the environment
used to start a service must contain that service's dependencies.

## Backend configuration

Backend configuration is loaded from `cnc_backend\.env` when present.

Important settings include:

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Person detection YOLO model |
| `PHONE_MODEL_PATH` | Phone detection model; defaults to `MODEL_PATH` |
| `CAMERA_01` ... `CAMERA_04` | RTSP URLs or local video paths |
| `PERSON_CONFIDENCE` | Person confidence threshold |
| `PHONE_CONFIDENCE` | Phone confidence threshold |
| `PHONE_DETECTION_INTERVAL` | Run phone detection every N frames |
| `PHONE_CLASS_ID` | Phone class ID; COCO uses `67` |
| `PERSON_IMAGE_SIZE` | Person model image size |
| `PERSON_TRACKER_CONFIG` | BoT-SORT tracker configuration |

RTSP values must be valid URLs, for example:

```dotenv
CAMERA_01=rtsp://username:password@192.168.1.10:554/Streaming/Channels/102
```

For local test video:

```dotenv
CAMERA_01=data/videos/test.mp4
```

Keep credentials out of committed files. Use a local `.env` and ensure it is
ignored by Git.

Place the configured model at the path selected by `MODEL_PATH`, or put the
default model in `cnc_backend\models\`.

## Run the application

Start the backend first in **Terminal 1**:

```powershell
cd C:\Users\Sahil\Downloads\CNC_Detection\cnc_backend
..\.venv\Scripts\python.exe -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
```

The simpler form, when the root `.venv` is activated, is:

```powershell
cd C:\Users\Sahil\Downloads\CNC_Detection\cnc_backend
python -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
```

Check the backend:

```text
http://127.0.0.1:9000/health
```

Start the dashboard in **Terminal 2**:

```powershell
cd C:\Users\Sahil\Downloads\CNC_Detection\cnc_frontend
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The frontend uses `http://127.0.0.1:9000` by default. To use another backend
address:

```powershell
$env:CNC_BACKEND_URL = "http://192.168.1.20:9000"
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

### Correct ASGI import paths

Do not run `python -m uvicorn api:app` from `cnc_backend` or
`cnc_frontend`; there is no root-level `api.py` in either directory.

| Service | Directory | Import path | Port |
|---|---|---|---|
| Backend | `cnc_backend` | `monitoring.api:app` | `9000` |
| Frontend | `cnc_frontend` | `web.app:app` | `8000` |

## Dashboard workflow

1. Open the dashboard at `http://127.0.0.1:8000`.
2. Add a machine name, RTSP URL, number of zones, and the required people
   limit for each zone. The default occupancy delay is 120 seconds (2
   minutes).
3. Click **Save machine**, then **Start**.
4. When you click **Start**, the dashboard requests a camera preview from the
   backend. Draw exactly the configured number of work zones in the browser
   and click **Start monitoring**. The backend then persists those polygons
   and starts detection; it never opens a local zone-setup window.
5. Set the limit for each zone in the machine card if the safety procedure
   changes.
6. Click **Update settings**. If the camera is running, it is restarted with
   the new limits.
7. Use **Stop** to stop detection without deleting the machine.
8. Use the red **Delete** button to stop the detector and remove the machine,
   its saved events, and its screenshots.

## Zone-specific occupancy limits

Each zone has its own administrator-defined limit. For example:

```text
Zone 1 limit: 1
Zone 2 limit: 3
Zone 3 limit: 2
```

When the count in a particular zone exceeds that zone's limit, only that
zone's occupancy timer starts. The backend returns the authoritative count for
each zone, and the dashboard calculates the total people count from those
same values.

The machine-wide **Maximum people** value is retained as the fallback/default
for machines without saved zone limits. The zone fields are the values used
for zone-specific occupancy rules.

## Safety and event behavior

- Exceeding a zone's configured limit for the configured occupancy delay
  starts a violation interval. When the zone returns to its allowed count, a
  `PEOPLE_OVER_ALLOWED_LIMIT` event records the interval, for example
  `09:52:00 AM - 10:00:00 AM`, and a screenshot is saved.
- A person whose detection disappears is kept present during the configured
  absence delay (five minutes by default). This prevents a temporary
  detector/tracker miss or person overlap from stopping the violation timer.
  The person is marked gone only after that delay.
- A person absent from a zone for the configured absence delay creates a
  `PERSON_AWAY_OVER_5_MINUTES` event and screenshot.
- Phone detection inside a person's work zone creates a `PHONE_DETECTED`
  event and screenshot.
- Duplicate person detections are suppressed before counting.
- Tracker-ID changes are matched back to the same physical person when
  spatially appropriate.
- A stable tracker ID is preferred when matching a person who moves within a
  zone.
- Screenshots are saved below
  `cnc_backend\data\outputs\cameras\<camera_id>\screenshots\`.
- Event timestamps use the computer's local wall-clock time, such as
  `2026-09-18 10:30:15 AM`. Video-relative time remains stored separately
  for detection calculations.

The timestamp is the backend computer's clock at detection time. Reading the
clock rendered inside the CCTV image would require a separate OCR feature.

## Event management

The dashboard's **Recent events** table provides:

- Event time, camera, type, details, and track IDs
- A link to open the saved screenshot
- A trash icon to delete the event and its screenshot

The backend event API is:

```text
GET    /events
DELETE /events/{event_id}
GET    /outputs/...
```

Events are stored in per-camera `events.csv` files. Existing event files
created by older versions may use video-relative timestamps; new events use
wall-clock timestamps.

## Backend API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/detection/start` | Start a camera worker |
| `POST` | `/detection/{camera_id}/stop` | Stop a camera worker |
| `DELETE` | `/detection/{camera_id}` | Stop and remove camera output data |
| `GET` | `/detection/{camera_id}/status` | Running state and zone counts |
| `GET` | `/detection/{camera_id}/stream` | MJPEG live stream |
| `GET` | `/events` | Recent events |
| `DELETE` | `/events/{event_id}` | Delete an event and screenshot |

## Troubleshooting

### `Could not import module "api"`

Use the correct directory and import path:

```powershell
cd C:\Users\Sahil\Downloads\CNC_Detection\cnc_backend
python -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000
```

### Frontend returns `502 Bad Gateway`

Confirm the backend is running on port `9000` and that this URL responds:

```text
http://127.0.0.1:9000/health
```

### Dashboard says Running but video is blank

Confirm the backend stream URL responds:

```text
http://127.0.0.1:9000/detection/camera_01/stream
```

Also check the backend terminal for RTSP connection or frame-read errors.
The browser cannot display an RTSP URL directly; it displays the backend's
MJPEG stream.

### `python-multipart` error

Install the repository requirements in the environment used by the frontend:

```powershell
python -m pip install -r C:\Users\Sahil\Downloads\CNC_Detection\requirements.txt
```

### The backend does not stop immediately with `Ctrl+C`

The dashboard keeps MJPEG streams open while cameras are running. Stop the
frontend first, then press `Ctrl+C` in the backend terminal. Uvicorn can also
be started with a short graceful-shutdown timeout:

```powershell
python -m uvicorn monitoring.api:app --host 127.0.0.1 --port 9000 --timeout-graceful-shutdown 2
```

### Counts do not match the video

The dashboard count comes from the backend's per-zone status response. Check
that the displayed machine maps to the expected `camera_NN` identifier and
restart the machine after changing zone limits.

### RTSP camera cannot be opened

Check the URL, credentials, network reachability, camera stream permissions,
and whether OpenCV/FFmpeg can access the stream. Test with a local video first
to separate camera connectivity from detection configuration.

## Development notes

Each camera has an independent reader and inference worker. Readers retain the
latest frame rather than building an unbounded queue, favoring low latency.
The backend's stream renders detection overlays before JPEG encoding.

For GPU memory issues, reduce `PERSON_IMAGE_SIZE` in the backend environment.
For CPU-only operation, the backend automatically selects CPU when CUDA is
unavailable.

When stopping or deleting a machine, the backend stops its reader thread before
returning. Restart both services after changing Python source files.
