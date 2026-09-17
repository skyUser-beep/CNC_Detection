# CNC Detection - Refactored Multi-Camera Monitor

This version separates camera reading, AI detection, tracking, zones, safety rules, event logging, screenshots, and display. Each camera has its own worker and person tracker, while the main thread is kept mostly for the GUI.

## Project structure

```text
CNC_Detection_Refactored/
├── config.py
├── .env.example
├── requirements.txt
├── README.md
├── run.bat
├── models/
│   └── yolo11n.pt              # put your model here
├── data/
│   ├── videos/                 # optional test videos
│   └── outputs/
├── camera/
│   ├── __init__.py
│   ├── reader.py
│   └── worker.py
├── detection/
│   ├── __init__.py
│   ├── person_detector.py
│   └── phone_detector.py
├── tracking/
│   ├── __init__.py
│   └── track_state.py
├── zones/
│   ├── __init__.py
│   └── zone_manager.py
├── safety/
│   ├── __init__.py
│   └── rules.py
├── events/
│   ├── __init__.py
│   └── event_manager.py
├── display/
│   ├── __init__.py
│   └── renderer.py
└── monitoring/
    ├── __init__.py
    └── main.py
```

## Windows setup

1. Create a Python 3.10/3.11/3.12 environment.
2. Install the packages:

```bat
python -m pip install -r requirements.txt
```

3. For NVIDIA GPU use, install a PyTorch build compatible with your CUDA setup if PyTorch is not already installed.
4. Copy `.env` to `.env`.
5. Put `yolo11n.pt` in `models/`, or set `model_path` to your existing model.
6. Put your test videos in `data/videos/`, or change the four camera paths in `.env`.
7. Run:

```bat
python -m monitoring.main
```

You can also double-click `run.bat` after Python and dependencies are installed.

## Dashboard API

The web dashboard can control this detector without importing its Python
modules. Start the API from this directory:

```bat
python -m uvicorn api:app --host 127.0.0.1 --port 9000
```

Check that it is running:

```text
http://127.0.0.1:9000/health
```

Start the dashboard separately from the `cnc_web_v1` project:

```bat
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

The dashboard uses `http://127.0.0.1:9000` by default. Set
`CNC_BACKEND_URL` if the API runs on another host or port.

## Controls

- `Q`: quit
- `Z`: redraw zones for the camera selected with a mouse click
- Left click a camera window: select that camera
- During zone setup: left click = point, Enter = finish zone, S = save, C = clear current polygon, D = delete last zone, Q = cancel

## Safety rules preserved

- More than one person in the same work zone for 2 minutes -> event + screenshot + 30-minute cooldown.
- Person away from a zone for more than 5 minutes -> event.
- Phone detected in a work zone -> immediate event + screenshot.
- A phone event is raised only when the phone detection is inside the detected
  person's bounding box; nearby phones or machine controls are ignored.
- A person is considered present when any part of their bounding box overlaps
  a work zone.
- Duplicate boxes are suppressed before counting.
- Recent track history is transferred when BoT-SORT changes an ID and the new detection is spatially close to the previous track.

## Important performance design

Each camera has its own reader and inference worker. The GUI does not run YOLO sequentially for all four cameras. The camera reader keeps the newest frame available, so the application favors low display latency over building up a delayed frame queue.

If GPU memory is tight on an 8 GB card, lower `person_image_size` in `monitoring/main.py` from 640 to 512.
