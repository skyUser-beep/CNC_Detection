# import os
# from pathlib import Path
# from dotenv import load_dotenv
# PROJECT_ROOT = Path(__file__).resolve().parent
# load_dotenv(PROJECT_ROOT / ".env")
#
# DATA_DIR = PROJECT_ROOT / "data"
# VIDEOS_DIR = DATA_DIR / "videos"
# MODELS_DIR = PROJECT_ROOT / "models"
# OUTPUTS_DIR = PROJECT_ROOT / "outputs"
#
# for _folder in (VIDEOS_DIR, MODELS_DIR, OUTPUTS_DIR):
#     _folder.mkdir(parents=True, exist_ok=True)
#
# MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "yolo11n.pt"))
# VIDEO_PATH = os.getenv("VIDEO_PATH", str(VIDEOS_DIR / "testv4.mp4"))
#
# CAMERA_IP = os.getenv("CAMERA_IP", "YOUR_IP")
# CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
# CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "YOUR_PASSWORD")
#
# _VLC_CANDIDATES = [os.getenv("VLC_PATH", ""),
#     r"C:\Program Files\VideoLAN\VLC\vlc.exe",
#     r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
# ]
#
# def find_vlc_executable():
#     """Return the first VLC executable path that exists on this
#     machine, or None if none of the candidates were found."""
#     for path in _VLC_CANDIDATES:
#         if path and os.path.exists(path):
#             return path
#     return None

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for folder in (VIDEOS_DIR, MODELS_DIR, OUTPUTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH",str(MODELS_DIR / "yolo11n.pt"))
PHONE_MODEL_PATH = os.getenv("PHONE_MODEL_PATH",MODEL_PATH)
CAMERAS = {
    "camera_01": os.getenv(
        "CAMERA_01",
        str(VIDEOS_DIR / "testv1.mp4")
    ),
    "camera_02": os.getenv(
        "CAMERA_02",
        str(VIDEOS_DIR / "testv2.mp4")
    ),
    "camera_03": os.getenv(
        "CAMERA_03",
        str(VIDEOS_DIR / "testv3.mp4")
    ),
    "camera_04": os.getenv(
        "CAMERA_04",
        str(VIDEOS_DIR / "testv4.mp4")
    ),
}
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "YOUR_PASSWORD")

_VLC_CANDIDATES = [
    os.getenv("VLC_PATH", ""),
    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
]
def find_vlc_executable():
    for path in _VLC_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return None