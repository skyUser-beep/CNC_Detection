import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

data_dir = project_root / "data"
videos_dir = data_dir / "videos"
models_dir = project_root / "models"
outputs_dir = data_dir / "outputs"

for folder in (videos_dir, models_dir, outputs_dir):
    folder.mkdir(parents=True, exist_ok=True)


def resolve_path(value, default):
    raw = os.getenv(value, default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root / path

model_path = resolve_path("MODEL_PATH", str(models_dir / "yolo11n.pt"))
phone_model_path = resolve_path("PHONE_MODEL_PATH", str(model_path))

cameras = {}
for index in range(1, 5):
    key = f"CAMERA_{index:02d}"
    value = os.getenv(key, "").strip()
    if value:
        path = Path(value).expanduser()
        source = str(path if path.is_absolute() else project_root / path)
        cameras[f"camera_{index:02d}"] = source

if not cameras:
    for index in range(1, 5):
        key = f"CAMERA_SOURCE_{index:02d}"
        value = os.getenv(key, "").strip()
        if value:
            path = Path(value).expanduser()
            source = str(path if path.is_absolute() else project_root / path)
            cameras[f"camera_{index:02d}"] = source

model_path = str(model_path)
phone_model_path = str(phone_model_path)

person_model_path = model_path
phone_class_id = 67
person_confidence = float(os.getenv("PERSON_CONFIDENCE", "0.30"))
phone_confidence = float(os.getenv("PHONE_CONFIDENCE", "0.50"))
person_image_size = int(os.getenv("PERSON_IMAGE_SIZE", "640"))
phone_image_size = int(os.getenv("PHONE_IMAGE_SIZE", "512"))
person_use_augment = os.getenv("PERSON_USE_AUGMENT", "false").strip().lower() == "true"
person_detection_interval = 1
phone_detection_interval = 6

# Tracker config. Point this at tracking_configs/botsort_reid.yaml once you
# have a GPU -- ReID-based re-identification survives occlusion far better
# than plain box/IoU matching, but is too expensive to run on CPU.
person_tracker_config = os.getenv("PERSON_TRACKER_CONFIG", "botsort.yaml")
_tracker_path = Path(person_tracker_config)
if not _tracker_path.is_absolute() and (project_root / person_tracker_config).exists():
    person_tracker_config = str(project_root / person_tracker_config)
multiple_person_limit_seconds = 120
multiple_person_cooldown_seconds = 1800
track_id_switch_max_seconds = 10.0
track_id_switch_center_ratio = 1.0
track_id_switch_iou_threshold = 0.10
absence_limit_seconds = 300
zone_margin_px = 60
track_grace_seconds = 1.5
inside_grace_seconds = 1.5
display_width = 900
multiple_limit_seconds=120
duplicate_iou_threshold = 0.65
duplicate_center_ratio = 0.25


# Backward-compatible aliases.
model_path = model_path
outputs_dir = outputs_dir
cameras = cameras
