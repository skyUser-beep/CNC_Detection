from pathlib import Path
import cv2

def format_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

class EventManager:
    def __init__(self, output_dir):
        self.base = Path(output_dir) / "cameras"
        self.base.mkdir(parents=True, exist_ok=True)

    def camera_dir(self, camera_id):
        path = self.base / camera_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, camera_id, video_seconds, event_name, track_ids="", details=""):
        timestamp = format_timestamp(video_seconds)
        print(f"\nSAFETY EVENT\nCamera    : {camera_id}\nTimestamp : {timestamp}\nEvent     : {event_name}")
        if track_ids:
            print("Track IDs :", track_ids)
        if details:
            print("Details   :", details)
        path = self.camera_dir(camera_id) / "events.csv"
        if not path.exists():
            path.write_text("timestamp,video_time,event,track_ids,details\n", encoding="utf-8")
        safe = lambda value: str(value).replace(",", ";")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{safe(timestamp)},{video_seconds:.2f},{safe(event_name)},{safe(track_ids)},{safe(details)}\n")

    def screenshot(self, camera_id, frame, video_seconds, event_name, track_ids, zone_index=None):
        directory = self.camera_dir(camera_id) / "screenshots"
        directory.mkdir(parents=True, exist_ok=True)
        ts = format_timestamp(video_seconds).replace(":", "-")
        ids = "_".join(str(i) for i in sorted(track_ids))
        if zone_index is not None:
            name = f"{ts}_ZONE_{zone_index + 1}_{event_name}_IDs_{ids}.jpg"
        else:
            name = f"{ts}_{event_name}_IDs_{ids}.jpg"
        path = directory / name
        return path if cv2.imwrite(str(path), frame) else None
