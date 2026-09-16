from pathlib import Path
import cv2

def format_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    return (f"{int(seconds // 3600):02d}:"
        f"{int((seconds % 3600) // 60):02d}:"
        f"{int(seconds % 60):02d}"
    )

class EventManager:
    def __init__(self, output_dir):
        self.base = Path(output_dir) / "cameras"
        self.base.mkdir(parents=True, exist_ok=True)

    def camera_dir(self, camera_id):
        path = self.base / camera_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self,camera_id,video_seconds,event_name,track_ids="",details="" ):
        timestamp = format_timestamp(video_seconds)

        print(f"\nSAFETY EVENT\n Camera    : {camera_id}\n Timestamp : {timestamp}\n Event     : {event_name}")

        if track_ids:
            print("Track IDs :", track_ids)
        if details:
            print("Details   :", details)
        path = self.camera_dir(camera_id) / "events.csv"

        if not path.exists():
            path.write_text("timestamp,video_time,event,track_ids,details\n",encoding="utf-8")

        def safe(value):
            return str(value).replace(",", ";")

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{safe(timestamp)},"
                f"{video_seconds:.2f},"
                f"{safe(event_name)},"
                f"{safe(track_ids)},"
                f"{safe(details)}\n"
            )

    def screenshot(self,camera_id,frame,video_time, event_type,track_ids, zone=None):
        # Use the same directory structure as camera_dir()
        camera_dir = self.camera_dir(camera_id)

        screenshot_dir = camera_dir / "screenshots"
        screenshot_dir.mkdir(parents=True,exist_ok=True)
        # format_timestamp() is a standalone function
        timestamp = format_timestamp(video_time).replace(":","-")
        # Make sure track_ids is iterable
        if isinstance(track_ids, (int, str)):
            track_ids = {track_ids}

        ids_text = "_".join(str(i) for i in sorted(track_ids, key=str))

        filename = f"{timestamp}_{event_type}_IDs_{ids_text}"
        if zone is not None:
            filename += f"_Zone_{zone + 1}"

        filename += ".jpg"
        path = screenshot_dir / filename

        success = cv2.imwrite(str(path),frame)

        if not success:
            raise RuntimeError(f"cv2.imwrite failed. Could not save screenshot: {path}")

        print(f"SCREENSHOT SAVED: {path}")
        return path