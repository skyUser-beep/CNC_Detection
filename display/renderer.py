import cv2
from events.event_manager import format_timestamp
from zones.zone_manager import ZoneManager

class Renderer:
    def __init__(self, display_width):
        self.display_width = display_width

    def render(self, frame, camera_id, video_time, zones, persons, inside, safety, processing_fps):
        frame = ZoneManager.draw(frame, zones)
        for person in persons:
            x1, y1, x2, y2 = person["x1"], person["y1"], person["x2"], person["y2"]
            zone = person["zone"]
            color = (0, 255, 0) if zone is not None else (0, 165, 255)
            status = f"ZONE {zone + 1}" if zone is not None else "OUTSIDE"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {person['id']} | {status}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        panel_height = max(210, 160 + len(zones) * 30)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (620, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        lines = [
            f"{camera_id} | Video: {format_timestamp(video_time)}",
            f"Processing FPS: {processing_fps:.1f}",
            f"People in zones: {sum(len(v) for v in inside.values())}",
            f"Work zones: {len(zones)}",
        ]
        y = 38
        for line in lines:
            cv2.putText(frame, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y += 27
        for z in range(len(zones)):
            inside_ids = inside.get(z,set())
            count=len(inside_ids)
            cooldown = safety.cooldown_until[z]
            if cooldown is not None and video_time < cooldown:
                text = f"Zone {z + 1}: COOLDOWN {format_timestamp(cooldown - video_time)}"
                color = (0, 165, 255)
            elif count > 1 and safety.multiple_start[z] is not None:
                elapsed = video_time - safety.multiple_start[z]
                text = f"Zone {z + 1}: {count} PEOPLE {elapsed:.0f}/{safety.multiple_limit_seconds}s"
                color = (0, 255, 255)
            else:
                text = f"Zone {z + 1}: {count}/1"
                color = (0, 255, 0)
            cv2.putText(frame, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 25
        cv2.putText(frame, "Click window = select | Z = redraw zones | Q = quit", (25, min(frame.shape[0] - 10, y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        if frame.shape[1] > self.display_width:
            scale = self.display_width / frame.shape[1]
            frame = cv2.resize(frame, (self.display_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        return frame
