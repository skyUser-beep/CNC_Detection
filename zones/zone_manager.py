import json
from pathlib import Path
import cv2
import numpy as np

class ZoneManager:
    def __init__(self, output_dir, margin_px=60):
        self.zone_dir = Path(output_dir) / "camera_zones"
        self.zone_dir.mkdir(parents=True, exist_ok=True)
        self.margin_px = margin_px

    def file_path(self, camera_id):
        return self.zone_dir / f"{camera_id}_zones.json"

    def save(self, camera_id, zones, width, height):
        data = {"zones": [[[float(x) / width, float(y) / height] for x, y in zone] for zone in zones], "margin_pixels": self.margin_px}
        with open(self.file_path(camera_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, camera_id, width, height):
        path = self.file_path(camera_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            zones = []
            for normalized in data.get("zones", []):
                if len(normalized) >= 3:
                    zones.append(np.array([[int(x * width), int(y * height)] for x, y in normalized], dtype=np.int32))
            return zones or None
        except Exception as exc:
            print(f"Could not load zones for {camera_id}:", exc)
            return None

    def masks(self, zones, width, height):
        result = []
        for zone in zones:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [zone], 255)
            if self.margin_px > 0:
                size = self.margin_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                mask = cv2.dilate(mask, kernel)
            result.append(mask)
        return result

    @staticmethod
    def point_inside(mask, x, y):
        h, w = mask.shape
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        return mask[y, x] > 0

    @classmethod
    def person_zone(cls, masks, x1, y1, x2, y2):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        points = [(cx, y2), (cx, y1 + height * 0.75), (cx, cy),
                  (x1 + width * 0.25, y1 + height * 0.75),
                  (x1 + width * 0.75, y1 + height * 0.75)]
        best_zone = None
        best_score = 0
        for index, mask in enumerate(masks):
            score = sum(cls.point_inside(mask, px, py) for px, py in points)
            if score > best_score:
                best_score = score
                best_zone = index
        return best_zone if best_score >= 2 else None

    @staticmethod
    def draw(frame, zones):
        output = frame.copy()
        for index, zone in enumerate(zones):
            overlay = output.copy()
            cv2.fillPoly(overlay, [zone], (255, 255, 0))
            output = cv2.addWeighted(overlay, 0.12, output, 0.88, 0)
            cv2.polylines(output, [zone], True, (255, 255, 0), 3)
            x, y, _, _ = cv2.boundingRect(zone)
            cv2.putText(output, f"ZONE {index + 1}", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return output

    def setup_interactive(self, camera_id, frame):
        zones = []
        points = []
        window = f"CNC - {camera_id} - Zone Setup"
        state = {"points": points}

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["points"].append((x, y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
        while True:
            display = frame.copy()
            for i, zone in enumerate(zones):
                cv2.polylines(display, [zone], True, (255, 255, 0), 2)
                x, y, _, _ = cv2.boundingRect(zone)
                cv2.putText(display, f"ZONE {i + 1}", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if len(points) > 1:
                cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
            for px, py in points:
                cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(display, f"{camera_id} - ZONE SETUP", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, "Left click = point | ENTER = finish zone", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, "S = save | C = clear | D = delete | Q = cancel", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 13:
                if len(points) >= 3:
                    zones.append(np.array(points, dtype=np.int32))
                    points.clear()
            elif key == ord("c"):
                points.clear()
            elif key == ord("d") and zones:
                zones.pop()
            elif key == ord("s"):
                if len(points) >= 3:
                    zones.append(np.array(points, dtype=np.int32))
                    points.clear()
                if zones:
                    self.save(camera_id, zones, frame.shape[1], frame.shape[0])
                    cv2.destroyWindow(window)
                    return zones
            elif key == ord("q"):
                cv2.destroyWindow(window)
                return None
