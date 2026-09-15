import threading
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, model_path, device, half, confidence, image_size, class_id):
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.device = device
        self.half = half
        self.confidence = confidence
        self.image_size = image_size
        self.class_id = class_id
        self.lock = threading.Lock()

    def detect_in_person(self, frame, x1, y1, x2, y2, padding=40):
        h, w = frame.shape[:2]
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(w, x2 + padding)
        crop_y2 = min(h, y2 + padding)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return []
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return []
        try:
            with self.lock:
                results = self.model.predict(
                    crop,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    classes=[self.class_id],
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )
        except Exception as exc:
            print("Phone detection error:", exc)
            return []
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        boxes = []
        for i, box in enumerate(xyxy):
            px1, py1, px2, py2 = map(int, box)
            boxes.append({
                "x1": px1 + crop_x1,
                "y1": py1 + crop_y1,
                "x2": px2 + crop_x1,
                "y2": py2 + crop_y1,
                "confidence": float(conf[i]) if len(conf) > i else self.confidence,
            })
        return boxes
