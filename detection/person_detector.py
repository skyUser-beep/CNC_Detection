from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path, device, half, confidence, image_size, augment=False):
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.device = device
        self.half = half
        self.confidence = confidence
        self.image_size = image_size
        self.augment = augment

    def track(self, frame):
        return self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            half=self.half,
            augment=self.augment,
            verbose=False,
        )
