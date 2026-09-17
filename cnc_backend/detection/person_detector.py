from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path, device, half, confidence, image_size, augment=False, tracker_config="botsort.yaml"):
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
        self.tracker_config = tracker_config

    def track(self, frame):
        return self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            half=self.half,
            augment=self.augment,
            verbose=False,
        )
