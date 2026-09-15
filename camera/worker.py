import threading
import time
import cv2

from detection.person_detector import PersonDetector
from detection.phone_detector import PhoneDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager
from safety.rules import SafetyRules
from display.renderer import Renderer

class CameraWorker(threading.Thread):
    def __init__(self, camera_id, source, config, device, half, phone_detector, event_manager, zone_manager):
        super().__init__(daemon=True, name=f"worker-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.device = device
        self.half = half
        self.phone_detector = phone_detector
        self.event_manager = event_manager
        self.zone_manager = zone_manager
        self.renderer = Renderer(config.display_width)
        self.reader = None
        self.person_detector = PersonDetector(
            config.person_model_path, device, half,
            config.person_confidence, config.person_image_size,
            config.person_use_augment,
        )
        self.zones = None
        self.zone_masks = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.safety = None
        self.running = True
        self.latest_raw = None
        self.latest_frame_number = -1
        self.latest_video_time = 0.0
        self.latest_persons = []
        self.latest_inside = {}
        self.state_lock = threading.Lock()
        self.fps_counter_start = time.perf_counter()
        self.processed_frames = 0
        self.processing_fps = 0.0
        self.startup_ready = threading.Event()
        self.error = None

    def prepare(self):
        from camera.reader import CameraReader
        self.reader = CameraReader(self.camera_id, self.source)
        self.width, self.height, self.fps = self.reader.width, self.reader.height, self.reader.fps
        self.zones = self.zone_manager.load(self.camera_id, self.width, self.height)
        self.reader.start()
        frame = None
        for _ in range(300):
            frame, _, _ = self.reader.get_latest()
            if frame is not None:
                break
            time.sleep(0.01)
        if frame is None:
            self.reader.stop()
            raise RuntimeError(f"{self.camera_id} did not produce a frame")
        if self.zones is None:
            self.zones = self.zone_manager.setup_interactive(self.camera_id, frame)
            if self.zones is None:
                self.reader.stop()
                self.running = False
                return False
        self.zone_masks = self.zone_manager.masks(self.zones, self.width, self.height)
        self.safety = SafetyRules(self.camera_id, len(self.zones), self.event_manager, self.config)
        self.latest_raw = frame
        print(f"{self.camera_id}: {self.width}x{self.height} @ {self.fps:.2f} FPS | {len(self.zones)} zone(s)")
        return True

    def get_display(self):
        frame, frame_number, video_time = self.reader.get_latest()
        if frame is None:
            return None
        with self.state_lock:
            persons = [p.copy() for p in self.latest_persons]
            inside = {k: set(v) for k, v in self.latest_inside.items()}
            processing_fps = self.processing_fps
        return self.renderer.render(
            frame, self.camera_id, video_time, self.zones,
            persons, inside, self.safety, processing_fps,
        )

    def redraw_zones(self):
        frame, _, _ = self.reader.get_latest()
        if frame is None:
            return
        zones = self.zone_manager.setup_interactive(self.camera_id, frame)
        if zones is not None:
            self.zones = zones
            self.zone_masks = self.zone_manager.masks(zones, self.width, self.height)
            self.safety.reset_zones(len(zones))

    def _extract_persons(self, results):
        persons = []
        if not results or len(results) == 0:
            return persons
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return persons
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
        for box, track_id, confidence in zip(xyxy, ids, confs):
            if track_id is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            persons.append({
                "id": int(track_id),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "zone": self.zone_manager.person_zone(self.zone_masks, x1, y1, x2, y2),
                "confidence": float(confidence),
            })
        return persons

    def _check_phones(self, frame, persons, video_time, frame_number):
        if frame_number % self.config.phone_detection_interval != 0:
            return
        for person in persons:
            if person["zone"] is None or not self.safety.phone_allowed(person["id"]):
                continue
            boxes = self.phone_detector.detect_in_person(
                frame, person["x1"], person["y1"], person["x2"], person["y2"]
            )
            if not boxes:
                continue
            shot = ZoneManager.draw(frame.copy(), self.zones)
            cv2.rectangle(shot, (person["x1"], person["y1"]), (person["x2"], person["y2"]), (0, 255, 0), 2)
            cv2.putText(shot, f"ID {person['id']} | ZONE {person['zone'] + 1}", (person["x1"], max(25, person["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for box in boxes:
                cv2.rectangle(shot, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 0, 255), 3)
                cv2.putText(shot, f"PHONE {box['confidence']:.2f}", (box["x1"], max(25, box["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            path = self.event_manager.screenshot(self.camera_id, shot, video_time, "PHONE_DETECTED", {person["id"]}, person["zone"])
            self.event_manager.log(self.camera_id, video_time, "PHONE_DETECTED", person["id"], f"Phone detected inside CNC work zone Zone {person['zone'] + 1}; screenshot={path if path else 'screenshot_failed'}")
            self.safety.mark_phone(person["id"])

    def _process_one(self, frame, frame_number, video_time):
        try:
            results = self.person_detector.track(frame)
            persons = self._extract_persons(results)
            persons, inside = self.safety.update(persons, video_time, len(self.zones))
            self._check_phones(frame, persons, video_time, frame_number)
            with self.state_lock:
                self.latest_persons = [p.copy() for p in persons]
                self.latest_inside = {k: set(v) for k, v in inside.items()}
                self.latest_raw = frame.copy()
                self.latest_video_time = video_time
                self.latest_frame_number = frame_number
                self.processed_frames += 1
                elapsed = time.perf_counter() - self.fps_counter_start
                if elapsed >= 1.0:
                    self.processing_fps = self.processed_frames / elapsed
                    self.processed_frames = 0
                    self.fps_counter_start = time.perf_counter()
        except Exception as exc:
            self.error = exc
            self.running = False
            print(f"[{self.camera_id}] processing error: {exc}")

    def run(self):
        try:
            if self.safety is None:
                return
            last_frame = -1
            while self.running and not self.reader.stop_event.is_set():
                frame, frame_number, video_time = self.reader.get_latest()
                if frame is None:
                    if self.reader.finished:
                        break
                    time.sleep(0.002)
                    continue
                if frame_number == last_frame:
                    if self.reader.finished:
                        break
                    time.sleep(0.001)
                    continue
                last_frame = frame_number
                self._process_one(frame, frame_number, video_time)
                if self.reader.finished and frame_number >= self.reader.current_frame_number - 1:
                    break
        finally:
            self.running = False
            if self.reader is not None:
                self.reader.stop()
