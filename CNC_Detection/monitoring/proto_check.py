import sys
from pathlib import Path
import cv2
import numpy as np
import json
import time
import threading
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

VIDEO_PATH = config.VIDEO_PATH
PERSON_MODEL_PATH = config.MODEL_PATH
PHONE_MODEL_PATH = config.MODEL_PATH

CAMERA_ID = getattr(config, "CAMERA_ID", "camera_04")

OUTPUT_DIR = Path(config.OUTPUTS_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZONE_DIR = OUTPUT_DIR / "camera_zones"
ZONE_DIR.mkdir(parents=True, exist_ok=True)

ZONE_FILE = ZONE_DIR / f"{CAMERA_ID}_zones.json"

EVENT_LOG_FILE = OUTPUT_DIR / "cnc_events.csv"

SCREENSHOT_DIR = OUTPUT_DIR / "cnc_screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

PERSON_CONFIDENCE = 0.20
PHONE_CONFIDENCE = 0.15

PERSON_IMAGE_SIZE = 640
PHONE_IMAGE_SIZE = 512

PERSON_DETECTION_INTERVAL = 1
PHONE_DETECTION_INTERVAL = 3

PHONE_CLASS_ID = 67

MULTIPLE_PERSON_LIMIT_SECONDS = 120
MULTIPLE_PERSON_COOLDOWN_SECONDS = 1800

ABSENCE_LIMIT_SECONDS = 300

ZONE_MARGIN_PX = 60

TRACK_GRACE_SECONDS = 1.5
INSIDE_GRACE_SECONDS = 1.5

WINDOW_NAME = "CNC Safety Monitoring"
DISPLAY_WIDTH = 1280

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = 0
        USE_HALF = True
        print("CUDA GPU DETECTED")
        print("GPU:", torch.cuda.get_device_name(0))
        print("Using FP16:", USE_HALF)
    else:
        DEVICE = "cpu"
        USE_HALF = False
        print("CUDA NOT AVAILABLE")
        print("Using CPU")

except Exception:
    DEVICE = "cpu"
    USE_HALF = False
    print("PyTorch CUDA check failed")
    print("Using CPU")

latest_frame = None
latest_frame_number = -1
latest_frame_time = 0.0

reader_finished = False
reader_lock = threading.Lock()
reader_stop_event = threading.Event()


def format_timestamp(seconds):
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def initialize_event_log():
    if not EVENT_LOG_FILE.exists():
        with open(EVENT_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("timestamp,video_time,event,track_ids,details\n")

def log_event(video_seconds, event_name, track_ids="", details=""):
    timestamp = format_timestamp(video_seconds)
    print()
    print("SAFETY EVENT")
    print("Timestamp :", timestamp)
    print("Event     :", event_name)
    if track_ids:
        print("Track IDs :", track_ids)
    if details:
        print("Details   :", details)
    with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
        safe_event = str(event_name).replace(",", ";")
        safe_ids = str(track_ids).replace(",", ";")
        safe_details = str(details).replace(",", ";")
        f.write(f"{timestamp},{video_seconds:.2f},{safe_event},{safe_ids},{safe_details}\n")

class LatestFrameReader(threading.Thread):
    def __init__(self, video_path):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 25.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_interval = 1.0 / self.fps
        self.current_frame_number = 0

    def run(self):
        global latest_frame
        global latest_frame_number
        global latest_frame_time
        global reader_finished

        next_frame_time = time.perf_counter()
        while not reader_stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                reader_finished = True
                break
            frame_number = self.current_frame_number
            self.current_frame_number += 1
            video_time = frame_number / self.fps
            with reader_lock:
                latest_frame = frame
                latest_frame_number = frame_number
                latest_frame_time = video_time

            next_frame_time += self.frame_interval
            sleep_time = next_frame_time- time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.perf_counter()
        self.cap.release()


def save_zones(zones, width, height):
    normalized_zones = []

    for zone in zones:
        normalized_points = []
        for x, y in zone:
            normalized_points.append([float(x) / width,float(y) / height])

        normalized_zones.append( normalized_points)

    data = {"zones": normalized_zones,"margin_pixels": ZONE_MARGIN_PX}
    with open(ZONE_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=4)

    print()
    print("Zones saved:")
    print(ZONE_FILE)
    print("Camera:", CAMERA_ID)
    print("Number of zones:", len(zones))

def load_zones(width, height):
    if not ZONE_FILE.exists():
        return []
    try:
        with open(ZONE_FILE, "r",encoding="utf-8") as f:
            data = json.load(f)
        normalized_zones = data.get("zones",[])
        zones = []
        for normalized_points in normalized_zones:
            if len(normalized_points) < 3:
                continue
            points = []
            for x, y in normalized_points:
                points.append([int(x * width), int(y * height) ])
            zones.append(
                np.array(
                    points,
                    dtype=np.int32
                )
            )

        return zones

    except Exception as e:

        print(
            "Could not load zones:",
            e
        )

        return []


def draw_zone_setup(
    frame,
    zones,
    current_points
):

    display_frame = frame.copy()

    for index, zone in enumerate(zones):

        cv2.polylines(
            display_frame,
            [zone],
            True,
            (255, 255, 0),
            2
        )

        center = np.mean(
            zone,
            axis=0
        ).astype(int)

        cv2.putText(
            display_frame,
            f"ZONE {index + 1}",
            (
                int(center[0]),
                int(center[1])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    if len(current_points) > 0:

        current_array = np.array(
            current_points,
            dtype=np.int32
        )

        if len(current_points) > 1:

            cv2.polylines(
                display_frame,
                [current_array],
                False,
                (0, 255, 255),
                2
            )

        for px, py in current_points:

            cv2.circle(
                display_frame,
                (px, py),
                6,
                (0, 255, 255),
                -1
            )

    cv2.putText(
        display_frame,
        f"Zones: {len(zones)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        "ENTER: finish zone | S: save | D: delete | C: clear | Q: cancel",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return display_frame


def create_zones_interactively(frame):

    zones = []
    current_points = []

    print()
    print("MULTIPLE CNC ZONE SETUP")
    print()
    print("Left click : Add point")
    print("ENTER      : Finish current zone")
    print("S          : Save all zones")
    print("C          : Clear current zone")
    print("D          : Delete last zone")
    print("Q          : Cancel")
    print()
    print(
        f"Extra boundary margin: "
        f"{ZONE_MARGIN_PX} pixels"
    )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            current_points.append(
                (x, y)
            )

            print(
                f"Current zone point "
                f"{len(current_points)}: "
                f"({x}, {y})"
            )

    cv2.namedWindow(
        WINDOW_NAME
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    while True:

        display_frame = draw_zone_setup(
            frame,
            zones,
            current_points
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 13:

            if len(current_points) >= 3:

                zone = np.array(
                    current_points,
                    dtype=np.int32
                )

                zones.append(zone)

                print(
                    f"Zone {len(zones)} created."
                )

                current_points.clear()

            else:

                print(
                    "Need at least 3 points."
                )

        elif key == ord("s"):

            if len(current_points) >= 3:

                zone = np.array(
                    current_points,
                    dtype=np.int32
                )

                zones.append(zone)

                current_points.clear()

                print(
                    f"Zone {len(zones)} created."
                )

            if len(zones) > 0:

                save_zones(
                    zones,
                    frame.shape[1],
                    frame.shape[0]
                )

                cv2.destroyWindow(
                    WINDOW_NAME
                )

                return zones

            print(
                "No zones created."
            )

        elif key == ord("c"):

            current_points.clear()

            print(
                "Current zone cleared."
            )

        elif key == ord("d"):

            if len(zones) > 0:

                deleted_zone = zones.pop()

                print(
                    f"Zone {len(zones) + 1} deleted."
                )

            else:

                print(
                    "No completed zone to delete."
                )

        elif key == ord("q"):

            cv2.destroyWindow(
                WINDOW_NAME
            )

            return None

    return zones


def create_expanded_zone_mask(
    zones,
    width,
    height
):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for zone in zones:

        cv2.fillPoly(
            mask,
            [zone],
            255
        )

    if ZONE_MARGIN_PX <= 0:
        return mask

    kernel_size = (
        ZONE_MARGIN_PX * 2 + 1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size
        )
    )

    expanded_mask = cv2.dilate(
        mask,
        kernel
    )

    return expanded_mask


def person_is_inside_zone(
    expanded_zone_mask,
    x1,
    y1,
    x2,
    y2
):

    center_x = int(
        (x1 + x2) / 2
    )

    bottom_y = int(y2)

    h, w = expanded_zone_mask.shape

    center_x = max(
        0,
        min(
            w - 1,
            center_x
        )
    )

    bottom_y = max(
        0,
        min(
            h - 1,
            bottom_y
        )
    )

    return (
        expanded_zone_mask[
            bottom_y,
            center_x
        ] > 0
    )


def draw_zone(
    frame,
    zones,
    expanded_zone_mask
):

    if not zones:
        return

    contours, _ = cv2.findContours(
        expanded_zone_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        frame,
        contours,
        -1,
        (0, 255, 255),
        2
    )

    for index, zone in enumerate(zones):

        cv2.polylines(
            frame,
            [zone],
            True,
            (255, 255, 0),
            2
        )

        center = np.mean(
            zone,
            axis=0
        ).astype(int)

        cv2.putText(
            frame,
            f"ZONE {index + 1}",
            (
                int(center[0]),
                int(center[1])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


def detect_phone_in_person(
    phone_model,
    frame,
    x1,
    y1,
    x2,
    y2
):

    h, w = frame.shape[:2]

    padding = 40

    crop_x1 = max(
        0,
        x1 - padding
    )

    crop_y1 = max(
        0,
        y1 - padding
    )

    crop_x2 = min(
        w,
        x2 + padding
    )

    crop_y2 = min(
        h,
        y2 + padding
    )

    if crop_x2 <= crop_x1:
        return False

    if crop_y2 <= crop_y1:
        return False

    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    if crop.size == 0:
        return False

    try:

        results = phone_model.predict(
            crop,
            imgsz=PHONE_IMAGE_SIZE,
            conf=PHONE_CONFIDENCE,
            classes=[PHONE_CLASS_ID],
            device=DEVICE,
            half=USE_HALF,
            verbose=False
        )

    except Exception as e:

        print(
            "Phone detection error:",
            e
        )

        return False

    if not results:
        return False

    result = results[0]

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    return True


def save_event_screenshot(
    frame,
    video_seconds,
    event_name,
    track_ids
):

    timestamp = format_timestamp(
        video_seconds
    ).replace(":", "-")

    ids_text = "_".join(
        str(x)
        for x in sorted(track_ids)
    )

    filename = (
        f"{timestamp}_"
        f"{event_name}_"
        f"IDs_{ids_text}.jpg"
    )

    screenshot_path = (
        SCREENSHOT_DIR /
        filename
    )

    success = cv2.imwrite(
        str(screenshot_path),
        frame
    )

    if success:

        print(
            "Screenshot saved:",
            screenshot_path
        )

        return screenshot_path

    print(
        "Failed to save screenshot:",
        screenshot_path
    )

    return None


def main():

    global latest_frame

    initialize_event_log()

    print()
    print(
        "Loading person model..."
    )

    person_model = YOLO(
        PERSON_MODEL_PATH
    )

    print(
        "Loading phone model..."
    )

    phone_model = YOLO(
        PHONE_MODEL_PATH
    )

    reader = LatestFrameReader(
        VIDEO_PATH
    )

    fps = reader.fps
    width = reader.width
    height = reader.height
    total_frames = reader.total_frames

    print()
    print(
        "VIDEO INFORMATION"
    )

    print(
        "Camera     :",
        CAMERA_ID
    )

    print(
        "Resolution :",
        width,
        "x",
        height
    )

    print(
        "FPS        :",
        fps
    )

    print(
        "Frames     :",
        total_frames
    )

    if total_frames > 0:

        print(
            "Duration   :",
            format_timestamp(
                total_frames / fps
            )
        )

    zones = load_zones(
        width,
        height
    )

    reader.start()

    print()
    print(
        "Waiting for first video frame..."
    )

    while latest_frame is None:

        if reader_finished:

            print(
                "Video ended before first frame."
            )

            return

        time.sleep(
            0.01
        )

    if not zones:

        with reader_lock:

            setup_frame = (
                latest_frame.copy()
            )

        zones = create_zones_interactively(
            setup_frame
        )

        if zones is None:

            reader_stop_event.set()

            print(
                "Zone setup cancelled."
            )

            return

    expanded_zone_mask = (
        create_expanded_zone_mask(
            zones,
            width,
            height
        )
    )

    print()
    print(
        "Loaded/created zones:",
        len(zones)
    )

    print(
        f"Expanded zone margin: "
        f"{ZONE_MARGIN_PX}px"
    )

    track_states = {}

    last_processed_frame_number = -1

    last_person_results = None

    processing_fps = 0.0

    fps_counter_start = (
        time.perf_counter()
    )

    fps_counter_frames = 0

    multiple_person_start = None

    multiple_person_logged = False

    multiple_person_cooldown_until = None

    last_phone_frame_number = -1

    number_inside = 0

    event_screenshot_required = False
    event_screenshot_ids = set()
    event_screenshot_time = 0.0

    try:

        while True:

            with reader_lock:

                if latest_frame is None:

                    current_frame = None
                    current_frame_number = -1
                    current_video_time = 0.0

                else:

                    current_frame = (
                        latest_frame.copy()
                    )

                    current_frame_number = (
                        latest_frame_number
                    )

                    current_video_time = (
                        latest_frame_time
                    )

            if current_frame is None:

                if reader_finished:
                    break

                time.sleep(
                    0.005
                )

                continue

            if (
                current_frame_number
                ==
                last_processed_frame_number
            ):

                if reader_finished:
                    break

                time.sleep(
                    0.001
                )

                continue

            last_processed_frame_number = (
                current_frame_number
            )

            frame = current_frame

            run_person_detection = (
                current_frame_number
                %
                PERSON_DETECTION_INTERVAL
                ==
                0
            )

            if run_person_detection:

                inference_start = (
                    time.perf_counter()
                )

                try:

                    results = (
                        person_model.track(
                            frame,
                            persist=True,
                            tracker="botsort.yaml",
                            classes=[0],
                            conf=PERSON_CONFIDENCE,
                            imgsz=PERSON_IMAGE_SIZE,
                            device=DEVICE,
                            half=USE_HALF,
                            verbose=False
                        )
                    )

                    last_person_results = (
                        results
                    )

                except Exception as e:

                    print(
                        "Person tracking error:",
                        e
                    )

                    results = (
                        last_person_results
                    )

                inference_time = (
                    time.perf_counter()
                    -
                    inference_start
                )

            else:

                results = (
                    last_person_results
                )

            fps_counter_frames += 1

            fps_elapsed = (
                time.perf_counter()
                -
                fps_counter_start
            )

            if fps_elapsed >= 1.0:

                processing_fps = (
                    fps_counter_frames
                    /
                    fps_elapsed
                )

                fps_counter_frames = 0

                fps_counter_start = (
                    time.perf_counter()
                )

            current_inside_ids = set()

            current_persons = []

            if (
                results is not None
                and
                len(results) > 0
            ):

                result = results[0]

                if (
                    result.boxes is not None
                    and
                    len(result.boxes) > 0
                ):

                    boxes = result.boxes

                    xyxy = (
                        boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    if boxes.id is not None:

                        track_ids = (
                            boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                    else:

                        track_ids = [
                            None
                            for _ in range(
                                len(xyxy)
                            )
                        ]

                    for box, track_id in zip(
                        xyxy,
                        track_ids
                    ):

                        x1, y1, x2, y2 = map(
                            int,
                            box
                        )

                        if track_id is None:
                            continue

                        track_id = int(
                            track_id
                        )

                        inside = (
                            person_is_inside_zone(
                                expanded_zone_mask,
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        )

                        if inside:

                            current_inside_ids.add(
                                track_id
                            )

                        current_persons.append({
                            "id": track_id,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "inside": inside
                        })

            for person in current_persons:

                track_id = person["id"]

                inside = person["inside"]

                if track_id not in track_states:

                    track_states[track_id] = {
                        "last_seen":
                            current_video_time,

                        "last_inside":
                            current_video_time
                            if inside
                            else None,

                        "absence_logged":
                            False,

                        "phone_logged":
                            False
                    }

                state = track_states[
                    track_id
                ]

                state["last_seen"] = (
                    current_video_time
                )

                if inside:

                    state["last_inside"] = (
                        current_video_time
                    )

                    state["absence_logged"] = (
                        False
                    )

                else:

                    if (
                        state["last_inside"]
                        is not None
                    ):

                        away_duration = (
                            current_video_time
                            -
                            state["last_inside"]
                        )

                        if (
                            away_duration
                            >=
                            ABSENCE_LIMIT_SECONDS
                            and
                            not state[
                                "absence_logged"
                            ]
                        ):

                            log_event(
                                current_video_time,
                                "PERSON_AWAY_OVER_5_MINUTES",
                                str(track_id),
                                (
                                    f"Track {track_id} "
                                    f"outside zone for "
                                    f"{away_duration:.1f} "
                                    f"seconds"
                                )
                            )

                            state[
                                "absence_logged"
                            ] = True

            effective_inside_ids = set()

            for (
                track_id,
                state
            ) in track_states.items():

                time_since_seen = (
                    current_video_time
                    -
                    state["last_seen"]
                )

                if (
                    state["last_inside"]
                    is not None
                ):

                    time_since_inside = (
                        current_video_time
                        -
                        state["last_inside"]
                    )

                else:

                    time_since_inside = (
                        float("inf")
                    )

                if (
                    time_since_seen
                    <=
                    TRACK_GRACE_SECONDS
                    and
                    time_since_inside
                    <=
                    INSIDE_GRACE_SECONDS
                ):

                    effective_inside_ids.add(
                        track_id
                    )

            number_inside = len(
                effective_inside_ids
            )

            cooldown_active = False

            if (
                multiple_person_cooldown_until
                is not None
            ):

                if (
                    current_video_time
                    <
                    multiple_person_cooldown_until
                ):

                    cooldown_active = True

                else:

                    multiple_person_cooldown_until = (
                        None
                    )

                    multiple_person_logged = (
                        False
                    )

            if cooldown_active:

                multiple_person_start = None

            else:

                if number_inside > 1:

                    if (
                        multiple_person_start
                        is None
                    ):

                        multiple_person_start = (
                            current_video_time
                        )

                        multiple_person_logged = (
                            False
                        )

                        print(
                            "[INFO] Multiple people "
                            "detected. Timer started at "
                            f"{format_timestamp(current_video_time)}"
                        )

                    multiple_duration = (
                        current_video_time
                        -
                        multiple_person_start
                    )

                    if (
                        multiple_duration
                        >=
                        MULTIPLE_PERSON_LIMIT_SECONDS
                        and
                        not multiple_person_logged
                    ):

                        multiple_person_logged = (
                            True
                        )

                        event_ids = set(
                            effective_inside_ids
                        )

                        ids_text = "|".join(
                            str(x)
                            for x in sorted(
                                event_ids
                            )
                        )

                        cooldown_end = (
                            current_video_time
                            +
                            MULTIPLE_PERSON_COOLDOWN_SECONDS
                        )

                        multiple_person_cooldown_until = (
                            cooldown_end
                        )

                        multiple_person_start = (
                            None
                        )

                        event_details = (
                            f"{number_inside} people "
                            f"inside CNC zone for "
                            f"{multiple_duration:.1f} seconds"
                        )

                        event_screenshot_required = (
                            True
                        )

                        event_screenshot_ids = (
                            event_ids
                        )

                        event_screenshot_time = (
                            current_video_time
                        )

                        log_event(
                            current_video_time,
                            "MULTIPLE_PEOPLE_OVER_2_MINUTES",
                            ids_text,
                            (
                                event_details
                                +
                                "; "
                                +
                                "30 minute cooldown started"
                            )
                        )

                    else:

                        event_screenshot_required = (
                            False
                        )

                        event_screenshot_ids = set()

                        event_screenshot_time = (
                            0.0
                        )

                else:

                    multiple_person_start = (
                        None
                    )

                    multiple_person_logged = (
                        False
                    )

                    event_screenshot_required = (
                        False
                    )

                    event_screenshot_ids = set()

                    event_screenshot_time = (
                        0.0
                    )

            if cooldown_active:

                event_screenshot_required = (
                    False
                )

                event_screenshot_ids = set()

                event_screenshot_time = (
                    0.0
                )

            run_phone_detection = (
                current_frame_number
                %
                PHONE_DETECTION_INTERVAL
                ==
                0
            )

            if run_phone_detection:

                for person in current_persons:

                    if not person["inside"]:
                        continue

                    track_id = person["id"]

                    if (
                        track_id
                        in
                        track_states
                        and
                        track_states[
                            track_id
                        ]["phone_logged"]
                    ):

                        continue

                    phone_found = (
                        detect_phone_in_person(
                            phone_model,
                            frame,
                            person["x1"],
                            person["y1"],
                            person["x2"],
                            person["y2"]
                        )
                    )

                    if phone_found:

                        log_event(
                            current_video_time,
                            "PHONE_DETECTED",
                            str(track_id),
                            "Phone detected inside CNC work zone"
                        )

                        track_states[
                            track_id
                        ]["phone_logged"] = True

            states_to_remove = []

            for (
                track_id,
                state
            ) in track_states.items():

                time_since_seen = (
                    current_video_time
                    -
                    state["last_seen"]
                )

                if time_since_seen > 600:

                    states_to_remove.append(
                        track_id
                    )

            for track_id in states_to_remove:

                del track_states[
                    track_id
                ]

            draw_zone(
                frame,
                zones,
                expanded_zone_mask
            )

            for person in current_persons:

                x1 = person["x1"]
                y1 = person["y1"]
                x2 = person["x2"]
                y2 = person["y2"]

                track_id = person["id"]

                inside = person["inside"]

                if inside:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    status = "INSIDE"

                else:

                    box_color = (
                        0,
                        165,
                        255
                    )

                    status = "OUTSIDE"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                label = (
                    f"ID {track_id} | "
                    f"{status}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

            panel_height = 170

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (560, panel_height),
                (0, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0
            )

            cv2.putText(
                frame,
                (
                    "Video: "
                    +
                    format_timestamp(
                        current_video_time
                    )
                ),
                (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Processing FPS: {processing_fps:.1f}",
                (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"People in zone: {number_inside}",
                (25, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if cooldown_active:

                remaining = max(
                    0,
                    multiple_person_cooldown_until
                    -
                    current_video_time
                )

                cv2.putText(
                    frame,
                    (
                        "Multiple cooldown: "
                        +
                        format_timestamp(
                            remaining
                        )
                    ),
                    (25, 119),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "New timer available after cooldown",
                    (25, 146),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2
                )

            elif multiple_person_start is not None:

                multiple_elapsed = (
                    current_video_time
                    -
                    multiple_person_start
                )

                cv2.putText(
                    frame,
                    (
                        f"Multiple timer: "
                        f"{multiple_elapsed:.0f}/"
                        f"{MULTIPLE_PERSON_LIMIT_SECONDS}s"
                    ),
                    (25, 119),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Waiting for 2 minute violation",
                    (25, 146),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Multiple timer: READY",
                    (25, 119),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "At least 2 people required",
                    (25, 146),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            if event_screenshot_required:

                screenshot_path = (
                    save_event_screenshot(
                        frame,
                        event_screenshot_time,
                        "MULTIPLE_PEOPLE_OVER_2_MINUTES",
                        event_screenshot_ids
                    )
                )

                event_screenshot_required = False

                if screenshot_path is not None:

                    print(
                        "30 minute cooldown started."
                    )

                    print(
                        "Next multiple-person "
                        "timer can start at:",
                        format_timestamp(
                            multiple_person_cooldown_until
                        )
                    )

            if frame.shape[1] > DISPLAY_WIDTH:

                scale = (
                    DISPLAY_WIDTH
                    /
                    frame.shape[1]
                )

                display_frame = cv2.resize(
                    frame,
                    (
                        DISPLAY_WIDTH,
                        int(
                            frame.shape[0]
                            *
                            scale
                        )
                    ),
                    interpolation=cv2.INTER_AREA
                )

            else:

                display_frame = frame

            cv2.imshow(
                WINDOW_NAME,
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("z"):

                print()
                print(
                    "Redefining CNC zones..."
                )

                new_zones = (
                    create_zones_interactively(
                        frame
                    )
                )

                if new_zones is not None:

                    zones = new_zones

                    expanded_zone_mask = (
                        create_expanded_zone_mask(
                            zones,
                            width,
                            height
                        )
                    )

                    print(
                        f"{len(zones)} zones activated."
                    )

            elif key == ord("c"):

                print()
                print(
                    "C pressed. "
                    "To permanently redefine "
                    "the zones, press Z."
                )

            if reader_finished:

                with reader_lock:

                    latest_number = (
                        latest_frame_number
                    )

                if (
                    latest_number
                    <=
                    current_frame_number
                ):

                    cv2.waitKey(100)

                    break

    finally:

        reader_stop_event.set()

        if reader.is_alive():

            reader.join(
                timeout=1.0
            )

        cv2.destroyAllWindows()

    print()
    print(
        "CNC MONITORING FINISHED"
    )

    print(
        "Event log:",
        EVENT_LOG_FILE
    )

    print(
        "Screenshots:",
        SCREENSHOT_DIR
    )


if __name__ == "__main__":
    main()