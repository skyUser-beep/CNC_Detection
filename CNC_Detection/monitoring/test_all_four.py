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

PERSON_MODEL_PATH = config.MODEL_PATH
PHONE_MODEL_PATH = getattr(config, "PHONE_MODEL_PATH", config.MODEL_PATH)

OUTPUT_DIR = Path(config.OUTPUTS_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZONE_DIR = OUTPUT_DIR / "camera_zones"
ZONE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_OUTPUT_DIR = OUTPUT_DIR / "cameras"
CAMERA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERSON_CONFIDENCE = 0.20
PHONE_CONFIDENCE = 0.15

PERSON_IMAGE_SIZE = 512
PHONE_IMAGE_SIZE = 512

PERSON_DETECTION_INTERVAL = 2
PHONE_DETECTION_INTERVAL = 6

PHONE_CLASS_ID = 67

MULTIPLE_PERSON_LIMIT_SECONDS = 120
MULTIPLE_PERSON_COOLDOWN_SECONDS = 1800

ABSENCE_LIMIT_SECONDS = 300

ZONE_MARGIN_PX = 60

TRACK_GRACE_SECONDS = 1.5
INSIDE_GRACE_SECONDS = 1.5

DISPLAY_WIDTH = 900

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


class CameraReader(threading.Thread):

    def __init__(self, camera_id, source):
        super().__init__(daemon=True)

        self.camera_id = camera_id
        self.source = source

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open {camera_id}: {source}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 25.0

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.latest_frame = None
        self.latest_frame_number = -1
        self.latest_frame_time = 0.0

        self.finished = False

        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.current_frame_number = 0

    def run(self):

        next_frame_time = time.perf_counter()

        while not self.stop_event.is_set():

            ret, frame = self.cap.read()

            if not ret:
                self.finished = True
                break

            frame_number = self.current_frame_number
            self.current_frame_number += 1

            video_time = frame_number / self.fps

            with self.lock:

                self.latest_frame = frame
                self.latest_frame_number = frame_number
                self.latest_frame_time = video_time

            next_frame_time += 1.0 / self.fps

            sleep_time = (
                next_frame_time -
                time.perf_counter()
            )

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.perf_counter()

        self.cap.release()

    def get_latest(self):

        with self.lock:

            if self.latest_frame is None:
                return None, -1, 0.0

            return (
                self.latest_frame.copy(),
                self.latest_frame_number,
                self.latest_frame_time
            )

    def stop(self):
        self.stop_event.set()


def format_timestamp(seconds):

    seconds = max(0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def get_zone_file(camera_id):

    return (
        ZONE_DIR /
        f"{camera_id}_zones.json"
    )


def get_camera_output_dir(camera_id):

    directory = (
        CAMERA_OUTPUT_DIR /
        camera_id
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


def initialize_event_log(camera_id):

    camera_dir = get_camera_output_dir(
        camera_id
    )

    event_file = (
        camera_dir /
        "events.csv"
    )

    if not event_file.exists():

        with open(
            event_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "timestamp,video_time,event,"
                "track_ids,details\n"
            )

    return event_file


def log_event(
    camera_id,
    video_seconds,
    event_name,
    track_ids="",
    details=""
):

    timestamp = format_timestamp(
        video_seconds
    )

    print()
    print("SAFETY EVENT")
    print("Camera    :", camera_id)
    print("Timestamp :", timestamp)
    print("Event     :", event_name)

    if track_ids:
        print("Track IDs :", track_ids)

    if details:
        print("Details   :", details)

    event_file = initialize_event_log(
        camera_id
    )

    with open(
        event_file,
        "a",
        encoding="utf-8"
    ) as f:

        safe_event = (
            str(event_name)
            .replace(",", ";")
        )

        safe_ids = (
            str(track_ids)
            .replace(",", ";")
        )

        safe_details = (
            str(details)
            .replace(",", ";")
        )

        f.write(
            f"{timestamp},"
            f"{video_seconds:.2f},"
            f"{safe_event},"
            f"{safe_ids},"
            f"{safe_details}\n"
        )


def save_zones(
    camera_id,
    zones,
    width,
    height
):

    normalized_zones = []

    for zone in zones:

        normalized_points = []

        for x, y in zone:

            normalized_points.append(
                [
                    float(x) / width,
                    float(y) / height
                ]
            )

        normalized_zones.append(
            normalized_points
        )

    data = {
        "zones": normalized_zones,
        "margin_pixels": ZONE_MARGIN_PX
    }

    zone_file = get_zone_file(
        camera_id
    )

    with open(
        zone_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    print()
    print(
        f"{camera_id} zones saved:"
    )
    print(zone_file)


def load_zones(
    camera_id,
    width,
    height
):

    zone_file = get_zone_file(
        camera_id
    )

    if not zone_file.exists():
        return None

    try:

        with open(
            zone_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        normalized_zones = data.get(
            "zones",
            []
        )

        zones = []

        for normalized_points in normalized_zones:

            if len(normalized_points) < 3:
                continue

            points = []

            for x, y in normalized_points:

                points.append(
                    [
                        int(x * width),
                        int(y * height)
                    ]
                )

            zones.append(
                np.array(
                    points,
                    dtype=np.int32
                )
            )

        if not zones:
            return None

        return zones

    except Exception as e:

        print(
            f"Could not load zones for "
            f"{camera_id}:",
            e
        )

        return None


def create_expanded_zone_mask(
    zones,
    width,
    height
):

    masks = []

    for zone in zones:

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [zone],
            255
        )

        if ZONE_MARGIN_PX > 0:

            kernel_size = (
                ZONE_MARGIN_PX * 2 + 1
            )

            kernel = (
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        kernel_size,
                        kernel_size
                    )
                )
            )

            mask = cv2.dilate(
                mask,
                kernel
            )

        masks.append(mask)

    return masks


def draw_setup_frame(
    frame,
    zones,
    current_points,
    camera_id
):

    output = frame.copy()

    for index, zone in enumerate(zones):

        cv2.polylines(
            output,
            [zone],
            True,
            (255, 255, 0),
            2
        )

        x, y, w, h = (
            cv2.boundingRect(zone)
        )

        cv2.putText(
            output,
            f"ZONE {index + 1}",
            (
                x,
                max(25, y - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    if len(current_points) > 1:

        points_array = np.array(
            current_points,
            dtype=np.int32
        )

        cv2.polylines(
            output,
            [points_array],
            False,
            (0, 255, 255),
            2
        )

    for px, py in current_points:

        cv2.circle(
            output,
            (px, py),
            6,
            (0, 255, 255),
            -1
        )

    cv2.putText(
        output,
        f"{camera_id} - ZONE SETUP",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Left click = point | ENTER = finish zone",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "S = save | C = clear | D = delete | Q = cancel",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output


def create_zones_interactively(
    camera_id,
    frame
):

    zones = []
    current_points = []

    window_name = (
        f"CNC - {camera_id} - Zone Setup"
    )

    state = {
        "points": current_points
    }

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            state["points"].append(
                (x, y)
            )

            print(
                f"{camera_id} zone point "
                f"{len(state['points'])}: "
                f"({x}, {y})"
            )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print()
    print(
        f"ZONE SETUP: {camera_id}"
    )

    while True:

        display = draw_setup_frame(
            frame,
            zones,
            state["points"],
            camera_id
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 13:

            if len(state["points"]) >= 3:

                zone = np.array(
                    state["points"],
                    dtype=np.int32
                )

                zones.append(zone)

                print(
                    f"{camera_id}: "
                    f"Zone {len(zones)} completed."
                )

                state["points"].clear()

            else:

                print(
                    "Need at least 3 points."
                )

        elif key == ord("c"):

            state["points"].clear()

            print(
                "Current zone cleared."
            )

        elif key == ord("d"):

            if zones:

                zones.pop()

                print(
                    "Last zone deleted."
                )

        elif key == ord("s"):

            if len(state["points"]) >= 3:

                zone = np.array(
                    state["points"],
                    dtype=np.int32
                )

                zones.append(zone)

                state["points"].clear()

            if zones:

                save_zones(
                    camera_id,
                    zones,
                    frame.shape[1],
                    frame.shape[0]
                )

                cv2.destroyWindow(
                    window_name
                )

                return zones

            print(
                "No zones to save."
            )

        elif key == ord("q"):

            cv2.destroyWindow(
                window_name
            )

            return None


def point_inside_zone(
    mask,
    x,
    y
):

    h, w = mask.shape

    x = max(
        0,
        min(w - 1, int(x))
    )

    y = max(
        0,
        min(h - 1, int(y))
    )

    return (
        mask[y, x] > 0
    )


def get_person_zone(
    zone_masks,
    x1,
    y1,
    x2,
    y2
):

    center_x = int(
        (x1 + x2) / 2
    )

    bottom_y = int(y2)

    for zone_index, mask in enumerate(
        zone_masks
    ):

        if point_inside_zone(
            mask,
            center_x,
            bottom_y
        ):

            return zone_index

    return None


def draw_zones(
    frame,
    zones,
    zone_masks
):

    if not zones:
        return

    for index, zone in enumerate(
        zones
    ):

        cv2.polylines(
            frame,
            [zone],
            True,
            (255, 255, 0),
            2
        )

        x, y, w, h = (
            cv2.boundingRect(zone)
        )

        cv2.putText(
            frame,
            f"ZONE {index + 1}",
            (
                x,
                max(25, y - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        contours, _ = cv2.findContours(
            zone_masks[index],
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
    camera_id,
    frame,
    video_seconds,
    event_name,
    track_ids,
    zone_index=None
):

    camera_dir = get_camera_output_dir(
        camera_id
    )

    screenshot_dir = (
        camera_dir /
        "screenshots"
    )

    screenshot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = (
        format_timestamp(
            video_seconds
        )
        .replace(":", "-")
    )

    ids_text = "_".join(
        str(x)
        for x in sorted(track_ids)
    )

    if zone_index is not None:

        filename = (
            f"{timestamp}_"
            f"ZONE_{zone_index + 1}_"
            f"{event_name}_"
            f"IDs_{ids_text}.jpg"
        )

    else:

        filename = (
            f"{timestamp}_"
            f"{event_name}_"
            f"IDs_{ids_text}.jpg"
        )

    screenshot_path = (
        screenshot_dir /
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


class CameraState:

    def __init__(
        self,
        camera_id,
        source
    ):

        self.camera_id = camera_id
        self.source = source

        self.reader = None

        self.person_model = None

        self.zones = None
        self.zone_masks = None

        self.width = 0
        self.height = 0
        self.fps = 25.0

        self.track_states = {}

        self.last_processed_frame_number = -1

        self.last_person_results = None

        self.processing_fps = 0.0

        self.fps_counter_start = (
            time.perf_counter()
        )

        self.fps_counter_frames = 0

        self.zone_multiple_start = {}
        self.zone_multiple_logged = {}
        self.zone_cooldown_until = {}

        self.event_screenshot_queue = []

        self.number_inside = 0

        self.last_phone_frame_number = -1

        self.running = True


def setup_camera(
    state,
    phone_model
):

    print()
    print(
        f"Setting up {state.camera_id}"
    )

    state.person_model = YOLO(
        PERSON_MODEL_PATH
    )

    state.reader = CameraReader(
        state.camera_id,
        state.source
    )

    state.width = state.reader.width
    state.height = state.reader.height
    state.fps = state.reader.fps

    print(
        f"{state.camera_id}: "
        f"{state.width}x{state.height} "
        f"@ {state.fps:.2f} FPS"
    )

    state.zones = load_zones(
        state.camera_id,
        state.width,
        state.height
    )

    state.reader.start()

    print(
        f"Waiting for first frame from "
        f"{state.camera_id}..."
    )

    while True:

        frame, frame_number, video_time = (
            state.reader.get_latest()
        )

        if frame is not None:
            break

        if state.reader.finished:

            raise RuntimeError(
                f"{state.camera_id} ended "
                f"before first frame."
            )

        time.sleep(0.01)

    if state.zones is None:

        state.zones = (
            create_zones_interactively(
                state.camera_id,
                frame
            )
        )

        if state.zones is None:

            state.reader.stop()
            state.running = False

            return False

    state.zone_masks = (
        create_expanded_zone_mask(
            state.zones,
            state.width,
            state.height
        )
    )

    for zone_index in range(
        len(state.zones)
    ):

        state.zone_multiple_start[
            zone_index
        ] = None

        state.zone_multiple_logged[
            zone_index
        ] = False

        state.zone_cooldown_until[
            zone_index
        ] = None

    print(
        f"{state.camera_id}: "
        f"{len(state.zones)} zone(s) active."
    )

    return True


def process_camera(
    state,
    phone_model
):

    frame, current_frame_number, current_video_time = (
        state.reader.get_latest()
    )

    if frame is None:

        if state.reader.finished:
            state.running = False

        return None

    if (
        current_frame_number ==
        state.last_processed_frame_number
    ):

        return None

    state.last_processed_frame_number = (
        current_frame_number
    )

    run_person_detection = (
        current_frame_number %
        PERSON_DETECTION_INTERVAL
        == 0
    )

    if run_person_detection:

        try:

            results = state.person_model.track(
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

            state.last_person_results = (
                results
            )

        except Exception as e:

            print(
                f"{state.camera_id} "
                f"person tracking error:",
                e
            )

            results = (
                state.last_person_results
            )

    else:

        results = (
            state.last_person_results
        )

    state.fps_counter_frames += 1

    fps_elapsed = (
        time.perf_counter()
        -
        state.fps_counter_start
    )

    if fps_elapsed >= 1.0:

        state.processing_fps = (
            state.fps_counter_frames /
            fps_elapsed
        )

        state.fps_counter_frames = 0

        state.fps_counter_start = (
            time.perf_counter()
        )

    current_persons = []

    if (
        results is not None
        and len(results) > 0
    ):

        result = results[0]

        if (
            result.boxes is not None
            and len(result.boxes) > 0
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

                zone_index = (
                    get_person_zone(
                        state.zone_masks,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                current_persons.append(
                    {
                        "id": track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "zone": zone_index
                    }
                )

    for person in current_persons:

        track_id = person["id"]

        zone_index = person["zone"]

        if track_id not in state.track_states:

            state.track_states[
                track_id
            ] = {

                "last_seen":
                    current_video_time,

                "last_inside":
                    current_video_time
                    if zone_index is not None
                    else None,

                "last_zone":
                    zone_index,

                "absence_logged":
                    False,

                "phone_logged":
                    False
            }

        track_state = (
            state.track_states[
                track_id
            ]
        )

        track_state[
            "last_seen"
        ] = current_video_time

        if zone_index is not None:

            track_state[
                "last_inside"
            ] = current_video_time

            track_state[
                "last_zone"
            ] = zone_index

            track_state[
                "absence_logged"
            ] = False

    for (
        track_id,
        track_state
    ) in state.track_states.items():

        if (
            track_state["last_inside"]
            is not None
        ):

            away_duration = (
                current_video_time
                -
                track_state[
                    "last_inside"
                ]
            )

            if (
                away_duration
                >= ABSENCE_LIMIT_SECONDS
                and
                not track_state[
                    "absence_logged"
                ]
            ):

                log_event(
                    state.camera_id,
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

                track_state[
                    "absence_logged"
                ] = True

    zone_inside_ids = {
        zone_index: set()
        for zone_index in range(
            len(state.zones)
        )
    }

    for track_id, track_state in (
        state.track_states.items()
    ):

        time_since_seen = (
            current_video_time
            -
            track_state["last_seen"]
        )

        if (
            track_state["last_inside"]
            is not None
        ):

            time_since_inside = (
                current_video_time
                -
                track_state[
                    "last_inside"
                ]
            )

        else:

            time_since_inside = float(
                "inf"
            )

        if (
            time_since_seen
            <= TRACK_GRACE_SECONDS
            and
            time_since_inside
            <= INSIDE_GRACE_SECONDS
        ):

            zone_index = (
                track_state["last_zone"]
            )

            if zone_index is not None:

                if zone_index in zone_inside_ids:

                    zone_inside_ids[
                        zone_index
                    ].add(track_id)

    total_inside = 0

    for zone_index in zone_inside_ids:

        total_inside += len(
            zone_inside_ids[
                zone_index
            ]
        )

    state.number_inside = total_inside

    for zone_index in range(
        len(state.zones)
    ):

        inside_ids = (
            zone_inside_ids[
                zone_index
            ]
        )

        number_inside_zone = len(
            inside_ids
        )

        cooldown_until = (
            state.zone_cooldown_until[
                zone_index
            ]
        )

        cooldown_active = False

        if cooldown_until is not None:

            if (
                current_video_time
                < cooldown_until
            ):

                cooldown_active = True

            else:

                state.zone_cooldown_until[
                    zone_index
                ] = None

                state.zone_multiple_logged[
                    zone_index
                ] = False

        if cooldown_active:

            state.zone_multiple_start[
                zone_index
            ] = None

        else:

            if number_inside_zone > 1:

                if (
                    state.zone_multiple_start[
                        zone_index
                    ] is None
                ):

                    state.zone_multiple_start[
                        zone_index
                    ] = current_video_time

                    state.zone_multiple_logged[
                        zone_index
                    ] = False

                    print(
                        f"[{state.camera_id}] "
                        f"Zone {zone_index + 1}: "
                        f"Multiple people detected. "
                        f"Timer started at "
                        f"{format_timestamp(current_video_time)}"
                    )

                multiple_duration = (
                    current_video_time
                    -
                    state.zone_multiple_start[
                        zone_index
                    ]
                )

                if (
                    multiple_duration
                    >= MULTIPLE_PERSON_LIMIT_SECONDS
                    and
                    not state.zone_multiple_logged[
                        zone_index
                    ]
                ):

                    state.zone_multiple_logged[
                        zone_index
                    ] = True

                    event_ids = set(
                        inside_ids
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

                    state.zone_cooldown_until[
                        zone_index
                    ] = cooldown_end

                    state.zone_multiple_start[
                        zone_index
                    ] = None

                    event_details = (
                        f"{number_inside_zone} "
                        f"people inside Zone "
                        f"{zone_index + 1} "
                        f"for "
                        f"{multiple_duration:.1f} "
                        f"seconds"
                    )

                    state.event_screenshot_queue.append(
                        {
                            "time":
                                current_video_time,

                            "event":
                                "MULTIPLE_PEOPLE_OVER_2_MINUTES",

                            "ids":
                                event_ids,

                            "zone":
                                zone_index
                        }
                    )

                    log_event(
                        state.camera_id,
                        current_video_time,
                        "MULTIPLE_PEOPLE_OVER_2_MINUTES",
                        ids_text,
                        (
                            event_details
                            +
                            "; 30 minute cooldown started"
                        )
                    )

            else:

                state.zone_multiple_start[
                    zone_index
                ] = None

                state.zone_multiple_logged[
                    zone_index
                ] = False

    run_phone_detection = (
        current_frame_number %
        PHONE_DETECTION_INTERVAL
        == 0
    )

    if run_phone_detection:

        for person in current_persons:

            if person["zone"] is None:
                continue

            track_id = person["id"]

            if (
                track_id
                in state.track_states
                and
                state.track_states[
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
                    state.camera_id,
                    current_video_time,
                    "PHONE_DETECTED",
                    str(track_id),
                    (
                        f"Phone detected "
                        f"inside CNC work zone "
                        f"Zone "
                        f"{person['zone'] + 1}"
                    )
                )

                state.track_states[
                    track_id
                ]["phone_logged"] = True

    states_to_remove = []

    for (
        track_id,
        track_state
    ) in state.track_states.items():

        time_since_seen = (
            current_video_time
            -
            track_state["last_seen"]
        )

        if time_since_seen > 600:

            states_to_remove.append(
                track_id
            )

    for track_id in states_to_remove:

        del state.track_states[
            track_id
        ]

    draw_zones(
        frame,
        state.zones,
        state.zone_masks
    )

    for person in current_persons:

        x1 = person["x1"]
        y1 = person["y1"]
        x2 = person["x2"]
        y2 = person["y2"]

        track_id = person["id"]

        zone_index = person["zone"]

        if zone_index is not None:

            box_color = (
                0,
                255,
                0
            )

            status = (
                f"ZONE {zone_index + 1}"
            )

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
                max(25, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

    panel_height = 210

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (620, panel_height),
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
            f"{state.camera_id} | "
            f"Video: "
            f"{format_timestamp(current_video_time)}"
        ),
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        (
            f"Processing FPS: "
            f"{state.processing_fps:.1f}"
        ),
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        (
            f"People in zones: "
            f"{state.number_inside}"
        ),
        (25, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        (
            f"Work zones: "
            f"{len(state.zones)}"
        ),
        (25, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y_position = 146

    for zone_index in range(
        len(state.zones)
    ):

        inside_count = len(
            zone_inside_ids[
                zone_index
            ]
        )

        cooldown_until = (
            state.zone_cooldown_until[
                zone_index
            ]
        )

        if (
            cooldown_until is not None
            and
            current_video_time
            < cooldown_until
        ):

            remaining = (
                cooldown_until
                -
                current_video_time
            )

            text = (
                f"Zone {zone_index + 1}: "
                f"COOLDOWN "
                f"{format_timestamp(remaining)}"
            )

            text_color = (
                0,
                165,
                255
            )

        elif inside_count > 1:

            start_time = (
                state.zone_multiple_start[
                    zone_index
                ]
            )

            if start_time is not None:

                elapsed = (
                    current_video_time
                    -
                    start_time
                )

                text = (
                    f"Zone {zone_index + 1}: "
                    f"{inside_count} PEOPLE "
                    f"{elapsed:.0f}/"
                    f"{MULTIPLE_PERSON_LIMIT_SECONDS}s"
                )

                text_color = (
                    0,
                    255,
                    255
                )

            else:

                text = (
                    f"Zone {zone_index + 1}: "
                    f"{inside_count} PEOPLE"
                )

                text_color = (
                    0,
                    255,
                    255
                )

        else:

            text = (
                f"Zone {zone_index + 1}: "
                f"{inside_count}/1"
            )

            text_color = (
                0,
                255,
                0
            )

        cv2.putText(
            frame,
            text,
            (25, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2
        )

        y_position += 25

    cv2.putText(
        frame,
        "Click window = select | Z = redraw zones",
        (25, min(
            frame.shape[0] - 10,
            y_position + 5
        )),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    if state.event_screenshot_queue:

        event = (
            state.event_screenshot_queue.pop(
                0
            )
        )

        save_event_screenshot(
            state.camera_id,
            frame,
            event["time"],
            event["event"],
            event["ids"],
            event["zone"]
        )

        print(
            f"[{state.camera_id}] "
            f"30 minute cooldown started."
        )

    if frame.shape[1] > DISPLAY_WIDTH:

        scale = (
            DISPLAY_WIDTH /
            frame.shape[1]
        )

        display_frame = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                int(
                    frame.shape[0] *
                    scale
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        display_frame = frame

    return display_frame


def redraw_camera_zones(state):

    frame, _, _ = (
        state.reader.get_latest()
    )

    if frame is None:
        return

    new_zones = (
        create_zones_interactively(
            state.camera_id,
            frame
        )
    )

    if new_zones is not None:

        state.zones = new_zones

        state.zone_masks = (
            create_expanded_zone_mask(
                state.zones,
                state.width,
                state.height
            )
        )

        state.zone_multiple_start.clear()
        state.zone_multiple_logged.clear()
        state.zone_cooldown_until.clear()

        for zone_index in range(
            len(state.zones)
        ):

            state.zone_multiple_start[
                zone_index
            ] = None

            state.zone_multiple_logged[
                zone_index
            ] = False

            state.zone_cooldown_until[
                zone_index
            ] = None

        print(
            f"{state.camera_id}: "
            f"New zones activated."
        )


def main():

    print()
    print(
        "CNC 4 CAMERA MONITORING"
    )
    print()

    cameras = config.CAMERAS

    if not cameras:

        print(
            "No cameras configured."
        )

        return

    print(
        "Loading shared phone model..."
    )

    phone_model = YOLO(
        PHONE_MODEL_PATH
    )

    print(
        "Phone model loaded."
    )

    states = {}

    for camera_id, source in (
        cameras.items()
    ):

        try:

            state = CameraState(
                camera_id,
                source
            )

            success = setup_camera(
                state,
                phone_model
            )

            if success:

                states[
                    camera_id
                ] = state

                print(
                    f"{camera_id} READY"
                )

            else:

                print(
                    f"{camera_id} disabled."
                )

        except Exception as e:

            print()
            print(
                f"FAILED TO START "
                f"{camera_id}"
            )

            print(e)

    if not states:

        print(
            "No cameras could be started."
        )

        return

    window_names = {}

    for camera_id in states:

        window_name = (
            f"CNC - {camera_id}"
        )

        window_names[
            window_name
        ] = camera_id

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

    selected_camera = (
        list(states.keys())[0]
    )

    def make_mouse_callback(
        camera_id
    ):

        def callback(
            event,
            x,
            y,
            flags,
            param
        ):

            nonlocal selected_camera

            if event == (
                cv2.EVENT_LBUTTONDOWN
            ):

                selected_camera = (
                    camera_id
                )

                print(
                    f"Selected camera: "
                    f"{camera_id}"
                )

        return callback

    for camera_id in states:

        cv2.setMouseCallback(
            f"CNC - {camera_id}",
            make_mouse_callback(
                camera_id
            )
        )

    try:

        while states:

            active_states = list(
                states.values()
            )

            for state in active_states:

                if not state.running:
                    continue

                display_frame = (
                    process_camera(
                        state,
                        phone_model
                    )
                )

                if display_frame is None:
                    continue

                window_name = (
                    f"CNC - "
                    f"{state.camera_id}"
                )

                cv2.imshow(
                    window_name,
                    display_frame
                )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                break

            if key == ord("z"):

                if (
                    selected_camera
                    in states
                ):

                    print()

                    print(
                        f"Redefining zones "
                        f"for "
                        f"{selected_camera}"
                    )

                    redraw_camera_zones(
                        states[
                            selected_camera
                        ]
                    )

            finished_cameras = []

            for (
                camera_id,
                state
            ) in states.items():

                if not state.running:

                    finished_cameras.append(
                        camera_id
                    )

            for camera_id in (
                finished_cameras
            ):

                state = states[
                    camera_id
                ]

                state.reader.stop()

                cv2.destroyWindow(
                    f"CNC - {camera_id}"
                )

                del states[
                    camera_id
                ]

            if not states:
                break

    finally:

        for state in states.values():

            state.running = False

            if state.reader is not None:

                state.reader.stop()

        for state in states.values():

            if (
                state.reader is not None
                and
                state.reader.is_alive()
            ):

                state.reader.join(
                    timeout=1.0
                )

        cv2.destroyAllWindows()

    print()
    print(
        "CNC 4 CAMERA MONITORING FINISHED"
    )


if __name__ == "__main__":
    main()