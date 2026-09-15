import sys # locate the shared config module
from pathlib import Path

from ultralytics import YOLO
import cv2
import time
import subprocess
from urllib.parse import quote
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# override MODEL_PATH in your .env
MODEL_PATH = config.MODEL_PATH
WIDTH=640
HEIGHT=360

CONFIDENCE=0.25

PERSON_CLASS=0
CELL_PHONES_CLASS=67

PHONE_TIME_THRESHOLD=0.0

WINDOW_NAME="CNC Monitoring"

LOCAL_URL = f"http://127.0.0.1:8080/"

CAMERA_IP = config.CAMERA_IP
USERNAME = config.CAMERA_USERNAME
PASSWORD = config.CAMERA_PASSWORD

ENCODE_USERNAME= quote(str(USERNAME), safe='')
ENCODE_PASSWORD= quote(str(PASSWORD), safe='')

RTSP_URL= f"rtsp://{ENCODE_USERNAME}:{ENCODE_PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/102"

VLC_EXE = config.find_vlc_executable()

if VLC_EXE is None:
    print("VLC path not found.")
    print("Set VLC_PATH in your .env, or install VLC to one of the default locations.")
    raise SystemExit(1)

print()
print("Loading YOLO Model")
model=YOLO(MODEL_PATH)
print("Model loaded successfully.")

work_zone=[]
drawing_zone=False

person_data=defaultdict(lambda :{"first_seen":None, "last_seen":None,
                        "inside_zone_since":None,"phone_since":None,"state": "UNKNOWN"})

def start_vlc():
    print()
    print("STARTING VLC")
    print()
    vlc_command = [
        VLC_EXE,
        "--intf", "dummy",
        "--no-video-title-show",
        # Correct VLC RTSP option
        "--rtsp-tcp",
        # Small cache for low latency
        "--network-caching=300",
        "--live-caching=300",
        RTSP_URL,
        "--sout",
        (
            f"#transcode{{"
            f"vcodec=MJPG,"
            f"vb=5000,"
            f"width={WIDTH},"
            f"height={HEIGHT}"
            f"}}"
            f":http{{"
            f"mux=mpjpeg,"
            f"dst=:8080/"
            f"}}"
        ),
        # Correct option
        "--sout-keep"
    ]
    try:
        process = subprocess.Popen(vlc_command)

    except Exception as e:
        print()
        print("ERROR: Could not start VLC.")
        print(e)
        raise SystemExit(1)
    print("VLC started.")
    print("Waiting for local MJPEG stream...")
    return process

def open_video_stream():
    print()
    print("Connecting OpenCV to VLC...")
    print("URL:", LOCAL_URL)
    cap = cv2.VideoCapture(LOCAL_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def point_inside_zone(x, y):
    if len(work_zone) < 3:
        return False
    polygon=np.array(work_zone,dtype=np.int32)
    result = cv2.pointPolygonTest(polygon,(float(x), float(y)),False)
    return result >= 0

def draw_work_zone(frame):
    if len(work_zone) == 0:
        return
    for point in work_zone:
        cv2.circle(frame,point,
            4,(0, 255, 255),
            -1)
    if len(work_zone) >= 2:
        for i in range(len(work_zone) - 1):
            cv2.line(frame,
                work_zone[i],work_zone[i + 1],
                (0, 255, 255),2 )

    if not drawing_zone and len(work_zone) >= 3:
        polygon = np.array(work_zone, dtype=np.int32)
        cv2.line(frame,
            work_zone[-1],work_zone[0],
            (0, 255, 255),2)

        overlay = frame.copy()
        cv2.fillPoly(overlay,
            [polygon],
            (0, 255, 255)
        )
        frame[:] = cv2.addWeighted(
            overlay,
            0.12,frame,
            0.88,0
        )

def phone_belongs_to_person(person_box,phone_box):
    # Correct order:
    # x1, y1, x2, y2
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = phone_box
    phone_center_x = (fx1 + fx2) / 2
    phone_center_y = (fy1 + fy2) / 2
    # Check whether phone center is inside
    # the person's bounding box.
    inside_person = (px1 <= phone_center_x <= px2 and py1 <= phone_center_y <= py2)

    if inside_person:
        return True
    # Also allow a small margin around the person.
    # This helps when the phone is detected slightly
    # outside the person's bounding box.

    margin_x = (px2 - px1) * 0.20
    margin_y = (py2 - py1) * 0.20

    expanded_x1 = px1 - margin_x
    expanded_y1 = py1 - margin_y
    expanded_x2 = px2 + margin_x
    expanded_y2 = py2 + margin_y

    return expanded_x1 <= phone_center_x <= expanded_x2 and expanded_y1 <= phone_center_y <= expanded_y2

def mouse_callback(event,x,y,flags,param):

    global work_zone
    global drawing_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_zone:
            work_zone.append((x, y))
            print( f"Point added :({x},{y})")

def start_zone_drawing():
    global work_zone
    global drawing_zone
    # VERY IMPORTANT:
    # Completely remove the previous polygon.
    work_zone = []
    drawing_zone = True
    print()
    print("ZONE DRAWING STARTED")
    print("Click around the CNC work area.")
    print("Press ENTER when finished.")
    print("Press C to cancel.")
    print()

def clear_work_zone():
    global work_zone
    global drawing_zone
    work_zone = []
    drawing_zone = False
    print()
    print("Work zone cleared.")
    print()

def finish_zone():
    global drawing_zone
    if len(work_zone) < 3:
        print()
        print("Need at least 3 points to create a work zone.")
        print()
        return
    drawing_zone = False
    print()
    print("Work zone completed.")
    print("Points:", work_zone)
    print()


def main():
    global drawing_zone
    vlc_process = start_vlc()
    cap = None
    connected = False
    for attempt in range(30):
        cap = open_video_stream()
        time.sleep(0.5)
        if cap.isOpened():
            connected = True
            print()
            print("OpenCV connected to VLC.")
            print()
            break
        cap.release()

        print(f"Waiting for VLC stream... attempt {attempt + 1}/30")
        time.sleep(0.5)
    if not connected:
        print()
        print("ERROR: Could not connect to VLC.")
        print()
        vlc_process.terminate()
        return
    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,WIDTH,HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME,mouse_callback)

    previous_time = time.time()
    fps = 0.0
    consecutive_failures = 0
    MAX_FAILURES = 20
    print()
    print("CNC MONITORING STARTED")
    print()
    print("Z     = Draw NEW work zone")
    print("ENTER = Finish zone")
    print("C     = Clear zone")
    print("Q     = Quit")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"Failed to receive frame from VLC. Attempt {consecutive_failures}/{MAX_FAILURES}")
                # Try reading again first
                time.sleep(0.05)
                if consecutive_failures < MAX_FAILURES:
                    continue
                print()
                print("OpenCV lost VLC stream.")
                print("Attempting to reconnect...")
                print()

                cap.release()
                time.sleep(1)
                cap = open_video_stream()
                time.sleep(1)
                if cap.isOpened():
                    print("OpenCV reconnected to VLC.")
                    consecutive_failures = 0
                    continue
                else:
                    print("VLC stream still unavailable.")
                    consecutive_failures = 0
                    continue
            # Successful frame
            consecutive_failures = 0
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame,(WIDTH, HEIGHT) )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                if not drawing_zone:
                    start_zone_drawing()
                else:
                    print("Already drawing zone. Press ENTER to finish or C to cancel.")
            elif key == ord("c"):
                clear_work_zone()
            elif key == 13:  # ENTER
                if drawing_zone:
                    finish_zone()

            results = model.track(
                frame,
                persist=True,
                conf=CONFIDENCE,
                imgsz=640,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            persons = []
            phones = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int,xyxy)
                        if cls == PERSON_CLASS:
                            track_id = None
                            if box.id is not None:
                                track_id = int(box.id[0].item())
                            persons.append({"id": track_id,
                                    "box": (
                                        x1, y1,
                                        x2,y2),
                                    "confidence": confidence
                                }
                            )

                        elif cls == CELL_PHONES_CLASS:
                            phones.append((x1,y1,x2,y2))
            for phone_box in phones:
                fx1, fy1, fx2, fy2 = phone_box
                cv2.rectangle(
                    frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (0, 0, 255),
                    2
                )
                cv2.putText(frame,
                    "PHONE",
                    (fx1, max(20, fy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0, 0, 255),
                    2
                )
            current_time = time.time()
            for person in persons:
                person_id = person["id"]
                x1, y1, x2, y2 = person["box"]
                confidence = person["confidence"]
                center_x = int((x1 + x2) / 2)
                center_y = int( (y1 + y2) / 2)
                inside_zone = point_inside_zone(center_x,center_y)
                phone_detected = False
                for phone_box in phones:
                    if phone_belongs_to_person(person["box"],phone_box):
                        phone_detected = True
                        break

                if person_id is not None:
                    data = person_data[person_id]
                    if data["first_seen"] is None:
                        data["first_seen"] = current_time
                    data["last_seen"] = current_time
                    if inside_zone:
                        if data["inside_zone_since"] is None:
                            data["inside_zone_since"] = current_time
                    else:
                        data["inside_zone_since" ] = None
                    if phone_detected:
                        if data["phone_since"] is None:
                            data["phone_since"] = current_time
                    else:
                        data[ "phone_since"] = None

                    if not inside_zone:
                        data["state"] = "AWAY"
                    else:
                        if data["phone_since"] is not None and current_time - data["phone_since"] >= PHONE_TIME_THRESHOLD:
                            data["state"] = "PHONE"
                        else:
                            data["state"] = "PRESENT"
                    state = data["state"]

                else:
                    # Person has no tracking ID
                    # yet.
                    state = "UNKNOWN"

                if state == "PHONE":
                    box_color = (0,0,255)
                elif state == "PRESENT":
                    box_color = (0,255,0)
                elif state == "AWAY":
                    box_color = (255,0, 0)
                else:
                    box_color = (255,255,0)
                cv2.rectangle(frame,
                    (x1, y1),(x2, y2),
                    box_color,
                    2
                )
                if person_id is not None:
                    label = f"ID {person_id} "f"{state}"
                else:
                    label = state
                cv2.putText(frame,label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                cv2.circle(frame,(center_x, center_y),4,box_color, -1)
                if person_id is not None and person_data[person_id]["phone_since"] is not None:
                    phone_duration = (current_time-person_data[person_id]["phone_since"])
                    cv2.putText( frame,f"Phone: {phone_duration:.1f}s",
                        ( x1,min(HEIGHT - 10,y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        2
                    )
            draw_work_zone(frame)
            if drawing_zone:
                cv2.putText(frame,
                    "DRAWING ZONE - ENTER = FINISH | C = CANCEL",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2
                )
            elif len(work_zone) >= 3:
                cv2.putText(frame,
                    "Z = NEW ZONE | C = CLEAR",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2
                )
            else:
                cv2.putText(frame,
                    "Press Z to draw CNC work zone",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    2
                )

            current_frame_time = time.time()
            elapsed = current_frame_time-previous_time

            if elapsed > 0:
                fps = (0.9 * fps+0.1 * (1.0 / elapsed))
            previous_time = current_frame_time

            cv2.putText(frame,f"FPS: {fps:.1f}",
                (10,HEIGHT - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            cv2.putText(frame,
                "HIKVISION -> VLC -> YOLO",
                (10,HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )
            cv2.imshow(WINDOW_NAME,frame)
    finally:
        print()
        print("Stopping CNC monitoring...")
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            vlc_process.terminate()
            vlc_process.wait(timeout=3 )
        except Exception:
            try:
                vlc_process.kill()
            except Exception:
                pass
        print("CNC monitoring stopped.")

if __name__ == "__main__":
    main()