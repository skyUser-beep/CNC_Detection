import sys # locate the shared config module
from pathlib import Path

import os
import time
import subprocess
from urllib.parse import quote

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CAMERA_IP = config.CAMERA_IP
CAMERA_USERNAME = config.CAMERA_USERNAME
CAMERA_PASSWORD = config.CAMERA_PASSWORD

WIDTH = 640
HEIGHT = 360

WINDOW_NAME = "Hikvision VLC + OpenCV Test"

VLC_EXE = config.find_vlc_executable()
if VLC_EXE is None:
    raise RuntimeError("VLC executable was not found.")

username = quote(CAMERA_USERNAME,safe="")
password = quote(CAMERA_PASSWORD,safe="")

RTSP_URL = f"rtsp://{username}:{password}@{CAMERA_IP}:554/Streaming/Channels/102"

print("HIKVISION VLC + OPENCV TEST")
print("Camera IP :", CAMERA_IP)
print("Resolution:", f"{WIDTH}x{HEIGHT}")
print("VLC       :", VLC_EXE)

LOCAL_URL = f"http://127.0.0.1:8080/"

vlc_command = [
    VLC_EXE,
    # No graphical VLC window
    "--intf",
    "dummy",
    # Do not show title
    "--no-video-title-show",
    # RTSP over TCP
    "--rtsp-tcp",
    # Network buffering
    "--network-caching=300",
    "--live-caching=300",
    # RTSP input
    RTSP_URL,
    # Transcode video to MJPEG
    "--sout",
    (
        f"#transcode{{"
        f"vcodec=MJPG,"
        f"vb=2000,"
        f"width={WIDTH},"
        f"height={HEIGHT}"
        f"}}"
        f":http{{"
        f"mux=mpjpeg,"
        f"dst=:8080/"
        f"}}"
    ),
    # Keep VLC running
    "--sout-keep",
]
print()
print("Starting VLC receiver...")
print()

process = subprocess.Popen(vlc_command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

print("Waiting for VLC local stream...")
cap = None
start_time = time.time()

while time.time() - start_time < 15:
    cap = cv2.VideoCapture(LOCAL_URL,cv2.CAP_FFMPEG)
    if cap.isOpened():
        print()
        print("SUCCESS: OpenCV connected to VLC.")
        break
    cap.release()
    cap = None
    time.sleep(0.5)

if cap is None:
    print()
    print("ERROR: OpenCV could not connect to VLC.")
    print()
    print("VLC process status:", process.poll())
    process.terminate()

    raise RuntimeError("Could not open VLC local stream.")

frame_count = 0
fps = 0.0
fps_start = time.time()
last_frame = time.time()

print()
print("VIDEO TEST STARTED")
print()
print("You should now see the camera.")
print()
print("Move the camera physically to test stability.")
print()
print("Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print( "Frame read failed.")
            time.sleep(0.1)
            continue

        last_frame = time.time()
        frame_count += 1

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame,(WIDTH, HEIGHT))

        cv2.putText(frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(frame,
            f"{WIDTH}x{HEIGHT}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(frame,
            "HIKVISION -> VLC -> OPENCV",
            (10, HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )
        cv2.imshow(WINDOW_NAME,frame )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print()
            print("Q pressed.")
            break

        if process.poll() is not None:
            print()
            print("VLC process stopped unexpectedly.")
            break

finally:
    print()
    print("Stopping...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
    print("Done.")