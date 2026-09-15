import sys # locate the shared config module
from pathlib import Path

import cv2
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

print("HIKVISION RTSP TEST")

CAMERA_IP = config.CAMERA_IP
USERNAME = config.CAMERA_USERNAME
PASSWORD = config.CAMERA_PASSWORD

if not CAMERA_IP or CAMERA_IP == "YOUR_IP" or not USERNAME or not PASSWORD:
    print("\nERROR: Camera settings are missing. Fill them in your .env file.")
    exit()
USERNAME_ENCODED = quote(USERNAME, safe="")
PASSWORD_ENCODED = quote(PASSWORD, safe="")

RTSP_URL = f"rtsp://{USERNAME_ENCODED}:{PASSWORD_ENCODED}@{CAMERA_IP}:554/Streaming/Channels/101"

print("\nConnecting to Hikvision...")
print("Please wait...\n")

cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(" CONNECTION FAILED")
    print("\nOpenCV/FFmpeg could not open the Hikvision stream.")
    cap.release()
    exit()

print(" HIKVISION CONNECTED!")
print("\nReading video frames...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("\n Frame could not be read.")
        break

    frame_count += 1
    cv2.putText(
        frame,
        f"Hikvision | Frames: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2)

    cv2.imshow("Hikvision CCTV Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
print("\nTest finished.")