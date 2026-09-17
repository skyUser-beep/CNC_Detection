import threading # runs the camers reader independently of the detection process
import time # controls timing and frame-reading speed
from pathlib import Path # checks whether a source is a local file
import cv2 # opens cameras/videos and reads frames
# only responsible for getting frames from the camera reading video/RTSP stream
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]=("rtsp_transport;tcp")

class CameraReader(threading.Thread):
    def __init__(self, camera_id, source):
        super().__init__(daemon=True, name=f"reader-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {camera_id}: {source}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_file = bool(self.total_frames > 0 and Path(str(source)).exists())
        self.latest_frame = None
        self.latest_frame_number = -1
        self.latest_video_time = 0.0
        self.finished = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # Lock protects shared variables while they are being updated or read.
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
                self.latest_video_time = video_time

            if self.is_file:
                next_frame_time += 1.0 / self.fps
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.perf_counter()
            else:
                time.sleep(0.0005)

        self.cap.release()

    def get_latest(self): # Updating the latest frame  called by worker.py
        with self.lock:
            if self.latest_frame is None:
                return None, -1, 0.0
            return self.latest_frame.copy(), self.latest_frame_number, self.latest_video_time

    def stop(self): 
        self.stop_event.set()

# Camera - Reader reads a frame , Latest frame is stored