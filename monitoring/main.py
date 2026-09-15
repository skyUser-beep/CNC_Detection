import cv2
import torch

import config
from camera.worker import CameraWorker
from detection.phone_detector import PhoneDetector
from events.event_manager import EventManager
from zones.zone_manager import ZoneManager

def choose_device():
    if torch.cuda.is_available():
        print("CUDA GPU DETECTED")
        print("GPU:", torch.cuda.get_device_name(0))
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return 0, True
    print("CUDA NOT AVAILABLE - using CPU")
    return "cpu", False

def main():
    print("\nCNC 4 CAMERA MONITORING - REFACTORED\n")
    if not config.cameras:
        print("No cameras configured. Edit .env first.")
        return

    device, half = choose_device()
    events = EventManager(config.outputs_dir)
    zones = ZoneManager(config.outputs_dir, config.zone_margin_px)
    print("Loading shared phone model...")
    phone_detector = PhoneDetector(config.phone_model_path, device, half, config.phone_confidence, config.phone_image_size, config.phone_class_id)

    workers = {}
    for camera_id, source in config.cameras.items():
        try:
            worker = CameraWorker(camera_id, source, config, device, half, phone_detector, events, zones)
            if not worker.prepare():
                continue
            worker.startup_ready.set()
            worker.start()
            workers[camera_id] = worker
            print(f"{camera_id} READY")
        except Exception as exc:
            print(f"FAILED TO START {camera_id}: {exc}")

    if not workers:
        print("No cameras could be started.")
        return

    windows = {camera_id: f"CNC - {camera_id}" for camera_id in workers}
    for name in windows.values():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    selected_camera = next(iter(workers))

    def callback_for(camera_id):
        def callback(event, x, y, flags, param):
            nonlocal selected_camera
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_camera = camera_id
                print(f"Selected camera: {camera_id}")
        return callback

    for camera_id, window in windows.items():
        cv2.setMouseCallback(window, callback_for(camera_id))

    try:
        while workers:
            finished = []
            for camera_id, worker in list(workers.items()):
                if not worker.is_alive() or not worker.running:
                    finished.append(camera_id)
                    continue
                display = worker.get_display()
                if display is not None:
                    cv2.imshow(windows[camera_id], display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("z") and selected_camera in workers:
                workers[selected_camera].redraw_zones()

            for camera_id in finished:
                worker = workers.pop(camera_id)
                worker.running = False
                if worker.reader is not None:
                    worker.reader.stop()
                try:
                    cv2.destroyWindow(windows[camera_id])
                except Exception:
                    pass
            if not workers:
                break
    finally:
        for worker in workers.values():
            worker.running = False
            if worker.reader is not None:
                worker.reader.stop()
        for worker in workers.values():
            worker.join(timeout=2)
        cv2.destroyAllWindows()

    print("\nCNC 4 CAMERA MONITORING FINISHED")

if __name__ == "__main__":
    main()
