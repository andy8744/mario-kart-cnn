import time
import threading
import cv2
import numpy as np

from inference.preprocess import preprocess_bgr_frame

class VisionInferenceThread(threading.Thread):
    def __init__(self, cam_index: int, model, stop_evt: threading.Event, shared: dict, lock: threading.Lock,
                 capture_w=1280, capture_h=720, capture_fps=60, infer_fps=60):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.model = model
        self.stop_evt = stop_evt
        self.shared = shared
        self.lock = lock
        self.capture_w = capture_w
        self.capture_h = capture_h
        self.capture_fps = capture_fps
        self.infer_fps = infer_fps

        self.cap = None

    def open_camera(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.cam_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
        cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
        return cap

    def run(self):
        self.cap = self.open_camera()

        period = 1.0 / float(self.infer_fps)
        next_t = time.monotonic()

        # Warm start: set a safe default
        with self.lock:
            self.shared["lx"] = 0.0
            self.shared["ly"] = 0.0
            self.shared["ok"] = False
            self.shared["last_infer_ns"] = 0

        while not self.stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            ret, frame = self.cap.read()
            if not ret or frame is None:
                with self.lock:
                    self.shared["ok"] = False
                continue

            x = preprocess_bgr_frame(frame)  # (90,160,3) float32
            x = np.expand_dims(x, axis=0)    # (1,90,160,3)

            # Forward pass (random weights)
            y = self.model(x, training=False).numpy()[0]
            lx = float(np.clip(y[0], -1.0, 1.0))
            ly = float(np.clip(y[1], -1.0, 1.0))

            with self.lock:
                self.shared["lx"] = lx
                self.shared["ly"] = ly
                self.shared["ok"] = True
                self.shared["last_infer_ns"] = time.monotonic_ns()

        try:
            self.cap.release()
        except Exception:
            pass
