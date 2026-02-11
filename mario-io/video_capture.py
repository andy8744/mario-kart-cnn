# video_capture.py

import cv2
import time


class VideoCapture:
    def __init__(self, index: int, width=1280, height=720, fps=60):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.last_fps_time = time.monotonic()
        self.frame_counter = 0
        self.measured_fps = 0

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        ts_ns = time.monotonic_ns()

        self.frame_counter += 1
        now = time.monotonic()
        if now - self.last_fps_time >= 1.0:
            self.measured_fps = self.frame_counter
            self.frame_counter = 0
            self.last_fps_time = now

        return frame, ts_ns

    def release(self):
        self.cap.release()
