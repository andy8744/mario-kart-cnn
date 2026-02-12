import cv2
import numpy as np


def preprocess_bgr_frame(frame_bgr, out_w=160, out_h=90) -> np.ndarray:
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return x
