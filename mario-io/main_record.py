# main_record.py

import cv2
import time

from video_capture import VideoCapture
from dataset_writer import DatasetWriter
from control.joycon_device import JoyConDevice
from control.udp_bridge import UDPBridge
from control.control_packet import build_packet


OBS_INDEX = 0
CAPTURE_FPS = 30
SAVE_FPS = 30
NXBT_HZ = 120


def main():
    video = VideoCapture(OBS_INDEX, 1280, 720, CAPTURE_FPS)
    joycon = JoyConDevice()
    bridge = UDPBridge()

    writer = None
    recording = False

    save_stride = CAPTURE_FPS // SAVE_FPS
    capture_idx = 0

    last_send = time.monotonic()

    print("r=start recording, s=stop, q=quit")

    while True:
        frame, frame_ts = video.read()
        if frame is None:
            continue

        # Read controller
        control = joycon.read()

        # Build nxbt packet
        pkt = build_packet(
            control["lx"],
            control["ly"],
            control["a"],
            control["b"],
            control["x"],
            control["y"],
            control["drift"],
            control["pause"],
        )

        # Send at NXBT_HZ
        now = time.monotonic()
        if now - last_send >= 1.0 / NXBT_HZ:
            bridge.send(pkt)
            last_send = now

        # Save dataset at lower FPS
        if recording and capture_idx % save_stride == 0:
            writer.write(frame, frame_ts, control)

        display = frame.copy()

        cv2.putText(display, f"FPS: {video.measured_fps}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(display, f"Recording: {recording}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Mario Kart Recorder + NXBT", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r") and not recording:
            writer = DatasetWriter("datasets", 1280, 720, SAVE_FPS)
            recording = True
        elif key == ord("s") and recording:
            recording = False
            writer.close()

        capture_idx += 1

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
