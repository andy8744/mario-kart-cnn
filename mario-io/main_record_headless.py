# main_record_headless.py

import sys
import time
import select

from video_capture import VideoCapture
from dataset_writer import DatasetWriter
from control.joycon_device import JoyConDevice
from control.udp_bridge import UDPBridge
from control.control_packet import build_packet


OBS_INDEX = 0
CAPTURE_FPS = 30
SAVE_FPS = 30
NXBT_HZ = 120
START_DELAY_SECONDS = 5

def key_pressed():
    """Non-blocking check for terminal keypress."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr


def main():
    print("Starting headless recorder...")
    print("Press 'q' then ENTER to quit.\n")

    video = VideoCapture(OBS_INDEX, 1280, 720, CAPTURE_FPS)
    joycon = JoyConDevice()
    bridge = UDPBridge()

    writer = DatasetWriter("datasets", 1280, 720, SAVE_FPS)

    save_stride = CAPTURE_FPS // SAVE_FPS
    capture_idx = 0

    last_send = time.monotonic()
    start_time = time.monotonic()

    recording_active = False

    try:
        while True:
            # Quit check
            if key_pressed():
                cmd = sys.stdin.readline().strip()
                if cmd == "q":
                    print("Stopping...")
                    break

            frame, frame_ts = video.read()
            if frame is None:
                continue

            control = joycon.read()

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

            # Always send controls
            now = time.monotonic()
            if now - last_send >= 1.0 / NXBT_HZ:
                bridge.send(pkt)
                last_send = now

            # Countdown logic
            elapsed = now - start_time
            if not recording_active:
                remaining = START_DELAY_SECONDS - elapsed
                if remaining > 0:
                    # Print countdown once per second
                    if int(remaining) != int(remaining + 0.05):
                        print(f"Recording starts in {int(remaining)+1}...")
                else:
                    print("Recording started.\n")
                    recording_active = True

            # Write frames only after delay
            if recording_active and capture_idx % save_stride == 0:
                writer.write(frame, frame_ts, control)

            capture_idx += 1

    finally:
        writer.close()
        video.release()

    print("Done.")


if __name__ == "__main__":
    main()
