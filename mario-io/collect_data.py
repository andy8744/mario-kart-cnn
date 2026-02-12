# main_record_headless.py

import sys
import time
import select
import argparse
import random

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


def new_writer():
    return DatasetWriter("datasets", 1280, 720, SAVE_FPS)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_args():
    p = argparse.ArgumentParser(description="Headless recorder with probabilistic A throttle")
    p.add_argument("--a_hold", type=float, default=1.0,
                   help="Probability of A being pressed on a tick (0..1). Example: 0.6 => 60%% of ticks.")
    p.add_argument("--seed", type=int, default=123,
                   help="RNG seed for throttle pattern (matters if a_hold < 1.0).")
    return p.parse_args()


def main():
    args = parse_args()
    a_hold = clamp01(args.a_hold)
    rng = random.Random(args.seed)

    print("Starting headless recorder...")
    print("Commands:")
    print("  r + ENTER  -> start recording (with countdown)")
    print("  s + ENTER  -> stop recording (finalize/save)")
    print("  q + ENTER  -> quit\n")
    print(f"Throttle override: a_hold={a_hold:.2f} seed={args.seed} (A is sampled each tick)\n")

    video = VideoCapture(OBS_INDEX, 1280, 720, CAPTURE_FPS)
    joycon = JoyConDevice()
    bridge = UDPBridge()

    save_stride = max(1, CAPTURE_FPS // SAVE_FPS)
    capture_idx = 0

    last_send = time.monotonic()

    writer = None
    recording_active = False
    pending_start = False
    start_requested_at = None
    last_countdown_print = None

    try:
        while True:
            # ---- Terminal command handling (non-blocking) ----
            if key_pressed():
                cmd = sys.stdin.readline().strip().lower()

                if cmd == "q":
                    print("Quitting...")
                    break

                elif cmd == "r":
                    if recording_active or pending_start:
                        print("Already recording (or countdown in progress).")
                    else:
                        pending_start = True
                        start_requested_at = time.monotonic()
                        last_countdown_print = None
                        print(f"Start requested. Recording will begin in {START_DELAY_SECONDS}s...")

                elif cmd == "s":
                    if pending_start:
                        pending_start = False
                        start_requested_at = None
                        print("Start cancelled (was in countdown).")

                    if recording_active:
                        recording_active = False
                        if writer is not None:
                            writer.close()
                            writer = None
                        print("Recording stopped and saved.\n")
                    else:
                        print("Not currently recording.")

                else:
                    if cmd:
                        print(f"Unknown command: '{cmd}'. Use r/s/q.")

            # ---- Read frame ----
            frame, frame_ts = video.read()
            if frame is None:
                continue

            # ---- Read controls (raw) ----
            control_raw = joycon.read()

            # ---- Apply probabilistic throttle override (effective) ----
            # Same mechanism as your inference sender: per-tick Bernoulli.
            a_eff = 1 if (a_hold >= 1.0 or rng.random() < a_hold) else 0

            control = dict(control_raw)  # shallow copy
            control["a"] = a_eff
            control["a_hold"] = a_hold   # useful for debugging/analysis
            control["a_raw"] = int(control_raw.get("a", 0))

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

            # ---- Always send controls over UDP ----
            now = time.monotonic()
            if now - last_send >= 1.0 / NXBT_HZ:
                bridge.send(pkt)
                last_send = now

            # ---- Countdown-to-start logic ----
            if pending_start and start_requested_at is not None:
                elapsed = now - start_requested_at
                remaining = START_DELAY_SECONDS - elapsed

                if remaining > 0:
                    sec = int(remaining) + 1
                    if last_countdown_print != sec:
                        print(f"Recording starts in {sec}...")
                        last_countdown_print = sec
                else:
                    pending_start = False
                    recording_active = True
                    writer = new_writer()
                    capture_idx = 0
                    print("Recording started.\n")

            # ---- Write frames only when recording ----
            if recording_active and writer is not None:
                if capture_idx % save_stride == 0:
                    # Write EFFECTIVE control (matches what you sent)
                    writer.write(frame, frame_ts, control)
                capture_idx += 1

    finally:
        if writer is not None:
            writer.close()
        video.release()

    print("Done.")


if __name__ == "__main__":
    main()
