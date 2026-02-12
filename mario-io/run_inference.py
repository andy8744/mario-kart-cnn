import argparse
import random
import threading
import time

import cv2
import tensorflow as tf

from control.control_packet import build_packet
from control.udp_bridge import UDPBridge
from inference.vision_thread import VisionInferenceThread

# ---- DEFAULT CONFIG ----
OBS_INDEX = 0
VM_IP = "192.168.64.2"
PORT = 5005

SEND_HZ = 120
CAPTURE_FPS = 30
INFER_FPS = 30

SHOW_HUD = True  # set False for fully headless
HUD_SCALE = 2  # 1=160x90, 2=320x180, 4=640x360
# ------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Mario Kart CNN inference sender")
    p.add_argument(
        "--model",
        default="./models/mario_kart_cnn.keras",
        help="Path to Keras model file (.keras / .h5). Default: ./models/mario_kart_cnn.keras",
    )
    p.add_argument(
        "--obs_index", type=int, default=OBS_INDEX, help="OBS virtual cam index"
    )
    p.add_argument("--ip", default=VM_IP, help="UDP target IP (e.g. VM IP)")
    p.add_argument("--port", type=int, default=PORT, help="UDP target port")

    p.add_argument(
        "--send_hz", type=float, default=SEND_HZ, help="Control packet send rate"
    )
    p.add_argument("--capture_fps", type=float, default=CAPTURE_FPS, help="Capture FPS")
    p.add_argument("--infer_fps", type=float, default=INFER_FPS, help="Inference FPS")

    # Throttle hold percentage: A pressed for this fraction of send ticks.
    p.add_argument(
        "--a_hold",
        type=float,
        default=1.0,
        help="Fraction of ticks to hold A (0.0..1.0). Example: 0.95 = 95%% of the time.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="RNG seed for throttle hold pattern (only matters if a_hold < 1.0).",
    )

    # HUD toggles
    p.add_argument("--no_hud", action="store_true", help="Disable HUD window")
    p.add_argument(
        "--hud_scale", type=int, default=HUD_SCALE, help="HUD scale factor (1,2,4,...)"
    )

    return p.parse_args()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main():
    args = parse_args()

    show_hud = not args.no_hud
    hud_scale = max(1, int(args.hud_scale))

    a_hold = clamp01(args.a_hold)
    rng = random.Random(args.seed)

    print("Starting inference...")
    print(f"Model: {args.model}")
    print(
        f"A hold: {a_hold * 100:.1f}%  | SEND_HZ={args.send_hz}  CAPTURE_FPS={args.capture_fps}  INFER_FPS={args.infer_fps}"
    )
    print("Quit: press 'q' in HUD window (if enabled) or Ctrl+C.\n")

    model = tf.keras.models.load_model(args.model, compile=False)

    shared = {}
    lock = threading.Lock()
    stop_evt = threading.Event()

    vision = VisionInferenceThread(
        cam_index=args.obs_index,
        model=model,
        stop_evt=stop_evt,
        shared=shared,
        lock=lock,
        capture_fps=args.capture_fps,
        infer_fps=args.infer_fps,
    )
    vision.start()

    bridge = UDPBridge(ip=args.ip, port=args.port)

    period = 1.0 / float(args.send_hz)
    next_t = time.monotonic()

    # HUD FPS
    hud_timer = time.monotonic()
    hud_count = 0
    hud_fps = 0

    # simple rate-limited logging
    last_log_s = -1

    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            with lock:
                lx = float(shared.get("lx", 0.0))
                ly = float(shared.get("ly", 0.0))
                ok = bool(shared.get("ok", False))
                hud_rgb = shared.get("hud_rgb", None)

            if not ok:
                lx, ly = 0.0, 0.0

            # Percentage throttle: press A on this tick with probability a_hold.
            # If a_hold==1.0 -> always pressed; if 0.0 -> never pressed.
            a_pressed = 1 if (a_hold >= 1.0 or rng.random() < a_hold) else 0

            pkt = build_packet(
                lx=lx,
                ly=0,
                a=a_pressed,
                b=0,
                x=0,
                y=0,
                drift=0,
                pause=0,
            )
            bridge.send(pkt)

            sec = int(time.monotonic())
            if sec != last_log_s:
                last_log_s = sec
                print(f"lx={lx:+.3f} ok={ok} a={a_pressed}")

            if show_hud:
                if hud_rgb is not None:
                    bgr = cv2.cvtColor(hud_rgb, cv2.COLOR_RGB2BGR)
                    if hud_scale != 1:
                        bgr = cv2.resize(
                            bgr,
                            (bgr.shape[1] * hud_scale, bgr.shape[0] * hud_scale),
                            interpolation=cv2.INTER_NEAREST,
                        )
                else:
                    bgr = (
                        255
                        * (cv2.UMat(90 * hud_scale, 160 * hud_scale, cv2.CV_8UC3)).get()
                    )
                    bgr[:] = 0

                hud_count += 1
                t = time.monotonic()
                if t - hud_timer >= 1.0:
                    hud_fps = hud_count
                    hud_count = 0
                    hud_timer = t

                cv2.putText(
                    bgr,
                    "MODE: MODEL",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bgr,
                    f"lx={lx:+.2f}  ly={ly:+.2f}",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bgr,
                    f"infer={'OK' if ok else 'NO'}  a={a_pressed}  a_hold={a_hold:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bgr,
                    f"hud_fps={hud_fps}",
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bgr,
                    "q=quit",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Inference HUD", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        if show_hud:
            cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
