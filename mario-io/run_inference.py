import time
import threading
import cv2

from control.udp_bridge import UDPBridge
from control.control_packet import build_packet

from inference.random_cnn import build_random_policy_model
from inference.vision_thread import VisionInferenceThread


# ---- CONFIG ----
OBS_INDEX = 0
VM_IP = "192.168.64.2"
PORT = 5005

SEND_HZ = 120
CAPTURE_FPS = 30
INFER_FPS = 30

HOLD_A = 1

SHOW_HUD = True          # set False for fully headless
HUD_SCALE = 2            # 1=160x90, 2=320x180, 4=640x360
# ----------------


def main():
    print("Starting inference...")
    print("Quit: press 'q' in HUD window (if enabled) or Ctrl+C.\n")

    model = build_random_policy_model(input_shape=(90, 160, 3))

    shared = {}
    lock = threading.Lock()
    stop_evt = threading.Event()

    vision = VisionInferenceThread(
        cam_index=OBS_INDEX,
        model=model,
        stop_evt=stop_evt,
        shared=shared,
        lock=lock,
        capture_fps=CAPTURE_FPS,
        infer_fps=INFER_FPS,
    )
    vision.start()

    bridge = UDPBridge(ip=VM_IP, port=PORT)

    period = 1.0 / float(SEND_HZ)
    next_t = time.monotonic()

    # HUD FPS
    hud_timer = time.monotonic()
    hud_count = 0
    hud_fps = 0

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
                # vision_thread should set this; if not present, HUD just won't show video
                hud_rgb = shared.get("hud_rgb", None)

            if not ok:
                lx, ly = 0.0, 0.0

            pkt = build_packet(
                lx=lx,
                ly=ly,
                a=HOLD_A,
                b=0,
                x=0,
                y=0,
                drift=0,
                pause=0,
            )
            bridge.send(pkt)

            if SHOW_HUD:
                if hud_rgb is not None:
                    # hud_rgb is RGB; OpenCV expects BGR
                    bgr = cv2.cvtColor(hud_rgb, cv2.COLOR_RGB2BGR)
                    if HUD_SCALE != 1:
                        bgr = cv2.resize(bgr, (bgr.shape[1] * HUD_SCALE, bgr.shape[0] * HUD_SCALE),
                                         interpolation=cv2.INTER_NEAREST)
                else:
                    bgr = 255 * (cv2.UMat(90 * HUD_SCALE, 160 * HUD_SCALE, cv2.CV_8UC3)).get()
                    bgr[:] = 0

                # HUD FPS estimate
                hud_count += 1
                t = time.monotonic()
                if t - hud_timer >= 1.0:
                    hud_fps = hud_count
                    hud_count = 0
                    hud_timer = t

                cv2.putText(bgr, f"MODE: MODEL", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(bgr, f"lx={lx:+.2f}  ly={ly:+.2f}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(bgr, f"infer={'OK' if ok else 'NO'} ", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(bgr, "q=quit", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Inference HUD", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        if SHOW_HUD:
            cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
