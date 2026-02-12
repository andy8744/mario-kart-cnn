import argparse
import json
import random
import socket
import time
import pygame

VM_IP = "192.168.64.2"
PORT = 5005
SEND_HZ = 120
DEADZONE = 0.08

def dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v

def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def parse_args():
    p = argparse.ArgumentParser(description="Joy-Con UDP sender (probabilistic A throttle)")
    p.add_argument("--ip", default=VM_IP, help="UDP target IP")
    p.add_argument("--port", type=int, default=PORT, help="UDP target port")
    p.add_argument("--send_hz", type=float, default=SEND_HZ, help="Send rate (Hz)")

    # Same mechanism as your inference sender: per-tick probability of pressing A
    p.add_argument(
        "--a_hold",
        type=float,
        default=0.60,
        help="Probability of A being pressed on a tick (0.0..1.0). Example: 0.6 = 60%% of ticks.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="RNG seed for the throttle pattern (only matters if a_hold < 1.0).",
    )
    return p.parse_args()

def main():
    args = parse_args()
    a_hold = clamp01(args.a_hold)
    rng = random.Random(args.seed)

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise SystemExit("No joystick detected. Pair/connect the Joy-Con first.")

    js = pygame.joystick.Joystick(0)
    js.init()

    print("Using:", js.get_name())
    print(f"Streaming UDP to {args.ip}:{args.port} @ {args.send_hz}Hz")
    print(f"A hold: {a_hold * 100:.1f}% | seed={args.seed}")
    print("Ctrl+C to quit.\_sr\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    period = 1.0 / float(args.send_hz)
    next_t = time.monotonic()
    last_log_s = -1

    while True:
        now = time.monotonic()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += period

        pygame.event.pump()

        lx = clamp(dz(js.get_axis(0)))
        ly = clamp(dz(js.get_axis(1)))

        # Your button indices
        b = js.get_button(1)   # physical B
        x = js.get_button(2)   # physical X
        y = js.get_button(3)   # physical Y
        phys_drift = js.get_button(10)  # drift (R)
        phys_pause = js.get_button(6)   # plus / pause
        phys_SL = js.get_button(9)

        # Constant probabilistic throttle (no curve)
        a_pressed = 1 if (a_hold >= 1.0 or rng.random() < a_hold) else 0

        pkt = {
            "lx": lx,
            "ly": ly,
            "a": int(a_pressed),
            "b": int(b),
            "x": int(x),
            "y": int(y),
            "drift": int(phys_drift),
            "pause": int(phys_pause),
            "SL": int(phys_SL),
            "ts": time.time(),
        }

        sock.sendto(json.dumps(pkt).encode("utf-8"), (args.ip, args.port))

        sec = int(time.monotonic())
        if sec != last_log_s:
            last_log_s = sec
            print(f"lx={lx:+.3f} ly={ly:+.3f} a={a_pressed} a_hold={a_hold:.2f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
