import json
import socket
import time
import pygame

VM_IP = "192.168.64.2"
PORT = 5005
HZ = 120
DEADZONE = 0.08

def dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v

def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise SystemExit("No joystick detected. Pair/connect the Joy-Con first.")

    js = pygame.joystick.Joystick(0)
    js.init()

    print("Using:", js.get_name())
    print(f"Streaming UDP to {VM_IP}:{PORT} @ {HZ}Hz")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        pygame.event.pump()

        # Your mapping
        lx = dz(js.get_axis(0))             # left/right
        ly = dz(js.get_axis(1))             # up/down (up=-1)

        # Raw physical buttons (your discovered indices)
        a = js.get_button(0)  # physical A
        b = js.get_button(1)  # physical B
        x = js.get_button(2)  # physical X
        y = js.get_button(3)  # physical Y

        # Additional buttons
        phys_drift = js.get_button(10)   # drift (R)
        phys_pause = js.get_button(6)  # plus / pause
        phys_SL = js.get_button(9)

        pkt = {
            "lx": clamp(lx),
            "ly": clamp(ly),
            "a": int(a),
            "b": int(b),
            "x": int(x),
            "y": int(y),

            # auxiliary (logged, not used in v0 policy)
            "drift": int(phys_drift),
            "pause": int(phys_pause),
	    "SL": int(phys_SL),

            "ts": time.time(),
        }

        sock.sendto(json.dumps(pkt).encode("utf-8"), (VM_IP, PORT))
        time.sleep(1.0 / HZ)

if __name__ == "__main__":
    main()
