import time
import math

from control.udp_bridge import UDPBridge
from control.control_packet import build_packet


# ---- CONFIG ----
VM_IP = "192.168.64.2"
PORT = 5005

SEND_HZ = 120
HOLD_A = 1

STEER_AMPLITUDE = 0.7     # max steering magnitude (0–1)
STEER_FREQUENCY = 0.5     # Hz (cycles per second)
# ----------------


def main():
    print("Starting sine driver...")
    print("Ctrl+C to quit.\n")

    bridge = UDPBridge(ip=VM_IP, port=PORT)

    period = 1.0 / float(SEND_HZ)
    next_t = time.monotonic()

    start_time = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period

            t = now - start_time

            # Sine steering
            lx = STEER_AMPLITUDE * math.sin(2 * math.pi * STEER_FREQUENCY * t)
            ly = 0.0

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

    except KeyboardInterrupt:
        print("\nStopping sine driver.")


if __name__ == "__main__":
    main()
