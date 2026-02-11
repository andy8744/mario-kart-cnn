# control/control_packet.py

import time


def build_packet(lx, ly, a, b, x, y, drift, pause):
    return {
        "lx": lx,
        "ly": ly,
        "a": int(a),
        "b": int(b),
        "x": int(x),
        "y": int(y),
        "drift": int(drift),
        "pause": int(pause),
        "ts": time.time(),  # nxbt expects wall clock
    }
