# control/joycon_device.py

import pygame

DEADZONE = 0.08


def dz(v):
    return 0.0 if abs(v) < DEADZONE else v


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


class JoyConDevice:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")

        self.js = pygame.joystick.Joystick(0)
        self.js.init()

        print("Using:", self.js.get_name())

    def read(self):
        pygame.event.pump()

        lx = dz(self.js.get_axis(0))
        ly = dz(self.js.get_axis(1))

        a = self.js.get_button(0)
        b = self.js.get_button(1)
        x = self.js.get_button(2)
        y = self.js.get_button(3)

        drift = self.js.get_button(10)
        pause = self.js.get_button(6)

        return {
            "lx": clamp(lx),
            "ly": clamp(ly),
            "a": int(a),
            "b": int(b),
            "x": int(x),
            "y": int(y),
            "drift": int(drift),
            "pause": int(pause),
        }
