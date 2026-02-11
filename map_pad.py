import time
import pygame

DEADZONE = 0.08

def dz(x):
    return 0.0 if abs(x) < DEADZONE else x

pygame.init()
pygame.joystick.init()

n = pygame.joystick.get_count()
print(f"Joysticks detected: {n}")
if n == 0:
    raise SystemExit("No controller detected.")

js = pygame.joystick.Joystick(0)
js.init()

print(f"Using: {js.get_name()}")
print(f"Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}, Hats: {js.get_numhats()}")
print("Move one control at a time. Ctrl+C to stop.\n")

prev_axes = [0.0] * js.get_numaxes()
prev_btns = [0] * js.get_numbuttons()
prev_hats = [(0,0)] * js.get_numhats()

while True:
    pygame.event.pump()

    axes = [dz(js.get_axis(i)) for i in range(js.get_numaxes())]
    btns = [js.get_button(i) for i in range(js.get_numbuttons())]
    hats = [js.get_hat(i) for i in range(js.get_numhats())]

    # Print only changes
    for i, (a, pa) in enumerate(zip(axes, prev_axes)):
        if abs(a - pa) > 0.05:
            print(f"axis[{i}] = {a:+.2f}")
    for i, (b, pb) in enumerate(zip(btns, prev_btns)):
        if b != pb:
            print(f"button[{i}] = {b}")
    for i, (h, ph) in enumerate(zip(hats, prev_hats)):
        if h != ph:
            print(f"hat[{i}] = {h}")

    prev_axes, prev_btns, prev_hats = axes, btns, hats
    time.sleep(0.02)
