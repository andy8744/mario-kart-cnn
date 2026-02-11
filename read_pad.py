import time
import pygame

def main():
    pygame.init()
    pygame.joystick.init()

    n = pygame.joystick.get_count()
    print(f"Joysticks detected: {n}")
    if n == 0:
        print("No controller detected. Pair/connect it first, then rerun.")
        return

    # Pick the first controller
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Using: {js.get_name()}")
    print(f"Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}, Hats: {js.get_numhats()}")
    print("Move sticks / press buttons. Ctrl+C to stop.\n")

    while True:
        # Pump event queue so joystick state updates
        pygame.event.pump()

        axes = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
        buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
        hats = [js.get_hat(i) for i in range(js.get_numhats())]

        print(f"axes={axes} buttons={buttons} hats={hats}")
        time.sleep(0.05)

if __name__ == "__main__":
    main()
