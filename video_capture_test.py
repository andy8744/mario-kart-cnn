import cv2
import time

WIDTH = 1280
HEIGHT = 720
FPS = 60
MAX_INDEX = 6


def test_index(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    time.sleep(0.5)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    return cap


def select_device():
    print("Scanning OpenCV camera indices...\n")

    for idx in range(MAX_INDEX):
        cap = test_index(idx)
        if cap is None:
            continue

        print(f"Testing index {idx}")
        print("Press 'y' if this is OBS Virtual Camera")
        print("Press 'n' to try next")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(
                display,
                f"OpenCV index: {idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Device Selection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("y"):
                cv2.destroyAllWindows()
                print(f"Selected index {idx}\n")
                return idx
            elif key == ord("n"):
                break

        cap.release()
        cv2.destroyAllWindows()

    raise RuntimeError("No suitable camera selected.")


def main():
    idx = select_device()

    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print("Running OBS Virtual Camera preview.")
    print("Press 'q' to quit.\n")

    fps_timer = time.monotonic()
    frame_count = 0
    measured_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        frame_count += 1
        now = time.monotonic()

        if now - fps_timer >= 1.0:
            measured_fps = frame_count
            frame_count = 0
            fps_timer = now

        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"OBS Virtual Camera (index {idx})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"Resolution: {overlay.shape[1]}x{overlay.shape[0]}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"Measured FPS: {measured_fps}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Press q to quit",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("OBS Virtual Camera Test", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
