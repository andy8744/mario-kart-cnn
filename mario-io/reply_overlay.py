#!/usr/bin/env python3
"""
Replay a recorded run (video.mp4 + labels.jsonl) with controls overlaid.

Usage:
  python3 replay_overlay.py /path/to/run_folder

Hotkeys (video window):
  q  quit
  space  pause/resume
  j  back 30 frames
  k  forward 30 frames

Notes:
- Expects labels.jsonl lines like:
  {"frame_idx": 0, "frame_ts_ns": ..., "lx": ..., "ly": ..., "a": 1, "drift": 0}
- If you have an extra meta line at the top, it will be skipped automatically.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np


def load_labels(labels_path: Path) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip metadata lines if present
            if "frame_idx" not in rec:
                continue
            labels.append(rec)
    labels.sort(key=lambda r: int(r["frame_idx"]))
    return labels


def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def draw_bar(img, x: int, y: int, w: int, h: int, v: float, label: str) -> None:
    """
    Draw a centered horizontal bar representing v in [-1,1].
    """
    v = float(clamp(v))
    # background
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    # center line
    cx = x + w // 2
    cv2.line(img, (cx, y), (cx, y + h), (120, 120, 120), 1)

    # filled portion
    if v >= 0:
        x2 = int(cx + (w // 2) * v)
        cv2.rectangle(img, (cx, y), (x2, y + h), (0, 200, 0), -1)
    else:
        x2 = int(cx + (w // 2) * v)
        cv2.rectangle(img, (x2, y), (cx, y + h), (0, 200, 0), -1)

    cv2.putText(img, f"{label}: {v:+.3f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def draw_toggle(img, x: int, y: int, name: str, on: int) -> None:
    on = int(on)
    color = (0, 200, 0) if on else (80, 80, 80)
    cv2.rectangle(img, (x, y), (x + 80, y + 30), color, -1)
    cv2.putText(img, name, (x + 10, y + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def get_label_for_frame(labels: List[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    if idx < 0 or idx >= len(labels):
        return None
    return labels[idx]


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 replay_overlay.py /path/to/run_folder")
        raise SystemExit(2)

    run_dir = Path(sys.argv[1])
    video_path = run_dir / "video.mp4"
    labels_path = run_dir / "labels.jsonl"

    if not video_path.exists():
        raise SystemExit(f"Missing: {video_path}")
    if not labels_path.exists():
        raise SystemExit(f"Missing: {labels_path}")

    labels = load_labels(labels_path)
    if not labels:
        raise SystemExit("No frame labels found in labels.jsonl")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video frames (reported): {total_frames}, FPS: {fps}")
    print(f"Labels loaded: {len(labels)}")
    if total_frames > 0 and abs(total_frames - len(labels)) > 1:
        print("WARNING: frame count and label count differ. Overlay may be misaligned.")

    paused = False
    frame_idx = 0

    # Seek helper
    def seek_to(idx: int):
        nonlocal frame_idx
        idx = max(0, idx)
        if total_frames > 0:
            idx = min(idx, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        frame_idx = idx

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

        # On pause, we still show the last frame
        # If paused at start, ensure we have a frame
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                break

        # Fetch label aligned by index
        rec = get_label_for_frame(labels, frame_idx)
        lx = float(rec.get("lx", 0.0)) if rec else 0.0
        ly = float(rec.get("ly", 0.0)) if rec else 0.0
        a = int(rec.get("a", 0)) if rec else 0
        drift = int(rec.get("drift", 0)) if rec else 0
        ts_ns = rec.get("frame_ts_ns", None) if rec else None

        overlay = frame.copy()

        # Header
        cv2.putText(overlay, f"frame {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"paused={paused}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if ts_ns is not None:
            cv2.putText(overlay, f"ts_ns={ts_ns}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Bars
        draw_bar(overlay, x=20, y=140, w=380, h=22, v=lx, label="lx")
        draw_bar(overlay, x=20, y=190, w=380, h=22, v=ly, label="ly")

        # Buttons
        draw_toggle(overlay, x=20, y=235, name="A", on=a)
        draw_toggle(overlay, x=110, y=235, name="DRIFT", on=drift)

        # Controls help
        cv2.putText(overlay, "space=pause  j=-30  k=+30  q=quit",
                    (20, overlay.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Replay Overlay", overlay)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("j"):
            seek_to(frame_idx - 30)
            ret, frame = cap.read()
            if not ret:
                break
        elif key == ord("k"):
            seek_to(frame_idx + 30)
            ret, frame = cap.read()
            if not ret:
                break

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
