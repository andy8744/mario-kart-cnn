# dataset_writer.py

import cv2
import json
import time
from pathlib import Path


class DatasetWriter:
    def __init__(self, root="datasets", width=1280, height=720, fps=30):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

        self.run_dir = self._create_new_run_dir()
        print(f"[dataset] Writing to {self.run_dir}")

        # Video writer
        video_path = self.run_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.video = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height),
        )

        if not self.video.isOpened():
            raise RuntimeError("Failed to open video writer")

        self.labels = open(self.run_dir / "labels.jsonl", "w", encoding="utf-8")
        self.frame_idx = 0

        # Write metadata file
        meta = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "width": width,
            "height": height,
            "fps": fps,
        }

        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _create_new_run_dir(self):
        while True:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_dir = self.root / f"run_{ts}"
            if not run_dir.exists():
                run_dir.mkdir()
                return run_dir
            time.sleep(1)  # wait one second to avoid collision

    # def write(self, frame, frame_ts_ns, control):
    #     record = {
    #         "frame_idx": self.frame_idx,
    #         "frame_ts_ns": frame_ts_ns,
    #         "control": control,
    #     }

    def write(self, frame, frame_ts_ns, control):
        record = {
            "frame_idx": self.frame_idx,
            "frame_ts_ns": frame_ts_ns,
            "lx": control["lx"],
            "ly": control["ly"],
            "a": control["a"],
            "drift": control["drift"],
        }

        self.video.write(frame)
        self.labels.write(json.dumps(record) + "\n")

        self.frame_idx += 1

    def close(self):
        print(f"[dataset] Closing run {self.run_dir}")
        self.video.release()
        self.labels.close()
