import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def load_labels_jsonl(labels_path: Path):
    records = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # expected format from your writer:
            # { frame_idx, frame_ts_ns, lx, ly, a, drift ... } OR { frame_idx, frame_ts_ns, control:{...} }
            if "lx" in rec:
                lx, ly = float(rec["lx"]), float(rec["ly"])
                frame_idx = int(rec["frame_idx"])
            else:
                c = rec["control"]
                lx, ly = float(c["lx"]), float(c["ly"])
                frame_idx = int(rec["frame_idx"])
            records.append((frame_idx, lx, ly))
    records.sort(key=lambda x: x[0])
    return records


def preprocess_bgr(frame_bgr, out_w=160, out_h=90):
    # BGR -> RGB, resize, normalize [0,1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return x


def build_model(input_shape=(90, 160, 3)):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(24, 5, strides=2, activation="relu")(inp)
    x = tf.keras.layers.Conv2D(36, 5, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(48, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # keep small; you want it to overfit track
    out = tf.keras.layers.Dense(2, activation="tanh")(x)  # [-1,1]
    return tf.keras.Model(inp, out)


def make_dataset(video_path: Path, labels, batch_size=64, shuffle=True, seed=1337, out_w=160, out_h=90):
    """
    Loads frames sequentially from video.mp4 and yields (image, [lx,ly]).
    Assumes labels are in ascending frame_idx, one label per saved frame.
    """
    # We'll precompute target array aligned to video frame order
    targets = np.array([[lx, ly] for (_, lx, ly) in labels], dtype=np.float32)

    def gen():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        i = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i >= len(targets):
                    break
                x = preprocess_bgr(frame, out_w=out_w, out_h=out_h)
                y = targets[i]
                yield x, y
                i += 1
        finally:
            cap.release()

    output_sig = (
        tf.TensorSpec(shape=(out_h, out_w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)

    if shuffle:
        ds = ds.shuffle(2000, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(targets)


def split_labels(labels, val_ratio=0.1):
    n = len(labels)
    if n < 50:
        # tiny dataset; still do a small split
        val_n = max(5, int(n * val_ratio))
    else:
        val_n = int(n * val_ratio)
    val_n = max(1, min(val_n, n - 1))
    train = labels[:-val_n]
    val = labels[-val_n:]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run_YYYYMMDD_HHMMSS folder containing video.mp4 and labels.jsonl")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out_w", type=int, default=160)
    ap.add_argument("--out_h", type=int, default=90)
    ap.add_argument("--model_out", default="model_trackfit.keras")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    video_path = run_dir / "video.mp4"
    labels_path = run_dir / "labels.jsonl"
    if not video_path.exists() or not labels_path.exists():
        raise SystemExit(f"Missing video.mp4 or labels.jsonl in {run_dir}")

    labels = load_labels_jsonl(labels_path)
    if len(labels) < 10:
        raise SystemExit(f"Too few samples in labels.jsonl: {len(labels)}")

    # Sanity check: video frame count vs labels count
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"[info] labels: {len(labels)} | video frames (reported): {total_frames}")

    train_labels, val_labels = split_labels(labels, val_ratio=args.val_ratio)
    print(f"[info] train: {len(train_labels)} | val: {len(val_labels)}")

    train_ds, n_train = make_dataset(video_path, train_labels, batch_size=args.batch, shuffle=True, out_w=args.out_w, out_h=args.out_h)
    val_ds, n_val = make_dataset(video_path, val_labels, batch_size=args.batch, shuffle=False, out_w=args.out_w, out_h=args.out_h)

    model = build_model(input_shape=(args.out_h, args.out_w, 3))
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Huber is usually better than raw MSE for steering noise
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=0.2), metrics=[tf.keras.metrics.MAE])

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_out,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"[done] best model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
