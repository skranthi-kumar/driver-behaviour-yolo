"""
Driver Behaviour Detection — Webcam Inference
Classes: Awake, Phone, Seatbelt, drowsy

Requirements:
    pip install ultralytics opencv-python

Usage:
    python webcam_inference.py
    python webcam_inference.py --model path/to/best.pt
    python webcam_inference.py --source path/to/video.mp4
"""

import argparse
import time
import cv2
from ultralytics import YOLO
from collections import deque

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "weights/yolo26n_driver_4cls.pt"
CONF_THRESH  = 0.45            # detection confidence threshold
DROWSY_ALERT = 2.0             # seconds of continuous drowsy before alert
# ─────────────────────────────────────────────────────────────────────────────

# Colors per class (BGR)
CLASS_COLORS = {
    "Awake":    (80, 200, 80),    # green
    "Phone":    (0, 165, 255),    # orange
    "Seatbelt": (255, 200, 0),    # cyan-yellow
    "drowsy":   (0, 0, 220),      # red  ← most critical
}

ALERT_MESSAGES = {
    "drowsy":   "⚠ DROWSINESS DETECTED",
    "Phone":    "⚠ PHONE USE DETECTED",
    "Seatbelt": "⚠ NO SEATBELT",
    "Awake":    None,
}


def draw_box(frame, box, label, conf, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_alert(frame, message, color=(0, 0, 220)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, message, (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def run(model_path, source, conf):
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")
    print("Press Q to quit | Press S to save screenshot\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # macOS fix: create named window explicitly before the loop
    win_name = "Driver Behaviour Detection  |  Q = quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    drowsy_start   = None   # when continuous drowsy detection began
    fps_history    = deque(maxlen=30)
    frame_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        results = model(frame, conf=conf, verbose=False)[0]

        active_classes = set()

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            label   = model.names[cls_id]
            conf_v  = float(box.conf[0])
            color   = CLASS_COLORS.get(label, (180, 180, 180))
            draw_box(frame, box.xyxy[0], label, conf_v, color)
            active_classes.add(label)

        # ── Drowsiness timer logic ──────────────────────────────────────────
        now = time.time()
        if "drowsy" in active_classes:
            if drowsy_start is None:
                drowsy_start = now
            elapsed = now - drowsy_start
            if elapsed >= DROWSY_ALERT:
                draw_alert(frame,
                           f"⚠  DROWSY FOR {elapsed:.0f}s — PLEASE STOP",
                           color=(0, 0, 200))
        else:
            drowsy_start = None

        # ── Other alerts (non-drowsy) ───────────────────────────────────────
        for cls in ["Phone", "Seatbelt"]:
            if cls in active_classes and "drowsy" not in active_classes:
                msg = ALERT_MESSAGES[cls]
                if msg:
                    draw_alert(frame, msg, color=(0, 120, 220))
                break

        # ── FPS counter ─────────────────────────────────────────────────────
        fps_history.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        draw_fps(frame, sum(fps_history) / len(fps_history))

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(10) & 0xFF  # 10ms gives macOS time to process events
        if key in (ord('q'), ord('Q'), 27):  # Q or Escape
            break
        elif key == ord('s'):
            fname = f"screenshot_{frame_count:04d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver behaviour webcam inference")
    parser.add_argument("--model",  default=MODEL_PATH, help="Path to best.pt")
    parser.add_argument("--source", default=0,          help="Webcam index or video path (default: 0)")
    parser.add_argument("--conf",   default=CONF_THRESH, type=float, help="Confidence threshold")
    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    run(args.model, source, args.conf)