"""
Driver Behaviour Detection — Web App
Run: python app.py
Then open: http://localhost:5000
"""

import io
import base64
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__, static_folder=".")

MODEL_PATH = "../weights/yolo26n_driver_4cls.pt"
CONF_THRESH = 0.45

# Class colors (RGB for PIL drawing)
CLASS_COLORS = {
    "Awake":    (80, 200, 80),
    "Phone":    (255, 140, 0),
    "Seatbelt": (0, 200, 220),
    "drowsy":   (220, 40, 40),
}

print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print(f"Model ready. Classes: {model.names}")


def draw_results(image_np, results):
    """Draw bounding boxes on image and return annotated image."""
    img = image_np.copy()
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color  = CLASS_COLORS.get(label, (180, 180, 180))
        bgr    = (color[2], color[1], color[0])

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 3)

        # Label background
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), bgr, -1)
        cv2.putText(img, text, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detections.append({
            "label": label,
            "confidence": round(conf * 100, 1),
            "box": [x1, y1, x2, y2]
        })

    return img, detections


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file  = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Run inference
    results = model(img, conf=CONF_THRESH, verbose=False)

    # Draw and encode
    annotated, detections = draw_results(img, results)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": f"data:image/jpeg;base64,{img_b64}",
        "detections": detections,
        "count": len(detections)
    })


if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: {MODEL_PATH} not found in current directory.")
        print("Make sure best.pt is in the same folder as app.py")
    else:
        print("\n🚀 Starting server at http://localhost:5000\n")
        app.run(debug=False, host="0.0.0.0", port=5000)