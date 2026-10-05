# Driver Behaviour Detection (YOLO26-nano)

Real-time in-cabin driver-behaviour detector trained with Ultralytics YOLO26-nano on a 4-class dataset:
**Awake · Phone (phone use) · Seatbelt · drowsy**. Runs on a laptop webcam with a sustained-drowsiness alert,
and ships with a small Flask demo for single-image inference.

Built as the perception prototype behind [Drivos](https://drivos.in), an AI driver-safety project for Indian fleets.


## What it does
- Detects the driver's state per frame: awake, drowsy, using a phone, seatbelt visible.
- Raises a **drowsiness alert after 2 s of continuous `drowsy` detections** (configurable), plus instant alerts for phone use and missing seatbelt.
- Shows a rolling FPS counter so on-device performance can be measured honestly.

## Quick start
```bash
pip install -r requirements.txt
python webcam_inference.py                     # default webcam
python webcam_inference.py --source video.mp4  # any video file
python webcam_inference.py --conf 0.5          # stricter threshold
```
Keys: `Q` quit · `S` save screenshot.

### Browser demo
```bash
cd demo && python app.py     # http://localhost:5000  — upload an image, get annotated boxes + JSON
```

## Model
| File | Architecture | Classes | Size |
|---|---|---|---|
| `weights/yolo26n_driver_4cls.pt` | YOLO26-nano (Ultralytics) | Awake, Phone, Seatbelt, drowsy | 5.4 MB |

Trained on a custom 4-class in-cabin dataset (`dataset_4class`, annotated in Roboflow). Training used the standard
Ultralytics pipeline; the dataset itself is not released.

## Repository layout
```
webcam_inference.py   real-time inference, alert logic, FPS counter
demo/app.py           Flask single-image inference API + page
demo/index.html       upload UI
weights/              trained checkpoint
```

## Limitations
Trained on a small, single-camera dataset; expect degraded accuracy with unusual cabin lighting, sunglasses, or
camera angles far from the training setup. Not a certified safety system.

## License
MIT. Model weights are released under the same license for research and evaluation.
