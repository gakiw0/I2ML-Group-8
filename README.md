# Classroom Behavior Monitor (I2ML Group 8)

Real-time classroom behavior monitoring with a desktop GUI. The app detects people in a webcam feed (YOLOv8) and classifies each student's behavior using a temporal deep-learning model, then logs session statistics and visualizes them in a reports view.

## Features

- Live webcam monitoring with per-person bounding boxes and behavior labels
- Camera selection + FPS control from the GUI
- Session recording (writes a timestamped JSON report)
- Reports page with a timeline chart (per 5 seconds) + overall distribution pie chart
- `Unknown` gating and temporal smoothing for more stable predictions

## How it works (high level)

1. **Person detection:** `ultralytics` YOLO detects people in each frame.
2. **Tracking + cropping:** per-person crops are collected into short clips.
3. **Behavior classification:** a temporal model (ConvNeXt backbone + temporal head) predicts the behavior class for each clip.
4. **Aggregation:** class percentages are logged every 5 seconds and saved as a session report.

Behavior labels are defined by the checkpoint (stored in `behavior_detection.pth`). Typical labels in this project include: `Looking_Forward`, `Turning_Around`, `Raising_Hand`, `Reading`, `Writing`, `Sleeping` (the app may also surface an `Unknown` label depending on configuration).

## Quick start

### Prerequisites

- Python 3.10+ recommended
- A webcam (or virtual camera)
- Packages used: `PyQt5`, `opencv-python`, `numpy`, `matplotlib`, `torch`, `torchvision`, `timm`, `Pillow`, `ultralytics`

### Download the model checkpoint (required)

Download it from:

https://drive.google.com/file/d/1iaIj_fT7FGueLlmRBtyNZhAPhCleI0Ye/view?usp=sharing

Then place the downloaded file in the project root (same folder as `behavior_engine.py`) and name it `behavior_detection.pth`.

### Install dependencies (example)

```bash
pip install pyqt5 opencv-python numpy matplotlib pillow timm ultralytics
```

`torch` / `torchvision` installation depends on your OS and whether you want CUDA; install a compatible build for your machine.

### Run the GUI

```bash
python app_main.py
```

In the app:

1. Click **Start**
2. Choose a camera and FPS (top controls)
3. Use **Start Record** / **Stop Record** to save a session
4. Open **Reports** to review saved sessions

## Outputs

- Session reports are saved to `reports/session_YYYYMMDD_HHMMSS.json`.
- Reports include:
  - `class_names` (may include `Unknown`)
  - `per_5sec` snapshots (distribution per 5-second window)
  - `duration_sec`

## Configuration

- Use a different checkpoint file name via environment variable:
  - Windows PowerShell: `setx BD_CKPT_FILE "your_checkpoint.pth"`
  - Current shell only: `$env:BD_CKPT_FILE="your_checkpoint.pth"`

The checkpoint is expected to be located next to `behavior_engine.py` (same directory as the app).

## Project structure

- `app_main.py` — PyQt5 GUI (Home / Monitor / Reports)
- `behavior_engine.py` — core logic (camera, YOLO, behavior model, smoothing, recording)
- `behavior_detection.ipynb` — training / experimentation notebook (behavior classifier)
- `behavior_detection.pth` — trained behavior checkpoint used by the app
- `yolov8n.pt` — YOLOv8 weights for person detection
- `reports/` — recorded session JSON files
- `imgs/` — sample frames used during development

## Troubleshooting

- **No camera feed / black screen:** try a different camera in the dropdown, click **Refresh**, or lower the FPS.
- **Slow performance on CPU:** reduce FPS and/or resolution; GPU acceleration is used automatically if PyTorch detects CUDA.
- **Missing weights error:** ensure `behavior_detection.pth` and `yolov8n.pt` exist in the project root.

## Notes

This is a course/project prototype. If you use it with real people, follow local privacy rules and obtain consent.
