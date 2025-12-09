import os
import colorsys
import cv2
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import defaultdict, deque
import numpy as np

# --- ui & detection config---
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 255, 217)
COLOR_ALERT = (0, 0, 255)
COLOR_OK = (0, 255, 0)

def draw_dashboard(frame, active_tracks, track_display, fps):
    """Draws a professional-looking dashboard overlay on the frame."""
    h, w = frame.shape[:2]
    
    # here is sidebar
    sidebar_w = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), COLOR_BG, -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # sidebar content
    x_start = w - sidebar_w + 20
    y_curr = 50
    
    # title
    cv2.putText(frame, "BEHAVIOR MONITOR", (x_start, y_curr), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
    y_curr += 40
    
    # sidebar's stars
    num_students = len(active_tracks)
    cv2.putText(frame, f"Students Detected: {num_students}", (x_start, y_curr), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    y_curr += 30
    cv2.putText(frame, f"FPS: {fps:.1f}", (x_start, y_curr), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    y_curr += 40

    cv2.line(frame, (x_start, y_curr), (w - 20, y_curr), (100, 100, 100), 1)
    y_curr += 30

    # all active behaviours its detecting
    cv2.putText(frame, "Active Behaviors:", (x_start, y_curr), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    y_curr += 30

    # counter
    behavior_counts = defaultdict(int)
    sleeping_students = 0
    
    for tid in active_tracks:
        label_txt, _ = track_display.get(tid, ("Unknown", (0,0,0)))
        label = label_txt.split(":")[0]
        behavior_counts[label] += 1
        if label == "Sleeping":
            sleeping_students += 1

    for label, count in behavior_counts.items():
        color = COLOR_OK
        if label in ["Sleeping", "Turning_Around"]:
            color = COLOR_ALERT
        
        text = f"{label}: {count}"
        cv2.putText(frame, text, (x_start, y_curr), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_curr += 25

    # alert if sleeping because funny
    if sleeping_students > 0:
        alert_text = "ALERT: SLEEPING"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        cv2.rectangle(frame, (50, 50), (50 + text_size[0] + 20, 50 + text_size[1] + 20), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (60, 50 + text_size[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

class TemporalMeanNet(nn.Module):
    """Mean-pooling temporal head (older checkpoints)."""
    def __init__(self, backbone_name: str, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=0.2,
        )
        self.embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(p=0.2),
            nn.Linear(self.embed_dim, n_classes),
        )

    def forward(self, clips):
        # clips: [B, T, C, H, W]
        b, t, c, h, w = clips.shape
        clips = clips.view(b * t, c, h, w)
        feats = self.backbone(clips)
        feats = feats.view(b, t, -1).mean(dim=1)
        return self.head(feats)


class TemporalConvNet(nn.Module):
    """Temporal conv head (matches behavior_detection.py training)."""
    def __init__(self, backbone_name: str, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=0.2,
        )
        self.embed_dim = self.backbone.num_features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(p=0.2),
            nn.Linear(self.embed_dim, n_classes),
        )

    def forward(self, clips):
        # clips: [B, T, C, H, W]
        b, t, c, h, w = clips.shape
        clips = clips.view(b * t, c, h, w)
        feats = self.backbone(clips)
        feats = feats.view(b, t, -1)
        feats = feats.transpose(1, 2)                  # (B, D, T)
        feats = self.temporal_conv(feats)              # temporal conv
        feats = feats.mean(dim=2)                      # temporal mean
        return self.head(feats)

# ========== Device & checkpoint ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(__file__)
CKPT_FILE = os.environ.get("BD_CKPT_FILE", "behavior_detection.pth")
MODEL_PATH = os.path.join(
    script_dir,
    CKPT_FILE,
)
ckpt = torch.load(MODEL_PATH, map_location=device)

# Match architecture to checkpoint
state_dict = ckpt["state_dict"]
has_temporal_conv = any(k.startswith("temporal_conv") or ".temporal_conv" in k for k in state_dict)
ModelClass = TemporalConvNet if has_temporal_conv else TemporalMeanNet
model = ModelClass(ckpt["model_name"], len(ckpt["class_names"])).to(device)
model.load_state_dict(state_dict)
if not has_temporal_conv:
    print("Warning: checkpoint lacks temporal_conv layers; using mean-pooling head.")
model.eval()

class_names = ckpt["class_names"]


def build_class_color_map(names):
    """Assign deterministic BGR colors per class using evenly spaced HSV hues."""
    total = max(len(names), 1)
    mapping = {}
    for idx, name in enumerate(names):
        hue = idx / total
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        mapping[name] = (int(b * 255), int(g * 255), int(r * 255))
    return mapping


class_colors = build_class_color_map(class_names)
UNKNOWN_LABEL = "unknown"
UNKNOWN_THRESH = float(os.environ.get("BD_UNKNOWN_THRESH", "0.50"))
UNKNOWN_COLOR = (160, 160, 160)
class_colors.setdefault(UNKNOWN_LABEL, UNKNOWN_COLOR)
CLIP_LEN = int(ckpt["clip_len"])
print(f"Model loaded: {ckpt['model_name']}")
print(f"Classes: {class_names}")
print(f"Class colors (BGR): {class_colors}")
print(f"Clip length: {CLIP_LEN}")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# ========== Preprocess ==========
IMG_SIZE = 224
# Class-wise bias: boost Reading/Writing, dampen Sleeping
CLASS_BIASES = {
    "Sleeping": 1,
    "Reading": 1,
    "Writing": 1,
}
BIAS_IDXS = {label: class_names.index(label) for label in CLASS_BIASES if label in class_names}
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ========== YOLO + ByteTrack for multi-person ==========
from ultralytics import YOLO
# yolov8s.pt is a tad more accurate but slightly slower.
yolo = YOLO("yolov8n.pt")

# Per-track circular buffers + EMA smoothing
buffers = defaultdict(lambda: deque(maxlen=CLIP_LEN))    # track_id -> deque of preprocessed frames (C,H,W)
ema_probs = defaultdict(lambda: None)                    # track_id -> np.array([num_classes])
track_display = {}                                       # track_id -> (label_txt, color)
next_classify = {}                                       # track_id -> frame_count when next classification should run
active_tracks = {}                                       # track_id -> last known box (x1,y1,x2,y2)
sleep_history = defaultdict(deque)                       # track_id -> deque[(frame_idx, sleeping_flag)]
stable_sleep = {}                                        # track_id -> bool
SMOOTH = 0.5                                             # slightly lower to switch faster
MIN_BOX = 28                                             # ignore tiny detections (width/height) to reduce noise
YOLO_CONF = 0.45                                         # higher conf to avoid duplicate ghosts
YOLO_IOU = 0.45                                          # stricter IoU for NMS suppression
DETECT_INTERVAL = 2                                      # run YOLO detection/tracking every N frames (lower = more frequent)
SLEEP_LABEL = "Sleeping"
SLEEP_WINDOW_SEC = 3.0
SLEEP_CONFIRM_THRESHOLD = 0.8
SLEEP_CLEAR_THRESHOLD = 0.5

# ========== Video source ==========
SOURCE = 2
cap = cv2.VideoCapture(SOURCE)
# cap = cv2.VideoCapture("http://your-ip-camera/video")

# Try to match capture to camera resolution (override via env vars)
desired_w = int(os.environ.get("BD_CAM_WIDTH", "0"))
desired_h = int(os.environ.get("BD_CAM_HEIGHT", "0"))
if desired_w > 0 and desired_h > 0:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1e-2:
    fps = 30.0
sleep_window_frames = max(1, int(round(fps * SLEEP_WINDOW_SEC)))
if actual_w and actual_h:
    print(f"Camera capture resolution: {actual_w}x{actual_h}")
else:
    print("Camera capture resolution: unknown (will infer after first frame)")
print(f"FPS estimate: {fps:.2f} -> sleep window frames: {sleep_window_frames}")


torch.backends.cudnn.benchmark = True
WINDOW_NAME = "Multi-Person Behavior Detection (YOLO + TemporalMeanNet)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
print("Press 'q' to quit.")

def pad_and_crop(frame, x1, y1, x2, y2, pad=24):
    """Crop around a box, pad generously, and make the crop closer to square for stable resize."""
    H, W = frame.shape[:2]
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = max(w, h) + 2 * pad               # expand to a square-ish region with extra context
    half = side / 2.0
    x1p = int(np.floor(cx - half)); y1p = int(np.floor(cy - half))
    x2p = int(np.ceil(cx + half));  y2p = int(np.ceil(cy + half))

    # Clamp to frame boundaries
    x1p = max(0, x1p); y1p = max(0, y1p)
    x2p = min(W, x2p); y2p = min(H, y2p)

    if x2p <= x1p or y2p <= y1p:  # degenerate
        return None
    return frame[y1p:y2p, x1p:x2p]

frame_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1

    # Run YOLO tracking only every DETECT_INTERVAL frames to reduce load.
    # Between detections, reuse last known boxes/IDs and keep classifying to refresh labels.
    do_detect = (frame_count % DETECT_INTERVAL == 1)
    if do_detect:
        res = yolo.track(
            source=frame,
            persist=True,
            classes=[0],           # person only
            tracker="bytetrack.yaml",
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
        )
        new_tracks = {}
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            ids_detected = res[0].boxes.id.int().cpu().tolist()
            xyxy_detected = res[0].boxes.xyxy.cpu().numpy()
            for i, tid in enumerate(ids_detected):
                new_tracks[tid] = xyxy_detected[i]
            # Drop stale per-ID state for tracks no longer present
            current_ids = set(ids_detected)
            for tid in list(buffers.keys()):
                if tid not in current_ids:
                    buffers.pop(tid, None)
                    ema_probs.pop(tid, None)
                    next_classify.pop(tid, None)
                    track_display.pop(tid, None)
                    sleep_history.pop(tid, None)
                    stable_sleep.pop(tid, None)
        # If detection succeeded, replace tracks; otherwise keep previous tracks
        if new_tracks:
            active_tracks = new_tracks
    # If not detecting this frame, just reuse last active_tracks (if any)

    # Draw and classify per track
    if active_tracks:
        ids = list(active_tracks.keys())
        xyxy = [active_tracks[tid] for tid in ids]
        # Adapt classification interval based on current load:
        # run each ID once every N frames where N = max(1, number of tracked IDs), staggered per-ID
        interval = max(1, len(ids))

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i]
            w, h = x2 - x1, y2 - y1
            if w < MIN_BOX or h < MIN_BOX:
                continue

            crop = pad_and_crop(frame, x1, y1, x2, y2, pad=24)
            if crop is None:
                continue

            # Preprocess person crop using your exact transforms
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor_frame = transform(pil_img)           # (C,H,W)
            buffers[tid].append(tensor_frame)

            prev_label_txt, prev_color = track_display.get(tid, ("warming...", (0, 255, 0)))
            label_txt, color = prev_label_txt, prev_color

            # Initialize a per-ID schedule to stagger classifications (round robin)
            if tid not in next_classify:
                next_classify[tid] = frame_count + (tid % interval)

            should_classify = len(buffers[tid]) == CLIP_LEN and frame_count >= next_classify[tid]

            # When we have a full clip for this track, run your model on its scheduled turn
            if should_classify:
                clip = torch.stack(list(buffers[tid]), dim=0)    # (T,C,H,W)
                clip = clip.unsqueeze(0).to(device)              # (1,T,C,H,W)
                with torch.no_grad():
                    logits = model(clip)
                    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

                # EMA smoothing to stabilize labels
                if ema_probs[tid] is None:
                    ema_probs[tid] = probs
                else:
                    ema_probs[tid] = SMOOTH * ema_probs[tid] + (1.0 - SMOOTH) * probs

                smoothed = ema_probs[tid]
                adjusted = smoothed
                if BIAS_IDXS:
                    # Copy to avoid mutating EMA state
                    adjusted = smoothed.copy()
                    for label, idx in BIAS_IDXS.items():
                        adjusted[idx] *= CLASS_BIASES[label]

                pred_idx = int(np.argmax(adjusted))
                pred_label = class_names[pred_idx]
                pred_conf = float(smoothed[pred_idx])

                display_label = pred_label
                display_conf = pred_conf
                if pred_conf <= UNKNOWN_THRESH:
                    display_label = UNKNOWN_LABEL

                label_txt_curr = f"{display_label}: {display_conf:.2f}"
                color_curr = class_colors.get(display_label, (0, 255, 0))

                if SLEEP_LABEL in class_names:
                    history = sleep_history[tid]
                    history.append((frame_count, 1 if pred_label == SLEEP_LABEL else 0))
                    cutoff = frame_count - sleep_window_frames
                    while history and history[0][0] < cutoff:
                        history.popleft()
                    sleep_ratio = (sum(v for _, v in history) / len(history)) if history else 0.0
                    window_covered = (len(history) > 1 and (history[-1][0] - history[0][0]) >= sleep_window_frames)

                    was_stable = stable_sleep.get(tid, False)
                    if window_covered and sleep_ratio >= SLEEP_CONFIRM_THRESHOLD:
                        stable_sleep[tid] = True
                    elif sleep_ratio < SLEEP_CLEAR_THRESHOLD:
                        stable_sleep[tid] = False
                    is_stable = stable_sleep.get(tid, False)

                    if is_stable:
                        label_txt_curr = f"{SLEEP_LABEL}: {sleep_ratio:.2f}"
                        color_curr = class_colors.get(SLEEP_LABEL, (0, 255, 0))
                    elif pred_label == SLEEP_LABEL:
                        # Keep showing the previous confirmed label while accumulating evidence
                        label_txt_curr = prev_label_txt
                        color_curr = prev_color
                        stable_sleep[tid] = was_stable

                label_txt = label_txt_curr
                color = color_curr
                track_display[tid] = (label_txt, color)
                next_classify[tid] = frame_count + interval
            else:
                # Keep displaying the last known label/color
                track_display.setdefault(tid, (label_txt, color))

            # Draw box + label + ID
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(frame, f"ID {tid} | {label_txt}", (x1i, max(20, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    draw_dashboard(frame, active_tracks, track_display, fps)

    # Show FPS (optional)
    if frame_count == 1:
        fh, fw = frame.shape[:2]
        cv2.resizeWindow(WINDOW_NAME, fw, fh)
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
