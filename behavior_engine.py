# behavior_engine.py
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import json
import colorsys


# =========================
# Model definitions
# =========================

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
        feats = self.backbone(clips)          # [B*T, D]
        feats = feats.view(b, t, -1).mean(dim=1)  # [B, D]
        return self.head(feats)               # [B, n_classes]


class TemporalConvNet(nn.Module):
    """Temporal conv head (matches sequence-based training)."""
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
        feats = self.backbone(clips)          # [B*T, D]
        feats = feats.view(b, t, -1)          # [B, T, D]
        feats = feats.transpose(1, 2)         # [B, D, T]
        feats = self.temporal_conv(feats)     # [B, D, T]
        feats = feats.mean(dim=2)             # [B, D]
        return self.head(feats)               # [B, n_classes]


def build_class_color_map(names):
    """Assign deterministic BGR colors per class using evenly spaced HSV hues."""
    total = max(len(names), 1)
    mapping = {}
    for idx, name in enumerate(names):
        hue = idx / total
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        mapping[name] = (int(b * 255), int(g * 255), int(r * 255))
    return mapping


# =========================
# BehaviorEngine
# =========================

class BehaviorEngine:
    """
    Core engine: webcam capture, YOLO detection, temporal model, smoothing,
    Unknown gating, Sleeping stabilization, and per-5-second stats logging.

    GUI uses:
      - read_frame() → (frame_with_overlay, current_stats, num_people)
      - start_record() / stop_record()
      - list_sessions()
    """

    def __init__(self, source=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load checkpoint ---
        script_dir = os.path.dirname(__file__)
        ckpt_file = os.environ.get("BD_CKPT_FILE", "behavior_detection.pth")
        model_path = os.path.join(script_dir, ckpt_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["state_dict"]

        # Detect architecture (with/without temporal conv)
        has_temporal_conv = any(
            k.startswith("temporal_conv") or ".temporal_conv" in k
            for k in state_dict.keys()
        )
        ModelClass = TemporalConvNet if has_temporal_conv else TemporalMeanNet

        self.base_class_names = ckpt["class_names"]  # classes from training (no Unknown)
        self.model = ModelClass(ckpt["model_name"], len(self.base_class_names)).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ---------- Special labels ----------
        # Unknown label if confidence is low
        self.UNKNOWN_LABEL = "Unknown"
        self.UNKNOWN_THRESH = 0.50

        # Sleeping stabilization (we'll match the original *frame-based* logic)
        self.SLEEP_LABEL = "Sleeping"
        self.SLEEP_WINDOW_SEC = 1.5
        self.SLEEP_CONFIRM_THRESHOLD = 0.75
        self.SLEEP_CLEAR_THRESHOLD = 0.6

        # For stats + reports: base classes + Unknown
        self.class_names = list(self.base_class_names) + [self.UNKNOWN_LABEL]
        self.class_colors = build_class_color_map(self.class_names)
        self.class_colors[self.UNKNOWN_LABEL] = (160, 160, 160)  # gray Unknown

        # Fallback color for Sleeping if missing
        if self.SLEEP_LABEL not in self.class_colors:
            self.class_colors[self.SLEEP_LABEL] = (255, 255, 0)

        self.clip_len = int(ckpt["clip_len"])

        print(f"[Engine] Model loaded: {ckpt['model_name']}")
        print(f"[Engine] Base classes: {self.base_class_names}")
        print(f"[Engine] Stats/Report classes (incl. Unknown): {self.class_names}")
        print(f"[Engine] Clip length: {self.clip_len}")
        print(f"[Engine] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Engine] CUDA Device Name: {torch.cuda.get_device_name(0)}")

        # --- Transform for crops ---
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Resize(int(self.img_size * 1.14)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # --- YOLOv8 (person detection + ByteTrack) ---
        self.yolo = YOLO("yolov8n.pt")

        # Detection filters (same spirit as original)
        self.min_w = 28
        self.min_h = 28
        self.min_conf = 0.45
        self.min_aspect = 0.0

        # EMA smoothing
        self.smooth_alpha = 0.5  # SMOOTH = 0.5

        # Per-track temporal buffers + EMA
        self.buffers = defaultdict(lambda: deque(maxlen=self.clip_len))  # tid -> deque(C,H,W)
        self.ema_probs = defaultdict(lambda: None)                       # tid -> np.array
        self.track_display = {}                                          # tid -> (label_txt, color)

        # -------- Sleeping histories (frame-based, like original) --------
        self.sleep_history = defaultdict(lambda: deque())  # tid -> deque[(frame_idx, is_sleep)]
        self.stable_sleep = {}                             # tid -> bool

        # Approx FPS for sleeping window (same as original: 30 fps)
        self.fps_est = 30.0
        self.sleep_window_frames = max(
            1, int(round(self.fps_est * self.SLEEP_WINDOW_SEC))
        )
        print(f"[Engine] Sleep window frames: {self.sleep_window_frames}")

        # Capture & frame counter
        self.cap = cv2.VideoCapture(source)
        self.frame_count = 0
        torch.backends.cudnn.benchmark = True

        # Recording / stats logging
        self.recording = False
        self.session_start_time = None
        self.per_second_stats = []       # list of {"t_sec": int, "dist": {...}}
        self.last_logged_second = None

        # Reports folder
        self.reports_dir = Path(script_dir) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ======================
    # Recording control
    # ======================

    def start_record(self):
        """Start a new recording session and clear stats."""
        print("[Engine] Start recording session")
        self.recording = True
        self.session_start_time = time.time()
        self.per_second_stats.clear()
        self.last_logged_second = None

    def stop_record(self):
        """Stop recording, aggregate stats, save JSON report, return summary."""
        print("[Engine] Stop recording session")
        self.recording = False
        if not self.session_start_time:
            return None

        duration_sec = time.time() - self.session_start_time
        summary = self._aggregate_stats(duration_sec)

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.session_start_time))
        out_path = self.reports_dir / f"session_{ts}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Engine] Session saved: {out_path}")

        return {
            "duration_sec": duration_sec,
            "path": str(out_path),
            "summary": summary,
        }

    def _aggregate_stats(self, duration_sec: float):
        """
        Convert logged snapshots (t=0,5,10,...) into per_5sec entries.
        Each snapshot is exactly the stats at that second.
        """
        if not self.per_second_stats:
            return {
                "duration_sec": duration_sec,
                "per_5sec": [],
                "class_names": self.class_names,
            }

        snapshots = sorted(self.per_second_stats, key=lambda s: s["t_sec"])

        per_5sec = []
        for s in snapshots:
            t = s["t_sec"]
            per_5sec.append({
                "window": t // 5,
                "start_sec": t,
                "dist": s["dist"],
            })

        return {
            "duration_sec": duration_sec,
            "per_5sec": per_5sec,
            "class_names": self.class_names,
        }

    def _log_snapshot(self, current_time, current_stats):
        """
        Log snapshot at times t = 0, 5, 10, ... since session_start_time.
        current_stats includes Unknown and all behaviors.
        """
        if not self.recording or not self.session_start_time:
            return

        t_sec = int(current_time - self.session_start_time)

        if t_sec % 5 != 0:
            return
        if self.last_logged_second is not None and t_sec <= self.last_logged_second:
            return

        self.last_logged_second = t_sec
        dist = {cls: float(current_stats.get(cls, 0.0)) for cls in self.class_names}
        self.per_second_stats.append({"t_sec": t_sec, "dist": dist})

    # ======================
    # Main per-frame API
    # ======================

    def read_frame(self):
        """
        Capture one frame, run YOLO + temporal model, return:
          frame_with_overlay, current_stats, num_people

        current_stats: {class_name: fraction_of_students}, including "Unknown".
        """
        ok, frame = self.cap.read()
        if not ok:
            return None

        self.frame_count += 1

        # Run YOLO + ByteTrack (person class only)
        res = self.yolo.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=self.min_conf,
            iou=0.45,
            device=0 if self.device.type == "cuda" else "cpu",
            verbose=False,
        )

        behavior_counts = defaultdict(int)
        people_this_frame = 0

        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            boxes = res[0].boxes
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                w = x2 - x1
                h = y2 - y1
                det_conf = float(confs[i])
                aspect = h / (w + 1e-6)

                # Filter out low-confidence / tiny detections (like hands)
                if det_conf < self.min_conf:
                    continue
                if w < self.min_w or h < self.min_h:
                    continue
                if self.min_aspect > 0.0 and aspect < self.min_aspect:
                    continue

                people_this_frame += 1

                # Crop and preprocess person
                crop = self._pad_and_crop(frame, x1, y1, x2, y2, pad=24)
                if crop is None:
                    continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor_frame = self.transform(pil_img)
                self.buffers[tid].append(tensor_frame)

                # Previous display (for sleeping smoothing & warm-up)
                prev_label_txt, prev_color = self.track_display.get(
                    tid, ("warming...", (0, 255, 0))
                )
                label_txt = prev_label_txt
                color = prev_color
                final_label_for_stats = None  # what we count for this student

                if len(self.buffers[tid]) == self.clip_len:
                    # Run model on last clip_len frames
                    clip = torch.stack(list(self.buffers[tid]), dim=0)  # (T,C,H,W)
                    clip = clip.unsqueeze(0).to(self.device)            # (1,T,C,H,W)

                    with torch.no_grad():
                        logits = self.model(clip)
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    # EMA smoothing
                    if self.ema_probs[tid] is None:
                        self.ema_probs[tid] = probs
                    else:
                        self.ema_probs[tid] = (
                            self.smooth_alpha * self.ema_probs[tid]
                            + (1.0 - self.smooth_alpha) * probs
                        )

                    smoothed = self.ema_probs[tid]
                    pred_idx = int(np.argmax(smoothed))
                    pred_label = self.base_class_names[pred_idx]
                    pred_conf = float(smoothed[pred_idx])

                    # ----- Unknown gating (display + stats) -----
                    display_label = pred_label
                    display_conf = pred_conf
                    if pred_conf <= self.UNKNOWN_THRESH:
                        display_label = self.UNKNOWN_LABEL

                    # ----- Sleeping stabilization (MATCH ORIGINAL CODE) -----
                    # Original code uses frame_count & sleep_window_frames:
                    #   - history: (frame_idx, is_sleep)
                    #   - ratio over last N frames
                    if self.SLEEP_LABEL in self.base_class_names:
                        history = self.sleep_history[tid]
                        history.append(
                            (self.frame_count, 1 if pred_label == self.SLEEP_LABEL else 0)
                        )
                        cutoff = self.frame_count - self.sleep_window_frames
                        while history and history[0][0] < cutoff:
                            history.popleft()

                        sleep_ratio = (
                            float(sum(v for _, v in history)) / len(history)
                            if history else 0.0
                        )
                        window_covered = (
                            len(history) > 1
                            and (history[-1][0] - history[0][0]) >= self.sleep_window_frames
                        )

                        was_stable = self.stable_sleep.get(tid, False)
                        if window_covered and sleep_ratio >= self.SLEEP_CONFIRM_THRESHOLD:
                            self.stable_sleep[tid] = True
                        elif sleep_ratio < self.SLEEP_CLEAR_THRESHOLD:
                            self.stable_sleep[tid] = False
                        is_stable = self.stable_sleep.get(tid, False)

                        if is_stable:
                            # "Locked" as Sleeping, like original: show Sleep label + ratio
                            display_label = self.SLEEP_LABEL
                            display_conf = sleep_ratio
                        elif pred_label == self.SLEEP_LABEL:
                            # Still building evidence: keep *previous* label to avoid flicker
                            display_label = prev_label_txt.split(":")[0]

                    # Final overlay text + color
                    label_txt_curr = f"{display_label}: {display_conf:.2f}"
                    color_curr = self.class_colors.get(display_label, (0, 255, 0))

                    label_txt = label_txt_curr
                    color = color_curr
                    self.track_display[tid] = (label_txt, color)

                    # For stats: count the final display label (can be Unknown or Sleeping)
                    final_label_for_stats = display_label

                else:
                    # Still warming up clip; keep previous display
                    self.track_display.setdefault(tid, (label_txt, color))

                # Count this student's current behavior into stats, if known
                if final_label_for_stats is not None:
                    behavior_counts[final_label_for_stats] += 1

                # Draw bounding box + label
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                cv2.putText(
                    frame,
                    f"ID {tid} | {label_txt}",
                    (x1i, max(20, y1i - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # ---------- Build current_stats for sidebar & logging ----------
        total_now = sum(behavior_counts.values())
        if total_now <= 0:
            current_stats = {cls: 0.0 for cls in self.class_names}
        else:
            current_stats = {
                cls: behavior_counts.get(cls, 0) / total_now
                for cls in self.class_names
            }

        # Snapshot for report (every 5 seconds)
        self._log_snapshot(time.time(), current_stats)

        return frame, current_stats, people_this_frame

    # ======================
    # Helpers
    # ======================

    def _pad_and_crop(self, frame, x1, y1, x2, y2, pad=24):
        """
        Crop a padded, approximately square region around the box,
        clamped to frame boundaries.
        """
        H, W = frame.shape[:2]
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        side = max(w, h) + 2 * pad
        half = side / 2.0

        x1p = int(np.floor(cx - half))
        y1p = int(np.floor(cy - half))
        x2p = int(np.ceil(cx + half))
        y2p = int(np.ceil(cy + half))

        x1p = max(0, x1p)
        y1p = max(0, y1p)
        x2p = min(W, x2p)
        y2p = min(H, y2p)

        if x2p <= x1p or y2p <= y1p:
            return None
        return frame[y1p:y2p, x1p:x2p]

    def release(self):
        """Release camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ======================
    # For reports page
    # ======================

    def list_sessions(self):
        """
        Return list of (filename, full_path) for saved session JSONs.
        Used by ReportsPage to populate the session list.
        """
        files = sorted(self.reports_dir.glob("session_*.json"))
        return [(f.name, str(f)) for f in files]
