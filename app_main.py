# app_main.py
import sys
import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QListWidget, QListWidgetItem,
    QMessageBox, QScrollArea, QCheckBox, QGridLayout
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import cv2
import numpy as np

from behavior_engine import BehaviorEngine


# =========================
# Home Page
# =========================

class HomePage(QWidget):
    """
    Simple home screen with 3 options:
      - Start (go to monitor page)
      - Reports (see saved sessions)
      - Exit (quit application)
    """
    def __init__(self, go_start, go_reports, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Classroom Behavior Monitor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        btn_start = QPushButton("Start")
        btn_reports = QPushButton("Reports")
        btn_exit = QPushButton("Exit")

        for b in (btn_start, btn_reports, btn_exit):
            b.setFixedWidth(200)
            b.setMinimumHeight(40)
            b.setStyleSheet("font-size: 16px;")

        btn_start.clicked.connect(go_start)
        btn_reports.clicked.connect(go_reports)
        btn_exit.clicked.connect(QApplication.instance().quit)

        layout.addWidget(title)
        layout.addSpacing(40)
        layout.addWidget(btn_start, alignment=Qt.AlignCenter)
        layout.addWidget(btn_reports, alignment=Qt.AlignCenter)
        layout.addWidget(btn_exit, alignment=Qt.AlignCenter)


# =========================
# Monitor Page
# =========================

class MonitorPage(QWidget):
    """
    Live monitoring page:
      - Left sidebar: session info
      - Center: camera feed with bounding boxes
      - Right sidebar: live behavior statistics
      - Bottom bar: Back, timer, Start/Stop Record
    """
    def __init__(self, engine: BehaviorEngine, go_home, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.go_home = go_home

        self.recording = False
        self.elapsed_sec = 0

        # --- Layouts: top (sidebars + video) + bottom bar ---
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # Left sidebar
        self.left_info = QLabel("Session Info:\n- Source: Webcam\n- Model: Loaded")
        self.left_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.left_info.setMinimumWidth(200)

        # Video area (center)
        self.video_label = QLabel("No video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #202020;")

        # Right sidebar (live stats)
        self.right_stats = QLabel("Stats:\nNo data yet.")
        self.right_stats.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.right_stats.setMinimumWidth(250)

        top_layout.addWidget(self.left_info)
        top_layout.addWidget(self.video_label, stretch=1)
        top_layout.addWidget(self.right_stats)

        # Bottom bar: Back | Timer | Start/Stop button
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self.on_back)

        self.timer_label = QLabel("Time: 00:00")
        self.timer_label.setStyleSheet("font-size: 16px;")

        self.btn_record = QPushButton("Start Record")
        self.btn_record.setMinimumWidth(160)
        self.btn_record.setStyleSheet("font-size: 16px;")
        self.btn_record.clicked.connect(self.toggle_record)

        bottom_layout.addWidget(self.btn_back)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.timer_label)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.btn_record)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        # Timer for video frames
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(30)  # ~30 fps

        # Timer for recording time
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)

    def on_back(self):
        """Handle going back to home (ask if recording is active)."""
        if self.recording:
            reply = QMessageBox.question(
                self,
                "Stop recording?",
                "Recording is active. Stop and go back?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            self.stop_record_session()
        self.go_home()

    def toggle_record(self):
        """Start or stop a recording session."""
        if not self.recording:
            self.start_record_session()
        else:
            self.stop_record_session()

    def start_record_session(self):
        """Start recording: reset timer and notify engine."""
        self.engine.start_record()
        self.recording = True
        self.elapsed_sec = 0
        self.timer_label.setText("Time: 00:00")
        self.btn_record.setText("Stop Record")
        self.time_timer.start(1000)

    def stop_record_session(self):
        """Stop recording: ask engine to save summary and show a message."""
        summary_info = self.engine.stop_record()
        self.recording = False
        self.btn_record.setText("Start Record")
        self.time_timer.stop()
        if summary_info is not None:
            duration_sec = summary_info["duration_sec"]
            path = summary_info["path"]
            mins = int(duration_sec // 60)
            secs = int(duration_sec % 60)
            QMessageBox.information(
                self,
                "Session Saved",
                f"Duration: {mins:02d}:{secs:02d}\nSaved to:\n{path}",
            )

    def update_time(self):
        """Update the recording timer label."""
        self.elapsed_sec += 1
        m = self.elapsed_sec // 60
        s = self.elapsed_sec % 60
        self.timer_label.setText(f"Time: {m:02d}:{s:02d}")

    def update_frame(self):
        """
        Pull a frame + stats from BehaviorEngine,
        draw it into the video_label, and update the right sidebar stats.
        """
        out = self.engine.read_frame()
        if out is None:
            return
        frame, stats, num_people = out

        # Convert BGR -> RGB for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

        # Update sidebar stats text
        if stats:
            lines = [f"Students in frame: {num_people}"]
            lines.append("")  # blank line
            lines.append("Stats (current):")
            # sort behaviors by fraction descending
            for k, v in sorted(stats.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"{k}: {v*100:.1f}%")
            self.right_stats.setText("\n".join(lines))


# =========================
# Reports Page
# =========================

class ReportsPage(QWidget):
    """
    Reports page:
      - Left: list of saved sessions (JSON)
      - Right: scrollable report with summary text + line graph + pie chart
      - Line graph: per-5-second snapshot over time
      - Pie chart: overall behavior distribution
      - Checkboxes above line graph allow filtering which behavior lines are visible.
    """
    def __init__(self, engine: BehaviorEngine, go_home, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.go_home = go_home

        self.graph_lines = {}   # class_name -> matplotlib Line2D
        self.checkboxes = {}    # class_name -> QCheckBox

        main_layout = QHBoxLayout(self)

        # ---------- LEFT: session list ----------
        left_layout = QVBoxLayout()
        title = QLabel("Saved Sessions")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.list_sessions = QListWidget()
        self.list_sessions.itemSelectionChanged.connect(self.on_session_selected)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_sessions)

        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self.go_home)

        left_layout.addWidget(title)
        left_layout.addWidget(self.list_sessions)
        left_layout.addWidget(btn_refresh)
        left_layout.addWidget(btn_back)

        # ---------- RIGHT: scrollable report area ----------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        right_container = QWidget()
        scroll_area.setWidget(right_container)

        right_layout = QVBoxLayout(right_container)

        # Session summary text
        self.summary_label = QLabel("Select a session to see details.")
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_label.setWordWrap(True)

        # Behavior filter checkboxes (row above line graph)
        self.checkbox_layout = QGridLayout()
        self.checkbox_layout.setSpacing(5)

        # Matplotlib figure (line chart + pie chart)
        self.fig = Figure(figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)

        # Assemble right layout
        right_layout.addWidget(self.summary_label)
        right_layout.addLayout(self.checkbox_layout)  # checkboxes row
        right_layout.addWidget(self.canvas)
        right_layout.addStretch(1)

        # Combine left + right
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(scroll_area, stretch=2)

        self.refresh_sessions()

    def refresh_sessions(self):
        """Reload the list of saved session JSON files."""
        self.list_sessions.clear()
        sessions = self.engine.list_sessions()
        for name, full in sessions:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, full)
            self.list_sessions.addItem(item)

    def on_session_selected(self):
        """Triggered when a session is clicked in the list."""
        items = self.list_sessions.selectedItems()
        if not items:
            return
        item = items[0]
        path = item.data(Qt.UserRole)
        self.load_session(path)

    def load_session(self, path):
        """Load a JSON report, update summary text, and redraw graphs."""
        path = Path(path)
        if not path.exists():
            QMessageBox.warning(self, "Error", f"Session file not found:\n{path}")
            return

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        duration_sec = data.get("duration_sec", 0.0)
        class_names = data.get("class_names", [])
        per_5sec = data.get("per_5sec", [])

        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        self.summary_label.setText(
            f"Session: {path.name}\n"
            f"Duration: {mins:02d}:{secs:02d}\n"
            f"5-second snapshots: {len(per_5sec)}"
        )

        # Clear old checkboxes
        for i in reversed(range(self.checkbox_layout.count())):
            item = self.checkbox_layout.itemAt(i)
            if item is not None and item.widget() is not None:
                item.widget().setParent(None)
        self.checkboxes.clear()
        self.graph_lines.clear()

        if not per_5sec or not class_names:
            self.fig.clear()
            self.canvas.draw()
            return

        # Build time axis (use stored start_sec if present)
        times_sec = [w.get("start_sec", i * 5) for i, w in enumerate(per_5sec)]

        # Build matrix of percentages [n_windows x n_classes]
        mat = []
        for w in per_5sec:
            dist = w.get("dist", {})
            row = [dist.get(cls, 0.0) * 100.0 for cls in class_names]
            mat.append(row)
        mat = np.array(mat)

        # Set up checkboxes for each behavior (default: all ON)
        for idx, cls in enumerate(class_names):
            cb = QCheckBox(cls)
            cb.setChecked(True)
            # connect each checkbox to a visibility update for its class
            cb.stateChanged.connect(lambda _, c=cls: self.update_graph_visibility(c))
            self.checkbox_layout.addWidget(cb, 0, idx)
            self.checkboxes[cls] = cb

        # Clear figure and create 2 subplots (top=timeline, bottom=pie chart)
        self.fig.clear()
        ax_line = self.fig.add_subplot(2, 1, 1)
        ax_pie = self.fig.add_subplot(2, 1, 2)

        # Colors (from engine, matches live view)
        cmap = getattr(self.engine, "class_colors", {})

        # ---------- 1) Line chart (per-5-second snapshot) ----------
        self.graph_lines = {}
        for ci, cls in enumerate(class_names):
            values = mat[:, ci]
            bgr = cmap.get(cls, (0, 255, 0))
            color_rgb = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

            line_obj, = ax_line.plot(
                times_sec, values, marker="o", label=cls, color=color_rgb
            )
            self.graph_lines[cls] = line_obj

        ax_line.set_title("Per-5-second Behavior Snapshot")
        ax_line.set_ylabel("Percentage (%)")
        ax_line.set_ylim(0, 100)
        ax_line.set_xticks(times_sec)
        ax_line.grid(True, alpha=0.3)
        ax_line.legend(loc="upper right", fontsize=8)

        # ---------- 2) Pie Chart (overall behavior distribution) ----------
        totals = mat.sum(axis=0)

        pie_colors = []
        for cls in class_names:
            bgr = cmap.get(cls, (0, 255, 0))
            pie_colors.append((bgr[2]/255, bgr[1]/255, bgr[0]/255))

        ax_pie.pie(
            totals,
            labels=class_names,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False
        )
        ax_pie.set_title("Overall Behavior Distribution (whole session)")

        self.fig.tight_layout()
        self.canvas.draw()

    def update_graph_visibility(self, class_name: str):
        """
        When a checkbox is toggled, show/hide the corresponding
        behavior line on the timeline graph.
        """
        if class_name not in self.graph_lines or class_name not in self.checkboxes:
            return

        line = self.graph_lines[class_name]
        checkbox = self.checkboxes[class_name]

        line.set_visible(checkbox.isChecked())
        self.canvas.draw_idle()


# =========================
# Main Window
# =========================

class MainWindow(QMainWindow):
    """
    Main application window:
      - Uses a QStackedWidget to switch between:
          0: HomePage
          1: MonitorPage
          2: ReportsPage
      - Holds a single BehaviorEngine instance shared by both live & reports.
    """
    def __init__(self):
        super().__init__()
        self.engine = BehaviorEngine(source=0)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage(self.show_monitor, self.show_reports)
        self.monitor_page = MonitorPage(self.engine, self.show_home)
        self.reports_page = ReportsPage(self.engine, self.show_home)

        self.stack.addWidget(self.home_page)    # index 0
        self.stack.addWidget(self.monitor_page) # index 1
        self.stack.addWidget(self.reports_page) # index 2

        self.show_home()
        self.setWindowTitle("Classroom Behavior Monitor")
        self.resize(1280, 720)

    def show_home(self):
        self.stack.setCurrentWidget(self.home_page)

    def show_monitor(self):
        self.stack.setCurrentWidget(self.monitor_page)

    def show_reports(self):
        self.stack.setCurrentWidget(self.reports_page)

    def closeEvent(self, event):
        """On close, release the engine's camera."""
        self.engine.release()
        super().closeEvent(event)


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
