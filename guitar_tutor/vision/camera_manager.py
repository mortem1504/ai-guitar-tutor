from __future__ import annotations

from threading import Condition, Event, Lock, Thread
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger("guitar_tutor.camera")


class CameraManager:
    """Persistent local macOS camera capture with a latest-frame-only buffer."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, device: int = 0):
        self.width, self.height, self.target_fps, self.device = width, height, fps, device
        self._capture = None
        self._thread: Thread | None = None
        self._stop = Event()
        self._condition = Condition()
        self._lock = Lock()
        self._latest: np.ndarray | None = None
        self._sequence = 0
        self._fps = 0.0
        self.error: str | None = None

    def start(self) -> None:
        if self.is_running():
            return
        self.error = None
        try:
            capture = cv2.VideoCapture(self.device, cv2.CAP_AVFOUNDATION)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.device)
            if not capture.isOpened():
                raise RuntimeError("Camera could not be opened. Enable camera access for Python/Terminal in macOS Privacy & Security.")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            # AVFoundation can report an opened handle before its first frame
            # is ready. Warm it up here, before the worker is advertised ready.
            warmup_deadline = time.monotonic() + 3.0
            ok, frame = False, None
            while time.monotonic() < warmup_deadline and not ok:
                ok, frame = capture.read()
                if not ok:
                    time.sleep(.05)
            if not ok or frame is None:
                capture.release()
                raise RuntimeError(
                    f"Camera {self.device} opened but returned no frames. "
                    "Grant camera access to the Python/Terminal app in macOS Privacy & Security, "
                    "close other camera apps, or select another camera device.")
            # Some macOS capture backends report success while returning an
            # all-zero buffer. Treat that as a failed device and try the next
            # index instead of showing a permanently black preview.
            if not np.any(frame):
                capture.release()
                raise RuntimeError(f"Camera {self.device} returned a black frame.")
            self._capture = capture
            with self._lock:
                self._latest = frame
                self._sequence = 1
            self._stop.clear()
            self._thread = Thread(target=self._run, name="guitar-camera", daemon=True)
            self._thread.start()
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Camera startup failed")
            self.stop()
            raise

    def _run(self) -> None:
        previous = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self._capture.read() if self._capture is not None else (False, None)
            if not ok or frame is None:
                self.error = "Camera read failed. Check macOS camera permission or that another app does not own the camera."
                time.sleep(.05)
                continue
            if not np.any(frame):
                self.error = f"Camera {self.device} returned a black frame. Select another camera device."
                time.sleep(.05)
                continue
            self.error = None
            now = time.perf_counter()
            interval = now - previous
            if interval > 0:
                self._fps = .9 * self._fps + .1 / interval if self._fps else 1 / interval
            previous = now
            with self._condition:
                with self._lock:
                    self._latest = frame
                    self._sequence += 1
                self._condition.notify_all()

    def get_latest_frame(self, after_sequence: int = -1, timeout: float = 0.0):
        with self._condition:
            if timeout:
                self._condition.wait_for(lambda: self._sequence > after_sequence or self._stop.is_set(), timeout)
            with self._lock:
                if self._latest is None or self._sequence <= after_sequence:
                    return None
                return self._sequence, self._latest.copy()

    @property
    def fps(self) -> float:
        return self._fps

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._lock:
            self._latest = None
            self._sequence = 0
        logger.info("Camera stopped")
