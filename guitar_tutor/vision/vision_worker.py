from __future__ import annotations

from threading import Event, Thread
import logging
import time

from guitar_tutor.config import TutorConfig
from guitar_tutor.state import SharedState
from guitar_tutor.vision.camera_manager import CameraManager
from guitar_tutor.vision.video_processor import GuitarVideoProcessor

logger = logging.getLogger("guitar_tutor.vision.worker")


class VisionWorker:
    """Processes only the newest camera frame and publishes processed output."""

    def __init__(self, state: SharedState, config: TutorConfig, camera: CameraManager):
        self.state, self.config, self.camera = state, config, camera
        self.processor = GuitarVideoProcessor(state, config)
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_sequence = -1

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        try:
            self.camera.start()
        except RuntimeError as first_error:
            # Device indexes can vary across macOS capture backends. If the
            # selected camera opens but produces no usable image, try the
            # other common local indexes before failing startup.
            original_device = self.camera.device
            for device in (0, 1):
                if device == original_device:
                    continue
                self.camera.device = device
                try:
                    self.camera.start()
                    self.camera.error = f"Camera {original_device} unavailable; using camera {device}."
                    logger.info("Camera fallback selected device %s", device)
                    break
                except RuntimeError:
                    continue
            else:
                self.camera.device = original_device
                raise first_error
        self._thread = Thread(target=self._run, name="guitar-vision", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self.camera.get_latest_frame(self._last_sequence, timeout=.25)
            if item is None:
                continue
            sequence, frame = item
            self._last_sequence = sequence
            try:
                processed = self.processor.process(frame)
                self.state.update_frame(processed)
                self.state.update_performance(camera_fps=self.camera.fps,
                    vision_fps=self.processor.fps, **self.processor.performance)
            except Exception as exc:
                logger.exception("Vision worker frame failed")
                self.state.update_vision_error(str(exc))

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        self.camera.stop()
        self.state.update_frame(None)

    def close(self) -> None:
        self.stop()
        self.processor.close()
