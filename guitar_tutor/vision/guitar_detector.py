from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

import cv2
from pathlib import Path

from guitar_tutor.state import GuitarDetectionState

logger = logging.getLogger("guitar_tutor.vision.guitar_detector")


@dataclass(frozen=True)
class GuitarDetection:
    confidence: float
    bbox: tuple[float, float, float, float]
    angle: float = 0.0


class GuitarDetector:
    """Local guitar-trained YOLO detector with process-wide model caching."""

    _models: dict[str, object] = {}
    _model_lock = threading.Lock()

    def __init__(self, model_name: str = "guitar-yolo11n.pt", confidence: float = .35,
                 interval: int = 3, input_size: int = 640):
        self.model_name = model_name
        self.confidence = confidence
        self.interval = max(1, interval)
        self.input_size = input_size
        self.device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
        except Exception:
            pass
        self._frame_index = 0
        self._last: GuitarDetection | None = None
        self._last_error: str | None = None
        self._future: Future | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guitar-yolo")
        self.available = True
        self.error: str | None = None
        self._model = None
        self._class_ids: list[int] = []
        self._model_future = self._executor.submit(self._load_model)

    def _load_model(self):
        try:
            import os
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            from ultralytics import YOLO
            model_path = Path(self.model_name)
            if not model_path.is_absolute():
                root = Path(__file__).resolve().parents[2]
            candidates = (root / "models" / model_path.name, root / "assets" / "models" / model_path.name)
            model_path = next((candidate for candidate in candidates if candidate.exists()), None)
            world_fallback = model_path is None and (root / "yolov8s-worldv2.pt").exists()
            if model_path is None and not world_fallback:
                self.available = False
                self.error = f"Local guitar model missing: {self.model_name}"
                logger.warning(self.error)
                return None
            if world_fallback:
                model_path = root / "yolov8s-worldv2.pt"
        except Exception as exc:
            self.available = False
            self.error = f"Ultralytics unavailable: {exc}"
            logger.warning(self.error)
            return None
        try:
            with self._model_lock:
                if self.model_name not in self._models:
                    model = YOLO(str(model_path))
                    if "world" in str(model_path).lower():
                        model.set_classes(["guitar", "acoustic guitar", "electric guitar"])
                        self._class_ids = list(range(len(model.names)))
                    else:
                        self._class_ids = [index for index, name in model.names.items() if str(name).lower() == "guitar"]
                    if not self._class_ids:
                        raise ValueError("Model has no guitar class and is not an open-vocabulary YOLO-World model")
                    self._models[self.model_name] = model
                return self._models[self.model_name]
        except Exception as exc:
            self.available = False
            self.error = f"Guitar model unavailable: {exc}"
            logger.warning(self.error)
            return None

    def detect(self, image: np.ndarray) -> GuitarDetection | None:
        """Run YOLO periodically and retain the last detection between runs."""
        self._frame_index += 1
        if self._model is None and self._model_future.done():
            try:
                self._model = self._model_future.result()
            except Exception as exc:
                self.available = False
                self.error = f"Guitar model worker failed: {exc}"
                logger.warning(self.error)
        if self._model is None:
            return None
        if self._future is not None and self._future.done():
            try:
                detection = self._future.result()
                if detection is not None:
                    self._last = detection
                    self._last_error = None
                elif self._last_error is None:
                    self._last = None
            except Exception as exc:
                logger.warning("Guitar detector worker failed: %s", exc)
            self._future = None
        if self._frame_index % self.interval == 1 and self._future is None:
            self._future = self._executor.submit(self._infer, cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA), image.shape[1], image.shape[0])
        return self._last

    def _infer(self, image: np.ndarray, source_width: int, source_height: int) -> GuitarDetection | None:
        try:
            guitar_ids = self._class_ids or [index for index, name in self._model.names.items() if str(name).lower() == "guitar"]
            results = self._model.predict(source=image, imgsz=self.input_size,
                conf=self.confidence, classes=guitar_ids,
                verbose=False, device=self.device)
            best: GuitarDetection | None = None
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    score = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls not in guitar_ids:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    candidate = GuitarDetection(score,
                        (max(0., x1 / self.input_size), max(0., y1 / self.input_size),
                         min(1., x2 / self.input_size), min(1., y2 / self.input_size)))
                    if best is None or candidate.confidence > best.confidence:
                        best = candidate
            return best
        except Exception as exc:
            logger.warning("Guitar detector frame failed: %s", exc)
            self._last_error = str(exc)
            return self._last

    @staticmethod
    def _estimate_angle(image: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = (max(0, int(value)) for value in bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        # Orientation is inferred by the fretboard geometry stage; running
        # Canny + Hough over the whole guitar crop on each YOLO result adds
        # latency without improving presence detection.
        return 0.0

    def close(self):
        if not getattr(self._executor, "_shutdown", False):
            self._executor.shutdown(wait=False, cancel_futures=True)


class GuitarTracker:
    """EMA tracker with appearance hysteresis and a short disappearance grace."""

    def __init__(self, stable_frames: int = 4, lost_grace_seconds: float = 1.25,
                 alpha: float = .28):
        self.stable_frames_required = max(1, stable_frames)
        self.lost_grace_seconds = lost_grace_seconds
        self.alpha = alpha
        self.state = GuitarDetectionState()

    @staticmethod
    def _ema(previous, current, alpha):
        if previous is None:
            return current
        return tuple((1 - alpha) * a + alpha * b for a, b in zip(previous, current))

    @staticmethod
    def _regions(bbox: tuple[float, float, float, float]):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        if height >= width:
            neck = (x1 + .28 * width, y1, x1 + .72 * width, y1 + .62 * height)
            body = (x1, y1 + .35 * height, x2, y2)
        else:
            neck = (x1, y1 + .28 * height, x1 + .62 * width, y1 + .72 * height)
            body = (x1 + .35 * width, y1, x2, y2)
        return body, neck

    def update(self, detection: GuitarDetection | None, now: float | None = None,
               available: bool = True, error: str | None = None) -> GuitarDetectionState:
        now = time.monotonic() if now is None else now
        old = self.state
        if detection is not None:
            bbox = self._ema(old.bbox, detection.bbox, self.alpha)
            confidence = ((1 - self.alpha) * old.confidence + self.alpha * detection.confidence
                if old.bbox else detection.confidence)
            stable = old.stable_frames + 1
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            body, neck = self._regions(bbox)
            self.state = GuitarDetectionState(stable >= self.stable_frames_required,
                float(np.clip(confidence, 0., 1.)), bbox, center, detection.angle,
                body, neck, now, stable, 0, available, error)
            return self.state
        if old.bbox is not None and old.last_seen_timestamp and now - old.last_seen_timestamp <= self.lost_grace_seconds:
            self.state = GuitarDetectionState(old.guitar_detected, old.confidence * .96,
                old.bbox, old.center, old.angle, old.body_region, old.neck_region,
                old.last_seen_timestamp, old.stable_frames, old.lost_frames + 1,
                available, error)
            return self.state
        self.state = GuitarDetectionState(False, 0., None, None, 0., None, None,
            old.last_seen_timestamp, 0, old.lost_frames + 1, available, error)
        return self.state
