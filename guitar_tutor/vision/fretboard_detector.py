from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from guitar_tutor.state import FretboardState, GuitarDetectionState
from guitar_tutor.vision.fretboard_geometry import FRET_RATIOS, FretboardGeometry, geometry_from_polygon


@dataclass(frozen=True)
class FretboardObservation:
    geometry: FretboardGeometry
    edge_score: float
    fret_score: float


def _line_features(line: np.ndarray):
    x1, y1, x2, y2 = (float(v) for v in line)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    angle = math.atan2(dy, dx)
    normal = np.array([-dy, dx], dtype=np.float32) / length
    direction = np.array([dx, dy], dtype=np.float32) / length
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    projection = points @ direction
    return length, angle, normal, direction, projection


class FretboardDetector:
    """Detects two long, parallel neck edges inside a guitar ROI."""

    def __init__(self, min_confidence: float = .42):
        self.min_confidence = min_confidence

    @staticmethod
    def _roi(guitar: GuitarDetectionState, width: int, height: int):
        if not guitar.bbox:
            return None
        x1, y1, x2, y2 = guitar.bbox
        pad_x, pad_y = (x2 - x1) * .06, (y2 - y1) * .06
        return (max(0, int((x1 - pad_x) * width)), max(0, int((y1 - pad_y) * height)),
                min(width, int((x2 + pad_x) * width)), min(height, int((y2 + pad_y) * height)))

    @staticmethod
    def _fret_positions(normalized: np.ndarray) -> tuple[tuple[float, ...], float]:
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(clahe, 45, 130)
        height, width = edges.shape
        projection = edges[int(height * .12):int(height * .88)].mean(axis=0) / 255.0
        # Group neighboring peaks so one wire cannot count multiple times.
        candidates = [index for index in range(2, width - 2)
                      if projection[index] >= projection[index - 1]
                      and projection[index] >= projection[index + 1]
                      and projection[index] > .035]
        grouped: list[int] = []
        for index in candidates:
            if not grouped or index - grouped[-1] > max(3, width // 90):
                grouped.append(index)
            elif projection[index] > projection[grouped[-1]]:
                grouped[-1] = index
        if len(grouped) < 2:
            return (), 0.0
        values = np.asarray(grouped, dtype=np.float32) / max(1, width - 1)
        selected = []
        for expected in FRET_RATIOS[1:]:
            nearest = values[np.argmin(np.abs(values - expected))]
            if abs(float(nearest) - expected) <= .13:
                selected.append(float(nearest))
        return tuple((0., *selected)), min(1., len(selected) / 4)

    def detect(self, image: np.ndarray, guitar: GuitarDetectionState) -> FretboardObservation | None:
        roi = self._roi(guitar, image.shape[1], image.shape[0])
        if roi is None or not guitar.guitar_detected:
            return None
        left, top, right, bottom = roi
        crop = image[top:bottom, left:right]
        if crop.size == 0 or min(crop.shape[:2]) < 48:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 140)
        min_length = max(crop.shape[:2]) * .22
        raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(18, int(min_length * .11)),
            minLineLength=int(min_length), maxLineGap=max(8, int(min_length * .08)))
        if raw is None:
            return None
        lines = []
        for item in np.asarray(raw).reshape(-1, 4):
            features = _line_features(item)
            if features:
                length, angle, normal, direction, projection = features
                points = np.asarray(((item[0], item[1]), (item[2], item[3])), dtype=np.float32)
                lines.append((points, length, angle, normal, direction, projection))
        if len(lines) < 2:
            return None
        best = None
        for i, first in enumerate(lines):
            for second in lines[i + 1:]:
                angle_delta = abs((first[2] - second[2] + np.pi / 2) % np.pi - np.pi / 2)
                if angle_delta > math.radians(10):
                    continue
                direction = first[4]
                projections = np.concatenate((first[0] @ direction, second[0] @ direction))
                start, end = float(projections.min()), float(projections.max())
                overlap = min(first[5][1], second[5][1]) - max(first[5][0], second[5][0])
                span = max(1., end - start)
                if overlap < span * .34:
                    continue
                separation = abs(float(np.mean(second[0] @ first[3]) - np.mean(first[0] @ first[3])))
                if separation < max(8., crop.shape[1] * .018) or separation > max(crop.shape) * .45:
                    continue
                score = (min(first[1], second[1]) / max(crop.shape)) * .55 + (overlap / span) * .3
                score += max(0., 1. - angle_delta / math.radians(10)) * .15
                if best is None or score > best[0]:
                    best = (score, first, second, direction, start, end)
        if best is None:
            return None
        score, first, second, direction, start, end = best
        normal = first[3]
        if np.dot(second[0].mean(axis=0) - first[0].mean(axis=0), normal) < 0:
            normal = -normal
        def edge_at(line, projection):
            points = line[0]
            values = points @ direction
            fraction = 0. if abs(values[1] - values[0]) < 1e-6 else (projection - values[0]) / (values[1] - values[0])
            return points[0] + fraction * (points[1] - points[0])
        a0, a1 = edge_at(first, start), edge_at(first, end)
        b0, b1 = edge_at(second, start), edge_at(second, end)
        width_start, width_end = np.linalg.norm(b0 - a0), np.linalg.norm(b1 - a1)
        if width_start <= width_end:
            polygon = np.array([a0, a1, b1, b0], dtype=np.float32)
        else:
            polygon = np.array([a1, a0, b0, b1], dtype=np.float32)
        polygon[:, 0] += left
        polygon[:, 1] += top
        polygon[:, 0] /= image.shape[1]
        polygon[:, 1] /= image.shape[0]
        if abs(cv2.contourArea(polygon)) < .002:
            return None
        # Warp the candidate so fret wires are measured in a stable frame.
        local = polygon.copy()
        local[:, 0] *= image.shape[1]
        local[:, 1] *= image.shape[0]
        matrix = cv2.getPerspectiveTransform(local.astype(np.float32),
            np.array([[0, 0], [320, 0], [320, 120], [0, 120]], dtype=np.float32))
        normalized = cv2.warpPerspective(image, matrix, (320, 120))
        frets, fret_score = self._fret_positions(normalized)
        geometry = geometry_from_polygon(polygon, min(1., score), frets)
        confidence = .72 * min(1., score) + .28 * fret_score
        if confidence < self.min_confidence:
            return None
        return FretboardObservation(FretboardGeometry(geometry.polygon, geometry.angle,
            geometry.length, geometry.width, confidence, geometry.fret_positions),
            min(1., score), fret_score)


class FretboardTracker:
    def __init__(self, alpha: float = .22, stable_frames: int = 5):
        self.alpha = alpha
        self.stable_required = max(1, stable_frames)
        self.geometry: FretboardGeometry | None = None
        self.state = FretboardState()

    def update(self, observation: FretboardObservation | None, guitar: GuitarDetectionState,
               now: float | None = None) -> tuple[FretboardState, FretboardGeometry | None]:
        now = time.monotonic() if now is None else now
        if not guitar.guitar_detected:
            self.geometry = None
            self.state = FretboardState("NO_GUITAR", 0., (), None, 0., 0., 0., None, (), 0,
                self.state.failures, "Hold your guitar inside the camera view.")
            return self.state, None
        if observation is None:
            failures = self.state.failures + 1
            status = "RECALIBRATING" if self.geometry else "SEARCHING_FOR_FRETBOARD"
            message = "Show the guitar neck and first four frets."
            if self.geometry and failures <= 12:
                self.state = FretboardState(status, self.state.confidence, self.state.polygon,
                    self.state.center, self.state.angle, self.state.length, self.state.width,
                    self.state.nut, self.state.frets, self.state.stable_frames, failures, message)
                return self.state, self.geometry
            self.geometry = None
            self.state = FretboardState(status, 0., (), None, 0., 0., 0., None, (), 0, failures, message)
            return self.state, None
        current = observation.geometry
        if self.geometry is None:
            polygon = current.points
            stable = 1
        else:
            polygon = (1 - self.alpha) * self.geometry.points + self.alpha * current.points
            stable = self.state.stable_frames + 1
        polygon_tuple = tuple(tuple(float(v) for v in point) for point in polygon)
        confidence = current.confidence if self.geometry is None else (1 - self.alpha) * self.geometry.confidence + self.alpha * current.confidence
        self.geometry = geometry_from_polygon(polygon_tuple, confidence, current.fret_positions)
        status = "CALIBRATING" if stable < self.stable_required else ("CALIBRATED" if stable == self.stable_required else "TRACKING")
        message = "Hold the guitar steady." if stable < self.stable_required else "Fretboard ready."
        self.state = FretboardState(status, confidence, self.geometry.polygon, self.geometry.center,
            self.geometry.angle, self.geometry.length, self.geometry.width, self.geometry.nut,
            self.geometry.fret_positions, stable, 0, message)
        return self.state, self.geometry
