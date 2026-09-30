from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


CANONICAL = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float32)
FRET_RATIOS = (0.0, 0.272, 0.529, 0.771, 1.0)


@dataclass(frozen=True)
class FretboardGeometry:
    """A camera-space fretboard quadrilateral and perspective conversion API."""

    polygon: tuple[tuple[float, float], ...]
    angle: float
    length: float
    width: float
    confidence: float
    fret_positions: tuple[float, ...] = ()

    def __post_init__(self):
        if len(self.polygon) != 4:
            raise ValueError("Fretboard geometry requires four corners")

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.polygon, dtype=np.float32)

    @property
    def center(self) -> tuple[float, float]:
        point = self.points.mean(axis=0)
        return float(point[0]), float(point[1])

    @property
    def nut(self) -> tuple[float, float]:
        point = self.points
        return float((point[0, 0] + point[3, 0]) / 2), float((point[0, 1] + point[3, 1]) / 2)

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        forward = cv2.getPerspectiveTransform(self.points, CANONICAL)
        inverse = cv2.getPerspectiveTransform(CANONICAL, self.points)
        return forward, inverse

    def camera_to_fretboard(self, point) -> tuple[float, float]:
        forward, _ = self.matrices()
        source = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(source, forward)[0, 0]
        return float(result[0]), float(result[1])

    def fretboard_to_camera(self, point) -> tuple[float, float]:
        _, inverse = self.matrices()
        source = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(source, inverse)[0, 0]
        return float(result[0]), float(result[1])

    def point_to_fretboard(self, x: float, y: float) -> dict:
        nx, ny = self.camera_to_fretboard((x, y))
        inside = 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        fret = None
        if inside:
            positions = self.fret_positions or FRET_RATIOS
            fret = next((index for index in range(1, len(positions))
                         if positions[index - 1] <= nx <= positions[index]), None)
        return {"normalized_x": nx, "normalized_y": ny, "fret": fret,
                "string": None, "inside_fretboard": inside}


def geometry_from_polygon(polygon, confidence: float, frets=()) -> FretboardGeometry:
    points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    top = float(np.linalg.norm(points[1] - points[0]))
    bottom = float(np.linalg.norm(points[2] - points[3]))
    left = float(np.linalg.norm(points[3] - points[0]))
    right = float(np.linalg.norm(points[2] - points[1]))
    axis = (points[1] + points[2] - points[0] - points[3]) / 2
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return FretboardGeometry(tuple(tuple(float(v) for v in p) for p in points), angle,
        (top + bottom) / 2, (left + right) / 2, float(np.clip(confidence, 0., 1.)), tuple(frets))
