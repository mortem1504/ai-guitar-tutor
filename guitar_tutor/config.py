from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "models" / "hand_landmarker.task"
SETTINGS_PATH = ROOT / ".tutor_settings.json"


@dataclass(frozen=True)
class FretBox:
    x: float = 0.18
    y: float = 0.22
    width: float = 0.64
    height: float = 0.56

    def clamped(self) -> "FretBox":
        w, h = min(max(self.width, .25), .9), min(max(self.height, .25), .9)
        return FretBox(min(max(self.x, 0), 1-w), min(max(self.y, 0), 1-h), w, h)


@dataclass(frozen=True)
class FretboardCalibration:
    # nut-upper, fret4-upper, fret4-lower, nut-lower in normalized image space.
    corners: tuple[tuple[float,float], ...] = ((.18,.27),(.82,.31),(.82,.70),(.18,.74))
    calibrated: bool = False

    def validate(self) -> "FretboardCalibration":
        import cv2, numpy as np
        pts=np.asarray(self.corners,dtype=np.float32)
        if pts.shape!=(4,2) or not np.isfinite(pts).all() or np.any(pts<0) or np.any(pts>1):
            raise ValueError("Calibration corners must be four normalized image points")
        if abs(cv2.contourArea(pts)) < .04: raise ValueError("Calibration region is too small or self-intersecting")
        return self


@dataclass
class TutorConfig:
    sample_rate: int = 48_000
    audio_block_size: int = 1024
    fft_size: int = 8192
    min_frequency: float = 70.0
    max_frequency: float = 1400.0
    silence_dbfs: float = -52.0
    audio_history: int = 7
    landmark_ema_alpha: float = 0.52
    min_hand_detection_confidence: float = 0.55
    min_hand_presence_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    max_hands: int = 2
    fusion_freshness_seconds: float = 1.0
    success_cooldown_seconds: float = 1.2
    vision_stable_frames: int = 4
    mirror: bool = True
    guitar_handedness: str = "Right-handed"
    fret_box: FretBox = FretBox()
    calibration: FretboardCalibration = FretboardCalibration()
    use_auto_tuning: bool = True
    onset_cooldown_seconds: float = .65
    guitar_model: str = "guitar-yolo11n.pt"
    guitar_detection_confidence: float = .20
    guitar_detection_interval: int = 5
    guitar_stable_frames: int = 4
    guitar_lost_grace_seconds: float = 1.25
    fretboard_ema_alpha: float = .22
    fretboard_min_confidence: float = .42
    fretboard_stable_frames: int = 5
    debug_vision: bool = False
    debug_performance: bool = False
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 24


def load_calibration() -> FretBox:
    try:
        data = json.loads(SETTINGS_PATH.read_text())
        return FretBox(**data["fret_box"]).clamped()
    except (OSError, ValueError, TypeError, KeyError):
        return FretBox()


def load_fretboard_calibration() -> FretboardCalibration:
    try:
        data=json.loads(SETTINGS_PATH.read_text()); raw=data["calibration"]
        return FretboardCalibration(tuple(tuple(float(v) for v in p) for p in raw["corners"]),bool(raw["calibrated"])).validate()
    except (OSError,ValueError,TypeError,KeyError): return FretboardCalibration()


def save_calibration(box: FretBox) -> None:
    SETTINGS_PATH.write_text(json.dumps({"fret_box": asdict(box.clamped())}, indent=2))


def save_fretboard_calibration(value: FretboardCalibration) -> None:
    value.validate(); SETTINGS_PATH.write_text(json.dumps({"calibration":asdict(value)},indent=2))
