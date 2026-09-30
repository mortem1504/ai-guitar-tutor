from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from threading import RLock
import time
import numpy as np


class EvaluationState(str, Enum):
    NO_HAND="NO_HAND"; HAND_OUTSIDE_FRET_BOX="HAND_OUTSIDE_FRET_BOX"
    HAND_OUTSIDE_FRETBOARD="HAND_OUTSIDE_FRETBOARD"; CALIBRATION_REQUIRED="CALIBRATION_REQUIRED"
    WAITING_FOR_SHAPE="WAITING_FOR_SHAPE"; SHAPE_CORRECT_WAITING_FOR_STRUM="SHAPE_CORRECT_WAITING_FOR_STRUM"
    WAITING_FOR_STRUM="WAITING_FOR_STRUM"; VISION_MISMATCH="VISION_MISMATCH"
    AUDIO_MISMATCH="AUDIO_MISMATCH"; VISION_ONLY_CORRECT="VISION_ONLY_CORRECT"
    AUDIO_ONLY_CORRECT="AUDIO_ONLY_CORRECT"; PERFECT="PERFECT"
    LOW_SIGNAL="LOW_SIGNAL"; SYSTEM_ERROR="SYSTEM_ERROR"
    OUT_OF_TUNE="OUT_OF_TUNE"


@dataclass(frozen=True)
class GuitarDetectionState:
    guitar_detected: bool = False
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    angle: float = 0.0
    body_region: tuple[float, float, float, float] | None = None
    neck_region: tuple[float, float, float, float] | None = None
    last_seen_timestamp: float = 0.0
    stable_frames: int = 0
    lost_frames: int = 0
    detector_available: bool = True
    error: str | None = None


@dataclass(frozen=True)
class FretboardState:
    status: str = "NO_GUITAR"
    confidence: float = 0.0
    polygon: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] | None = None
    angle: float = 0.0
    length: float = 0.0
    width: float = 0.0
    nut: tuple[float, float] | None = None
    frets: tuple[float, ...] = ()
    stable_frames: int = 0
    failures: int = 0
    message: str = "Hold your guitar inside the camera view."

    @property
    def calibration_confidence(self) -> float:
        return self.confidence


@dataclass
class VisionResult:
    timestamp: float = 0.0
    hand_detected: bool = False
    inside_fret_box: bool = False
    confidence: float = 0.0
    match: bool = False
    detected_shape: str = "—"
    fretting_fingers: tuple[str, ...] = ()
    assignments: dict[str, tuple[int,int]] = field(default_factory=dict)
    average_cell_error: float = 1.0
    tracking_confidence: float = 0.0
    stable_frames: int = 0
    handedness: str = "Unknown"
    fps: float = 0.0
    latency_ms: float = 0.0
    landmarks: tuple[tuple[float,float,float], ...] = ()
    joint_angles: dict[str, tuple[float, ...]] = field(default_factory=dict)
    confidence_components: dict[str, float] = field(default_factory=dict)
    finger_diagnostics: dict[str, dict] = field(default_factory=dict)
    canonical_points: dict[str, tuple[float,float]] = field(default_factory=dict)
    stable_since: float = 0.0
    hand_count: int = 0
    all_handedness: tuple[str, ...] = ()
    error: str | None = None
    guitar: GuitarDetectionState = field(default_factory=GuitarDetectionState)
    fretboard: FretboardState = field(default_factory=FretboardState)


@dataclass
class AudioResult:
    timestamp: float = 0.0
    raw_chord: str = "—"
    stable_chord: str = "—"
    confidence: float = 0.0
    match: bool = False
    target_similarity: float = 0.0
    best_similarity: float = 0.0
    margin: float = 0.0
    rms: float = 0.0
    dbfs: float = -120.0
    peaks: tuple[tuple[float,float], ...] = ()
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(1024, dtype=np.float32))
    expected_tone_coverage: float = 0.0
    latency_ms: float = 0.0
    similarities: dict[str, float] = field(default_factory=dict)
    sample_rate: int = 0
    queue_fill: int = 0
    spectral_flux: float = 0.0
    onset_state: str = "quiet"
    strum_timestamp: float = 0.0
    tuning_offset_cents: float = 0.0
    pitch_class_strengths: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Evaluation:
    state: EvaluationState = EvaluationState.NO_HAND
    message: str = "Start the tutor and place your hand in the guide."
    vision_confidence: float = 0.0
    audio_confidence: float = 0.0
    combined_confidence: float = 0.0
    selected_target: str = "E Major"
    recognized_audio_chord: str = "—"
    detected_hand_shape: str = "—"
    timestamp: float = 0.0
    success_event: int = 0
    attempt_event: int = 0


@dataclass
class Snapshot:
    vision: VisionResult
    audio: AudioResult
    evaluation: Evaluation
    running: bool
    audio_active: bool
    latest_frame: np.ndarray | None = None
    performance: dict[str, float | str] = field(default_factory=dict)


class SharedState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._vision, self._audio, self._evaluation = VisionResult(), AudioResult(), Evaluation()
        self.running = False
        self.audio_active = False
        self.target = "E Major"
        self._latest_frame: np.ndarray | None = None
        self._performance: dict[str, float | str] = {}

    def update_vision(self, value: VisionResult) -> None:
        with self._lock: self._vision = value

    def update_audio(self, value: AudioResult) -> None:
        with self._lock: self._audio = value

    def update_evaluation(self, value: Evaluation) -> None:
        with self._lock: self._evaluation = value

    def set_target(self, target: str) -> None:
            with self._lock: self.target = target

    def update_frame(self, frame: np.ndarray | None) -> None:
        with self._lock:
            self._latest_frame = None if frame is None else frame.copy()

    def update_performance(self, **values) -> None:
        with self._lock:
            self._performance.update({key: float(value) if isinstance(value, (int, float)) else value
                for key, value in values.items()})

    def update_vision_error(self, message: str) -> None:
        with self._lock:
            self._vision.error = message

    def set_running(self, running: bool) -> None:
        with self._lock:
            self.running = running

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(replace(self._vision, assignments=dict(self._vision.assignments),
                guitar=replace(self._vision.guitar), fretboard=replace(self._vision.fretboard)),
                replace(self._audio, chroma=self._audio.chroma.copy(), waveform=self._audio.waveform.copy(),
                    similarities=dict(self._audio.similarities)),
                replace(self._evaluation), self.running, self.audio_active,
                None if self._latest_frame is None else self._latest_frame.copy(), dict(self._performance))

    def stop(self) -> None:
        with self._lock:
            self.running = self.audio_active = False
            self._latest_frame = None
