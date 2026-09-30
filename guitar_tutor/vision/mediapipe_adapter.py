from __future__ import annotations

from dataclasses import dataclass
import platform, sys
import logging
from pathlib import Path
import numpy as np


class MediaPipeUnavailable(RuntimeError): pass
logger=logging.getLogger("guitar_tutor.vision.mediapipe")


def capability_report() -> dict[str,str|bool]:
    try:
        import mediapipe as mp
        return {"version": getattr(mp,"__version__","unknown"), "tasks": bool(getattr(mp,"tasks",None)),
                "legacy": bool(getattr(getattr(mp,"solutions",None),"hands",None))}
    except Exception as exc: return {"version": f"not importable: {exc}", "tasks": False, "legacy": False}


class HandTracker:
    """MediaPipe Tasks first, legacy Solutions fallback, with a single output shape."""
    def __init__(self, model_path: Path, detection=.55, presence=.50, tracking=.50, max_hands=2):
        self.backend=""; self._task=None; self._legacy=None
        errors=[]
        try:
            import mediapipe as mp
            BaseOptions=mp.tasks.BaseOptions; vision=mp.tasks.vision
            if not model_path.exists(): raise FileNotFoundError(f"Model missing: {model_path}; run python scripts/download_models.py")
            # Explicit CPU delegate avoids Metal graph-service crashes seen in some
            # macOS/headless launch contexts while remaining fast enough for one hand.
            base=BaseOptions(model_asset_path=str(model_path),delegate=BaseOptions.Delegate.CPU)
            options=vision.HandLandmarkerOptions(base_options=base,
                running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
                min_hand_detection_confidence=detection, min_hand_presence_confidence=presence,
                min_tracking_confidence=tracking)
            self._mp=mp; self._task=vision.HandLandmarker.create_from_options(options); self.backend="tasks"
            return
        except Exception as exc:
            errors.append(f"Tasks API: {type(exc).__name__}: {exc}"); logger.warning("Tasks initialization failed: %s",exc)
        try:
            import mediapipe as mp
            hands=getattr(getattr(mp,"solutions",None),"hands",None)
            if hands is None: raise AttributeError("mediapipe.solutions.hands is unavailable")
            self._legacy=hands.Hands(static_image_mode=False,max_num_hands=max_hands,
                min_detection_confidence=detection,min_tracking_confidence=tracking); self.backend="legacy"; return
        except Exception as exc:
            errors.append(f"Legacy API: {type(exc).__name__}: {exc}"); logger.warning("Legacy initialization failed: %s",exc)
        caps=capability_report()
        raise MediaPipeUnavailable("MediaPipe hand tracking could not initialize.\n"
            f"MediaPipe={caps['version']}; Python={sys.version.split()[0]}; machine={platform.machine()}; "
            f"macOS={platform.mac_ver()[0]}; Tasks={caps['tasks']}; legacy={caps['legacy']}.\n"
            + "\n".join(errors) + "\nRecovery: use the pinned requirements, then run `python scripts/download_models.py`.")

    def detect(self, rgb: np.ndarray, timestamp_ms: int):
        if self.backend=="tasks":
            image=self._mp.Image(image_format=self._mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
            result=self._task.detect_for_video(image,timestamp_ms)
            if not result.hand_landmarks: return []
            hands=[]
            for i,landmarks in enumerate(result.hand_landmarks):
                pts=np.array([[p.x,p.y,p.z] for p in landmarks],dtype=np.float32)
                cat=result.handedness[i][0] if i<len(result.handedness) and result.handedness[i] else None
                hands.append((pts,getattr(cat,"category_name","Unknown"),float(getattr(cat,"score",.7))))
            return hands
        result=self._legacy.process(rgb)
        if not result.multi_hand_landmarks: return []
        hands=[]
        for i,landmarks in enumerate(result.multi_hand_landmarks):
            pts=np.array([[p.x,p.y,p.z] for p in landmarks.landmark],dtype=np.float32)
            cls=result.multi_handedness[i].classification[0] if result.multi_handedness and i<len(result.multi_handedness) else None
            hands.append((pts,getattr(cls,"label","Unknown"),float(getattr(cls,"score",.7))))
        return hands

    def close(self):
        obj=self._task or self._legacy
        if obj and hasattr(obj,"close"): obj.close()
