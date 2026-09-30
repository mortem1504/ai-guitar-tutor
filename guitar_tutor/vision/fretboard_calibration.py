from __future__ import annotations
import cv2
import numpy as np
from guitar_tutor.config import FretboardCalibration

CANONICAL=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32)


def homographies(calibration: FretboardCalibration) -> tuple[np.ndarray,np.ndarray]:
    calibration.validate(); image=np.asarray(calibration.corners,dtype=np.float32)
    h=cv2.getPerspectiveTransform(image,CANONICAL)
    if not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-9: raise ValueError("Calibration homography is singular")
    return h,np.linalg.inv(h)


def perspective_point(point, matrix: np.ndarray) -> tuple[float,float]:
    src=np.array([[[float(point[0]),float(point[1])]]],dtype=np.float32)
    out=cv2.perspectiveTransform(src,matrix)[0,0]
    return float(out[0]),float(out[1])
