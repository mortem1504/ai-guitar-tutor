from __future__ import annotations

import numpy as np

FINGERS = {"thumb": (1,2,3,4), "index": (5,6,7,8), "middle": (9,10,11,12),
           "ring": (13,14,15,16), "pinky": (17,18,19,20)}
CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))


def joint_angle(a, b, c) -> float:
    u, v = np.asarray(a)-b, np.asarray(c)-b
    denom = np.linalg.norm(u)*np.linalg.norm(v)
    if denom < 1e-8: return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denom, -1, 1))))


def palm_size(points: np.ndarray) -> float:
    return float((np.linalg.norm(points[0]-points[9]) + np.linalg.norm(points[5]-points[17]))/2)


def finger_flexion(points: np.ndarray, finger: str) -> float:
    cmc, mcp, pip, tip = FINGERS[finger]
    if finger == "thumb":
        angles = (joint_angle(points[cmc], points[mcp], points[pip]), joint_angle(points[mcp], points[pip], points[tip]))
    else:
        dip = FINGERS[finger][2]; tip = FINGERS[finger][3]
        angles = (joint_angle(points[0], points[mcp], points[pip]), joint_angle(points[mcp], points[pip], points[dip]),
                  joint_angle(points[pip], points[dip], points[tip]))
    # 0 extended, 1 strongly flexed. Guitar fingers may remain fairly straight.
    return float(np.clip((170-min(angles))/75, 0, 1))


def finger_angles(points: np.ndarray, finger: str) -> tuple[float, ...]:
    ids=FINGERS[finger]
    if finger=="thumb": return (joint_angle(points[ids[0]],points[ids[1]],points[ids[2]]),joint_angle(points[ids[1]],points[ids[2]],points[ids[3]]))
    return (joint_angle(points[0],points[ids[0]],points[ids[1]]),joint_angle(points[ids[0]],points[ids[1]],points[ids[2]]),joint_angle(points[ids[1]],points[ids[2]],points[ids[3]]))


class LandmarkSmoother:
    def __init__(self, alpha=.52): self.alpha, self.value = alpha, None
    def update(self, points):
        p = np.asarray(points, dtype=np.float32)
        self.value = p if self.value is None else self.alpha*p + (1-self.alpha)*self.value
        return self.value.copy()
    def reset(self): self.value = None
