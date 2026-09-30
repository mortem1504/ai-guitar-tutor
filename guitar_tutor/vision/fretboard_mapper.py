from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from guitar_tutor.config import FretboardCalibration
from guitar_tutor.vision.fretboard_calibration import homographies, perspective_point


@dataclass(frozen=True)
class CellAssignment:
    string: int; fret: int; distance: float; canonical: tuple[float,float]


class FretboardMapper:
    def __init__(self, calibration: FretboardCalibration, left_handed=False):
        self.calibration=calibration; self.left_handed=left_handed; self.h,self.inverse=homographies(calibration)
    def to_canonical(self,point): return perspective_point(point,self.h)
    def to_image(self,point): return perspective_point(point,self.inverse)
    def cell_center(self,string:int,fret:int):
        fi=4-fret if self.left_handed else fret-1; si=string-1 if self.left_handed else 6-string
        return ((fi+.5)/4,(si+.5)/6)
    def assign(self,point) -> CellAssignment | None:
        x,y=self.to_canonical(point)
        if not (0<=x<=1 and 0<=y<=1): return None
        fi=min(3,max(0,int(x*4))); si=min(5,max(0,int(y*6)))
        fret=4-fi if self.left_handed else fi+1; string=si+1 if self.left_handed else 6-si
        cx,cy=self.cell_center(string,fret)
        return CellAssignment(string,fret,float(np.hypot((x-cx)*4,(y-cy)*6)),(x,y))

    def point_to_fretboard(self, x: float, y: float) -> dict:
        """Return normalized fretboard coordinates for future string mapping."""
        nx, ny = self.to_canonical((x, y))
        inside = 0 <= nx <= 1 and 0 <= ny <= 1
        fret = int(nx * 4) + 1 if inside else None
        return {"normalized_x": nx, "normalized_y": ny, "fret": fret,
                "string": None, "inside_fretboard": inside}
