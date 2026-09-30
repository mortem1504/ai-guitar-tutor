from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from guitar_tutor.chord_library import Chord, FingerPosition
from guitar_tutor.config import FretBox
from guitar_tutor.state import VisionResult
from guitar_tutor.vision.hand_geometry import FINGERS, finger_angles, finger_flexion
from guitar_tutor.vision.fretboard_mapper import FretboardMapper
import time


def cell_center(box: FretBox, string: int, fret: int, left_handed: bool=False) -> tuple[float,float]:
    # Camera guide: frets left→right; low E top→high E bottom.
    fret_index = 4-fret if left_handed else fret-1
    string_index = string-1 if left_handed else 6-string
    return box.x + (fret_index+.5)*box.width/4, box.y + (string_index+.5)*box.height/6


def point_to_cell(point, box: FretBox, left_handed: bool=False) -> tuple[int,int] | None:
    x,y = float(point[0]), float(point[1])
    if not (box.x <= x <= box.x+box.width and box.y <= y <= box.y+box.height): return None
    fret_index=min(3,max(0,int((x-box.x)/box.width*4)))
    string_index=min(5,max(0,int((y-box.y)/box.height*6)))
    fret = 4-fret_index if left_handed else fret_index+1
    string = string_index+1 if left_handed else 6-string_index
    return string, fret


class ShapeClassifier:
    def __init__(self, stable_frames=4):
        self.stable_frames = stable_frames; self.above = self.below = 0; self.matched = False; self.stable_since=0.

    def reset(self): self.above=self.below=0; self.matched=False; self.stable_since=0.

    def classify(self, points: np.ndarray, chord: Chord, box: FretBox | None=None, tracking=.8, handedness="Unknown", left_handed=False, mapper:FretboardMapper|None=None) -> VisionResult:
        fingertips = {name: points[idx[-1]] for name,idx in FINGERS.items() if name != "thumb"}
        if mapper:
            mapped={name:a for name,p in fingertips.items() if (a:=mapper.assign(p))}
            assignments={name:(a.string,a.fret) for name,a in mapped.items()}
            canonical={name:a.canonical for name,a in mapped.items()}
        else:
            assert box is not None
            assignments = {name: cell for name,p in fingertips.items() if (cell := point_to_cell(p,box,left_handed))}; canonical={}
        flexions = {name: finger_flexion(points,name) for name in fingertips}
        best = (-1., (), 1., {})
        for fingering in chord.fingerings:
            distances=[]; identity=0; flex=0
            for pos in fingering:
                p=fingertips[pos.finger]
                if mapper:
                    px,py=mapper.to_canonical(p); cx,cy=mapper.cell_center(pos.string,pos.fret); dx=(px-cx)*4; dy=(py-cy)*6
                else:
                    cx,cy=cell_center(box,pos.string,pos.fret,left_handed); dx=(p[0]-cx)/(box.width/4); dy=(p[1]-cy)/(box.height/6)
                d=float(np.hypot(dx,dy)); distances.append(d)
                identity += float(d <= chord.visual_tolerance); flex += flexions[pos.finger]
            required={p.finger for p in fingering}
            extras=sum(flexions[n]>.38 and n not in required for n in flexions)
            mean=float(np.mean(distances)); proximity=float(np.mean(np.exp(-np.square(np.array(distances)/chord.visual_tolerance))))
            inside=sum(p.finger in assignments for p in fingering)/len(fingering)
            components={"proximity":proximity,"identity":identity/len(fingering),"inside":inside,
                "flexion":flex/len(fingering),"tracking":tracking,"extra_penalty":extras}
            score=.44*proximity+.18*components["identity"]+.14*inside+.12*components["flexion"]+.12*tracking-.06*extras
            if score>best[0]: best=(score,tuple(p.finger for p in fingering),mean,components)
        score=float(np.clip(best[0],0,1)); required=best[1]
        all_inside=all(f in assignments for f in required)
        if score>=chord.minimum_visual_confidence and all_inside: self.above+=1; self.below=0
        elif score<chord.minimum_visual_confidence-.10: self.below+=1; self.above=0
        if self.above>=self.stable_frames and not self.matched: self.matched=True; self.stable_since=time.monotonic()
        if self.below>=3: self.matched=False
        flexed=tuple(n for n,v in flexions.items() if v>.30 and n in assignments)
        expected={p.finger:p for p in chord.fingerings[0]}; diagnostics={}
        for name,pos in expected.items():
            detected=assignments.get(name); distance=(mapped[name].distance if mapper and name in mapped else None)
            status="Missing" if detected is None else ("Correct" if detected==(pos.string,pos.fret) and (distance or 0)<=chord.visual_tolerance else "Near" if distance is not None and distance<=chord.visual_tolerance*1.5 else "Incorrect")
            diagnostics[name]={"target":(pos.string,pos.fret),"detected":detected,"distance":distance,"status":status}
        return VisionResult(inside_fret_box=all_inside, confidence=score, match=self.matched,
            detected_shape=chord.display_name if self.matched else "Unmatched", fretting_fingers=flexed,
            assignments=assignments, average_cell_error=best[2], tracking_confidence=tracking,
            stable_frames=self.above if self.matched else 0, handedness=handedness,
            landmarks=tuple(tuple(float(v) for v in p) for p in points),
            joint_angles={name:tuple(round(x,1) for x in finger_angles(points,name)) for name in FINGERS},
            confidence_components=best[3],finger_diagnostics=diagnostics,canonical_points=canonical,stable_since=self.stable_since)
