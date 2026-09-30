from __future__ import annotations

import time
import logging
import cv2
import numpy as np

from guitar_tutor.chord_library import get_chord
from guitar_tutor.config import TutorConfig
from guitar_tutor.state import FretboardState, GuitarDetectionState, SharedState, VisionResult
from guitar_tutor.vision.chord_shape import ShapeClassifier
from guitar_tutor.vision.fretboard_mapper import FretboardMapper
from guitar_tutor.vision.hand_geometry import CONNECTIONS, FINGERS, LandmarkSmoother
from guitar_tutor.vision.mediapipe_adapter import HandTracker
from guitar_tutor.vision.guitar_detector import GuitarDetector, GuitarTracker
from guitar_tutor.vision.fretboard_detector import FretboardDetector, FretboardTracker

logger=logging.getLogger("guitar_tutor.vision")


class GuitarVideoProcessor:
    def __init__(self, state: SharedState, config: TutorConfig):
        self.state, self.config = state, config
        self.tracker = HandTracker(__import__("guitar_tutor.config",fromlist=["MODEL_PATH"]).MODEL_PATH,
            config.min_hand_detection_confidence,config.min_hand_presence_confidence,config.min_tracking_confidence,config.max_hands)
        self.smoothers={}
        self.classifier=ShapeClassifier(config.vision_stable_frames); self._last=0.; self._fps=0.; self._timestamp=0
        self.guitar_detector=GuitarDetector(config.guitar_model, config.guitar_detection_confidence,
            config.guitar_detection_interval)
        self.guitar_tracker=GuitarTracker(config.guitar_stable_frames, config.guitar_lost_grace_seconds)
        self.fretboard_detector=FretboardDetector(config.fretboard_min_confidence)
        self.fretboard_tracker=FretboardTracker(config.fretboard_ema_alpha, config.fretboard_stable_frames)
        self.performance={"mediapipe_ms":0.0,"yolo_ms":0.0,"fretboard_ms":0.0}

    @property
    def fps(self):
        return self._fps

    def process(self, image: np.ndarray) -> np.ndarray:
        started=time.perf_counter(); image=image.copy()
        if self.config.mirror: image=cv2.flip(image,1)
        now=time.monotonic(); self._timestamp=max(self._timestamp+1,int(now*1000))
        try:
            yolo_started=time.perf_counter(); guitar_detection=self.guitar_detector.detect(image)
            self.performance["yolo_ms"]=(time.perf_counter()-yolo_started)*1000
            guitar=self.guitar_tracker.update(guitar_detection, now,
                self.guitar_detector.available, self.guitar_detector.error)
            fret_started=time.perf_counter(); fret_observation=self.fretboard_detector.detect(image, guitar)
            self.performance["fretboard_ms"]=(time.perf_counter()-fret_started)*1000
            fretboard, geometry=self.fretboard_tracker.update(fret_observation, guitar, now)
            auto_calibrated=geometry is not None and fretboard.status in ("CALIBRATED", "TRACKING")
            if auto_calibrated:
                self.config.calibration=self._calibration_from_geometry(geometry)
            mp_started=time.perf_counter()
            result=self.tracker.detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),self._timestamp)
            self.performance["mediapipe_ms"]=(time.perf_counter()-mp_started)*1000
            if not result:
                for smoother in self.smoothers.values(): smoother.reset()
                self.classifier.reset()
                vr=VisionResult(timestamp=now,hand_detected=False,fps=self._fps,latency_ms=(time.perf_counter()-started)*1000)
            else:
                smoothed=[]
                for index,(raw,handedness,tracking) in enumerate(result):
                    key=f"{handedness}-{index}";smoother=self.smoothers.setdefault(key,LandmarkSmoother(self.config.landmark_ema_alpha))
                    smoothed.append((smoother.update(raw),handedness,tracking))
                if auto_calibrated:
                    mapper=FretboardMapper(self.config.calibration,self.config.guitar_handedness=="Left-handed")
                    # The fretting hand is the one with the most fingertips in the
                    # calibrated neck; palm proximity breaks ties.
                    def candidate_score(item):
                        pts,_,confidence=item
                        inside=sum(mapper.assign(pts[i]) is not None for i in (8,12,16,20))
                        wrist=mapper.to_canonical(pts[0]);proximity=max(0.,1.-abs(wrist[1]-.5))
                        return inside*2+proximity+.1*confidence
                    points,handedness,tracking=max(smoothed,key=candidate_score)
                    vr=self.classifier.classify(points,get_chord(self.state.target),tracking=tracking,handedness=handedness,mapper=mapper)
                else:
                    points,handedness,tracking=smoothed[0]
                    vr=VisionResult(hand_detected=True,handedness=handedness,tracking_confidence=tracking,
                        landmarks=tuple(tuple(float(v) for v in p) for p in points))
                vr.timestamp=now; vr.hand_detected=True; vr.fps=self._fps;vr.hand_count=len(smoothed);vr.all_handedness=tuple(x[1] for x in smoothed)
                for pts,label,_ in smoothed:self._draw_hand(image,pts,vr,pts is points,label)
            vr.guitar=guitar; vr.fretboard=fretboard
            if not guitar.guitar_detected:
                vr.hand_detected=False; vr.inside_fret_box=False; vr.match=False; vr.confidence=0.; vr.detected_shape="—"
            self._draw_vision_overlay(image,guitar,fretboard)
            self._draw_fret_box(image,get_chord(self.state.target),vr,auto_calibrated)
            vr.latency_ms=(time.perf_counter()-started)*1000; self.state.update_vision(vr)
        except Exception as exc:
            vr=VisionResult(timestamp=now,error=f"Vision frame: {exc}",fps=self._fps,
                guitar=self.guitar_tracker.state, fretboard=self.fretboard_tracker.state)
            self.state.update_vision(vr); cv2.putText(image,"Vision temporarily unavailable",(18,32),0,.65,(30,80,255),2,cv2.LINE_AA)
            logger.warning("Vision frame failed: %s",exc)
        dt=now-self._last
        if dt>0: self._fps=.88*self._fps+.12*(1/dt) if self._last else 0
        self._last=now
        cv2.putText(image,f"{self._fps:.0f} FPS  VISION {vr.confidence:.0%}",(16,image.shape[0]-18),0,.48,(230,235,240),1,cv2.LINE_AA)
        return image

    def recv(self, frame):
        """Compatibility shim for callers outside the local worker."""
        return self.process(frame)

    def _xy(self,p,w,h): return int(p[0]*w),int(p[1]*h)

    def _draw_hand(self,img,points,vr,primary=True,label="Unknown"):
        h,w=img.shape[:2]
        line_color=(122,220,170) if primary else (170,145,220)
        for a,b in CONNECTIONS: cv2.line(img,self._xy(points[a],w,h),self._xy(points[b],w,h),line_color,2,cv2.LINE_AA)
        for name,ids in FINGERS.items():
            x,y=self._xy(points[ids[-1]],w,h); matching=primary and name in vr.assignments
            color=(63,220,140) if matching else ((53,180,245) if primary else (185,125,230))
            cv2.circle(img,(x,y),8,color,-1,cv2.LINE_AA)
            cv2.putText(img,name[0].upper(),(x+8,y-7),0,.42,(245,245,245),1,cv2.LINE_AA)
        wx,wy=self._xy(points[0],w,h);cv2.putText(img,f"{label} {'FRETTING' if primary else 'SECONDARY'}",(wx+8,wy+18),0,.4,line_color,1,cv2.LINE_AA)

    @staticmethod
    def _calibration_from_geometry(geometry):
        from guitar_tutor.config import FretboardCalibration
        return FretboardCalibration(tuple(tuple(float(v) for v in p) for p in geometry.polygon), True)

    def _draw_vision_overlay(self,img,guitar,fretboard):
        h,w=img.shape[:2]
        if guitar.bbox:
            x1,y1,x2,y2=(int(v) for v in (guitar.bbox[0]*w,guitar.bbox[1]*h,guitar.bbox[2]*w,guitar.bbox[3]*h))
            color=(90,210,155) if guitar.guitar_detected else (120,150,170)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,1,cv2.LINE_AA)
            if self.config.debug_vision:
                cv2.putText(img,f"GUITAR {guitar.confidence:.0%}",(x1,max(18,y1-8)),0,.5,color,1,cv2.LINE_AA)
        if self.config.debug_vision and guitar.neck_region:
            x1,y1,x2,y2=guitar.neck_region
            cv2.rectangle(img,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),(180,150,80),1,cv2.LINE_AA)
        if fretboard.polygon and fretboard.status not in ("NO_GUITAR", "SEARCHING_FOR_FRETBOARD"):
            points=np.array([(int(x*w),int(y*h)) for x,y in fretboard.polygon],np.int32)
            cv2.polylines(img,[points],True,(80,215,255),1,cv2.LINE_AA)
            if self.config.debug_vision:
                cv2.putText(img,f"FRETBOARD {fretboard.confidence:.0%}",(points[0][0],max(18,points[0][1]-8)),0,.45,(80,215,255),1,cv2.LINE_AA)

    def _draw_fret_box(self,img,chord,vr,auto_calibrated=False):
        h,w=img.shape[:2]
        if not auto_calibrated:
            message={"NO_GUITAR":"LOOKING FOR GUITAR","SEARCHING_FOR_FRETBOARD":"LOCATING FRETBOARD",
                "RECALIBRATING":"RECALIBRATING..."}.get(vr.fretboard.status,"SHOW FIRST FOUR FRETS")
            cv2.putText(img,message,(18,34),0,.62,(80,215,255),2,cv2.LINE_AA)
            return
        corners=self.config.calibration.corners
        pts=np.array([(int(x*w),int(y*h)) for x,y in corners],np.int32)
        cv2.polylines(img,[pts],True,(80,215,255),1,cv2.LINE_AA)
        mapper=FretboardMapper(self.config.calibration,self.config.guitar_handedness=="Left-handed")
        def pixel(p): x,y=mapper.to_image(p); return int(x*w),int(y*h)
        for i in range(1,4): cv2.line(img,pixel((i/4,0)),pixel((i/4,1)),(120,150,165),1,cv2.LINE_AA)
        for i in range(6):
            y=(i+.5)/6; cv2.line(img,pixel((0,y)),pixel((1,y)),(105,125,140),1,cv2.LINE_AA)
        for fret in range(1,5):
            xy=pixel(((fret-.5)/4,0)); cv2.putText(img,str(fret),(xy[0]-4,xy[1]-7),0,.4,(210,220,225),1)
        for i,label in enumerate(("6E","5A","4D","3G","2B","1E")):
            xy=pixel((0,(i+.5)/6)); cv2.putText(img,label,(xy[0]-25,xy[1]+4),0,.35,(200,210,220),1)
        for pos in chord.fingerings[0]:
            xy=pixel(mapper.cell_center(pos.string,pos.fret))
            cv2.drawMarker(img,xy,(245,230,90),cv2.MARKER_DIAMOND,16,2,cv2.LINE_AA)
            cv2.putText(img,pos.finger[0].upper(),(xy[0]+9,xy[1]+4),0,.38,(245,230,90),1,cv2.LINE_AA)

    def close(self):
        self.tracker.close()
        self.guitar_detector.close()
