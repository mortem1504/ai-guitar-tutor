from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from guitar_tutor.state import AudioResult,VisionResult

@dataclass(frozen=True)
class SynchronizedAttempt:
    strum_timestamp:float; vision:VisionResult; audio:AudioResult; vision_age_ms:float; audio_delay_ms:float

class EventSynchronizer:
    def __init__(self,vision_lookback=1.0,audio_after=.4):
        self.vision_lookback=vision_lookback;self.audio_after=audio_after;self.history=deque(maxlen=60);self.last_strum=0.
    def add_vision(self,value):
        if value.timestamp and (not self.history or value.timestamp!=self.history[-1].timestamp):self.history.append(value)
    def synchronize(self,audio:AudioResult)->SynchronizedAttempt|None:
        s=audio.strum_timestamp
        if not s or s==self.last_strum or not (0<=audio.timestamp-s<=self.audio_after):return None
        candidates=[v for v in self.history if 0<=s-v.timestamp<=self.vision_lookback]
        if not candidates:return None
        self.last_strum=s;vision=min(candidates,key=lambda v:abs(s-v.timestamp))
        return SynchronizedAttempt(s,vision,audio,(s-vision.timestamp)*1000,(audio.timestamp-s)*1000)
