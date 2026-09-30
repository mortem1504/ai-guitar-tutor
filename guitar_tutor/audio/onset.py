from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

@dataclass(frozen=True)
class OnsetResult:
    detected: bool; state: str; spectral_flux: float; energy_slope: float; timestamp: float

class StrumOnsetDetector:
    def __init__(self,cooldown=.65):
        self.cooldown=cooldown; self.previous_rms=0.; self.previous_spectrum=None; self.last_onset=-1e9; self.state="quiet"
    def reset(self): self.previous_rms=0.; self.previous_spectrum=None; self.last_onset=-1e9; self.state="quiet"
    def update(self,rms:float,magnitude:np.ndarray,now:float|None=None)->OnsetResult:
        now=time.monotonic() if now is None else now
        mag=np.sqrt(np.maximum(magnitude,0)); mag/=np.linalg.norm(mag)+1e-9
        flux=0. if self.previous_spectrum is None else float(np.mean(np.maximum(mag-self.previous_spectrum,0))*100)
        slope=float((rms-self.previous_rms)/max(self.previous_rms,1e-4))
        detected=now-self.last_onset>=self.cooldown and rms>.004 and slope>.45 and flux>.018
        if detected: self.last_onset=now; self.state="attack"
        elif rms<.002: self.state="quiet"
        elif slope<-.18: self.state="decay"
        else: self.state="sustain"
        self.previous_rms=.65*self.previous_rms+.35*rms; self.previous_spectrum=mag
        return OnsetResult(detected,self.state,flux,slope,self.last_onset if detected else 0.)
