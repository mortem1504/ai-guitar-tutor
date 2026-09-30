from __future__ import annotations
import numpy as np

def estimate_tuning_offset(peaks: tuple[tuple[float,float],...]) -> float:
    offsets=[]; weights=[]
    for frequency,strength in peaks:
        if frequency<=0: continue
        midi=69+12*np.log2(frequency/440.0); offsets.append((midi-round(midi))*100); weights.append(max(strength,.01))
    if not offsets:return 0.
    order=np.argsort(offsets); values=np.asarray(offsets)[order]; w=np.asarray(weights)[order]; cumulative=np.cumsum(w)
    return float(values[np.searchsorted(cumulative,cumulative[-1]/2)])

def tuning_adjusted_frequency(frequency:float,offset_cents:float)->float:
    return float(frequency*2**(offset_cents/1200))
