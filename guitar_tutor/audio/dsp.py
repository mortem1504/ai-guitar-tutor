from __future__ import annotations

import numpy as np


def dbfs(rms: float) -> float:
    return float(20 * np.log10(max(rms, 1e-6)))


def preprocess(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    x = x - np.mean(x)
    # Stable one-pole high-pass around 55 Hz, useful against handling/desk noise.
    rc, dt = 1.0 / (2*np.pi*55.0), 1.0/sample_rate
    alpha = rc/(rc+dt)
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * (y[i-1] + x[i] - x[i-1])
    return y


def spectrum(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    windowed = samples * np.hanning(len(samples))
    magnitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(samples), 1/sample_rate)
    return frequencies, magnitude


def dominant_peaks(freqs: np.ndarray, mags: np.ndarray, low=70., high=1400., count=8):
    valid = np.where((freqs >= low) & (freqs <= high))[0]
    if len(valid) < 3: return ()
    m = mags[valid]
    local = np.where((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]))[0] + 1
    if not len(local): return ()
    chosen = local[np.argsort(m[local])[-count:]][::-1]
    peak_max = max(float(m[chosen[0]]), 1e-9)
    return tuple((float(freqs[valid[i]]), float(m[i]/peak_max)) for i in chosen)


def cents_distance(detected: float, reference: float) -> float:
    return float(1200 * np.log2(detected/reference))
