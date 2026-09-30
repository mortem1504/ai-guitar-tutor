from __future__ import annotations

import numpy as np


def frequency_to_midi(frequency: float) -> float:
    if frequency <= 0: raise ValueError("frequency must be positive")
    return float(69 + 12*np.log2(frequency/440.0))


def frequency_to_pitch_class(frequency: float) -> int:
    return int(round(frequency_to_midi(frequency))) % 12


def chromagram(freqs: np.ndarray, magnitudes: np.ndarray, low: float=70., high: float=1400.) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high) & (magnitudes > 0)
    f, m = freqs[mask], magnitudes[mask]
    result = np.zeros(12, dtype=np.float64)
    if not len(f): return result.astype(np.float32)
    midi = 69 + 12*np.log2(f/440.0)
    nearest = np.rint(midi)
    # Reject bins far between semitones and compress magnitude dominance.
    cents = np.abs(midi-nearest) * 100
    weights = np.sqrt(m) * np.exp(-0.5*(cents/35.0)**2)
    np.add.at(result, nearest.astype(int) % 12, weights)
    norm = np.linalg.norm(result)
    return (result/norm if norm > 0 else result).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom else 0.0
