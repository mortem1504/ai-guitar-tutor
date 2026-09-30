from __future__ import annotations

from collections import Counter, deque
import time
import numpy as np

from guitar_tutor.audio.chroma import chromagram, cosine_similarity
from guitar_tutor.audio.dsp import cents_distance, dbfs, dominant_peaks, preprocess, spectrum
from guitar_tutor.chord_library import CHORDS
from guitar_tutor.config import TutorConfig
from guitar_tutor.state import AudioResult
from guitar_tutor.audio.onset import StrumOnsetDetector
from guitar_tutor.audio.tuning import estimate_tuning_offset, tuning_adjusted_frequency


class ChordDetector:
    def __init__(self, config: TutorConfig):
        self.config = config
        self.history: deque[str] = deque(maxlen=config.audio_history)
        self.smooth = np.zeros(12, dtype=np.float32)
        self._matched = False
        self.onset=StrumOnsetDetector(config.onset_cooldown_seconds); self.tuning_offset=0.

    def reset(self):
        self.history.clear(); self.smooth[:] = 0; self._matched = False; self.onset.reset(); self.tuning_offset=0.

    def analyze(self, samples: np.ndarray, sample_rate: int, target_name: str) -> AudioResult:
        start = time.perf_counter(); x = preprocess(samples, sample_rate)
        rms = float(np.sqrt(np.mean(x*x))); level = dbfs(rms)
        waveform = np.asarray(samples[-1024:], dtype=np.float32).copy()
        freqs, mags = spectrum(x, sample_rate)
        onset=self.onset.update(rms,mags,time.monotonic())
        if level < self.config.silence_dbfs:
            self.history.append("—"); self._matched = False
            return AudioResult(timestamp=time.monotonic(), rms=rms, dbfs=level, waveform=waveform,
                sample_rate=sample_rate,
                latency_ms=(time.perf_counter()-start)*1000)
        chroma = chromagram(freqs, mags, self.config.min_frequency, self.config.max_frequency)
        self.smooth = .62*self.smooth + .38*chroma
        n = np.linalg.norm(self.smooth)
        if n: self.smooth /= n
        scores = {name: cosine_similarity(self.smooth, chord.template) for name,chord in CHORDS.items()}
        ordered = sorted(scores, key=scores.get, reverse=True); raw = ordered[0]
        best, second = scores[raw], scores[ordered[1]]
        margin = max(0., best-second); self.history.append(raw)
        votes = Counter(x for x in self.history if x != "—")
        stable = votes.most_common(1)[0][0] if votes and votes.most_common(1)[0][1] >= max(3, len(self.history)//2+1) else "—"
        peaks = dominant_peaks(freqs, mags, self.config.min_frequency, self.config.max_frequency)
        estimated=estimate_tuning_offset(peaks)
        if self.config.use_auto_tuning and abs(estimated)<50: self.tuning_offset=.9*self.tuning_offset+.1*estimated
        target = CHORDS[target_name]
        refs=tuple(tuning_adjusted_frequency(f,self.tuning_offset) for f in target.sounding_frequencies)
        covered = sum(any(abs(cents_distance(f, ref)) <= 45 for f,_ in peaks) for ref in refs)
        coverage = covered/len(target.sounding_frequencies)
        target_score = scores[target_name]
        score_conf = np.clip((target_score-.45)/.45, 0, 1)
        margin_conf = np.clip(margin/.18, 0, 1)
        confidence = float(.68*score_conf + .20*margin_conf + .12*coverage)
        on = stable == target_name and target_score >= target.minimum_audio_confidence and margin >= .025
        off = stable != target_name or target_score < target.minimum_audio_confidence-.08
        self._matched = on or (self._matched and not off)
        return AudioResult(time.monotonic(), raw, stable, confidence, self._matched, target_score,
            best, margin, rms, level, peaks, self.smooth.copy(), waveform, coverage,
            (time.perf_counter()-start)*1000, scores, sample_rate, 0, onset.spectral_flux,onset.state,onset.timestamp,
            self.tuning_offset,{pc:float(self.smooth[i]) for i,pc in enumerate(__import__('guitar_tutor.chord_library',fromlist=['PITCH_CLASSES']).PITCH_CLASSES)})
