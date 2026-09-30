from __future__ import annotations

from collections import deque
from threading import Condition, Event, Thread
import logging
import time
import numpy as np

from guitar_tutor.audio.chord_detector import ChordDetector
from guitar_tutor.config import TutorConfig
from guitar_tutor.state import AudioResult, SharedState

logger=logging.getLogger("guitar_tutor.audio")


def list_input_devices() -> list[dict]:
    import sounddevice as sd
    return [{"index": i, "name": d["name"], "channels": d["max_input_channels"],
             "sample_rate": int(d["default_samplerate"])}
            for i,d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def default_input_device() -> dict | None:
    import sounddevice as sd
    try:
        d=sd.query_devices(None,"input")
        return {"name":d["name"],"sample_rate":int(d["default_samplerate"]),"channels":d["max_input_channels"]}
    except Exception:
        return None


class AudioManager:
    def __init__(self, state: SharedState, config: TutorConfig):
        self.state, self.config = state, config
        self.detector = ChordDetector(config)
        self._blocks: deque[np.ndarray] = deque(maxlen=24)
        self._condition = Condition(); self._stop = Event(); self._stream = None; self._worker = None
        self._rolling = np.zeros(0, dtype=np.float32); self.overflows = 0

    @property
    def running(self): return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        if getattr(status, "input_overflow", False): self.overflows += 1
        block = np.asarray(indata[:,0], dtype=np.float32).copy()
        with self._condition:
            self._blocks.append(block); self._condition.notify()

    def start(self, device: int | None = None):
        if self.running: return
        import sounddevice as sd
        self._stop.clear(); self.detector.reset()
        try:
            info = sd.query_devices(device, "input")
            sample_rate = int(info["default_samplerate"] or self.config.sample_rate)
            self._sample_rate = sample_rate
            self._stream = sd.InputStream(device=device, channels=1, samplerate=sample_rate,
                blocksize=self.config.audio_block_size, dtype="float32", callback=self._callback)
            self._stream.start(); self._worker = Thread(target=self._run, name="guitar-audio-dsp", daemon=True)
            self._worker.start(); self.state.audio_active = True
            logger.info("Microphone started: %s at %s Hz",info["name"],sample_rate)
        except Exception as exc:
            self._stream = None
            message=self._friendly_error(exc)
            self.state.update_audio(AudioResult(timestamp=time.monotonic(), error=message))
            logger.exception("Microphone startup failed")
            raise

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        text=str(exc).lower()
        if "permission" in text or "-9986" in text: return "Microphone permission denied. Enable it in macOS Privacy & Security."
        if "device" in text and ("unavailable" in text or "invalid" in text): return "Selected microphone is unavailable. Choose another input device."
        return f"PortAudio failed: {exc}"

    def _run(self):
        hop = max(512, self.config.fft_size//4); since = 0
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._blocks or self._stop.is_set(), timeout=.25)
                blocks = list(self._blocks); self._blocks.clear()
            if not blocks: continue
            joined = np.concatenate(blocks); since += len(joined)
            self._rolling = np.concatenate((self._rolling, joined))[-self.config.fft_size:]
            if len(self._rolling) == self.config.fft_size and since >= hop:
                since = 0
                try:
                    result=self.detector.analyze(self._rolling, self._sample_rate, self.state.target)
                    result.queue_fill=len(self._blocks); self.state.update_audio(result)
                except Exception as exc:
                    logger.exception("Audio analysis failed")
                    self.state.update_audio(AudioResult(timestamp=time.monotonic(), error=f"Audio DSP failed: {exc}"))

    def stop(self):
        self._stop.set()
        with self._condition: self._condition.notify_all()
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            finally: self._stream = None
        if self._worker and self._worker.is_alive(): self._worker.join(timeout=1.5)
        self._worker = None; self._rolling = np.zeros(0, dtype=np.float32)
        with self._condition: self._blocks.clear()
        self.state.audio_active = False
        logger.info("Microphone stopped")


MicrophoneEngine = AudioManager
