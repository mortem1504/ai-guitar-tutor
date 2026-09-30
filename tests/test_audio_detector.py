import numpy as np
from guitar_tutor.audio.chord_detector import ChordDetector
from guitar_tutor.config import TutorConfig

def signal(freqs,sr=48000,n=8192):
    t=np.arange(n)/sr
    return sum(np.sin(2*np.pi*f*t) for f in freqs).astype(np.float32)

def test_silence_never_returns_chord():
    result=ChordDetector(TutorConfig()).analyze(np.zeros(8192,np.float32),48000,"E Major")
    assert result.raw_chord==result.stable_chord=="—"
    assert not result.match

def test_audio_voting_becomes_stable():
    detector=ChordDetector(TutorConfig(audio_history=5)); x=signal((164.81,207.65,246.94))
    results=[detector.analyze(x,48000,"E Major") for _ in range(5)]
    assert results[-1].stable_chord=="E Major"
    assert results[-1].match
