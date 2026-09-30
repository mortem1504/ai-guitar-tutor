import numpy as np
from guitar_tutor.audio.chroma import chromagram, frequency_to_midi, frequency_to_pitch_class
from guitar_tutor.audio.dsp import spectrum

def test_e_major_signal_chroma():
    sr=48000;n=8192;t=np.arange(n)/sr
    x=sum(np.sin(2*np.pi*f*t) for f in (164.81,207.65,246.94)).astype(np.float32)
    f,m=spectrum(x,sr); c=chromagram(f,m)
    assert set(np.argsort(c)[-3:])=={4,8,11}

def test_frequency_conversions():
    assert np.isclose(frequency_to_midi(440),69)
    assert frequency_to_pitch_class(440)==9
    assert frequency_to_pitch_class(261.63)==0

def test_chromagram_is_normalized_and_silence_safe():
    f=np.linspace(0,2000,1001)
    assert np.linalg.norm(chromagram(f,np.zeros_like(f)))==0
    c=chromagram(f,np.ones_like(f))
    assert np.isclose(np.linalg.norm(c),1)
