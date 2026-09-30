import numpy as np
from guitar_tutor.audio.chroma import cosine_similarity
from guitar_tutor.chord_library import CHORDS
from guitar_tutor.audio.dsp import spectrum
from guitar_tutor.audio.chroma import chromagram

def test_templates_self_match_best():
    for name,chord in CHORDS.items():
        scores={n:cosine_similarity(chord.template,c.template) for n,c in CHORDS.items()}
        assert max(scores,key=scores.get)==name
        assert np.isclose(scores[name],1)

def test_realistic_synthetic_chords_match_templates():
    rng=np.random.default_rng(42); sr=48000; n=8192; t=np.arange(n)/sr
    for name,chord in CHORDS.items():
        # Detune each partial slightly and add deterministic room-like noise.
        x=np.zeros(n)
        for i,freq in enumerate(chord.sounding_frequencies):
            detuned=freq*2**(((i%3)-1)*7/1200)
            x+=(1/(1+i*.12))*np.sin(2*np.pi*detuned*t)
        x+=rng.normal(0,.08,n)
        f,m=spectrum(x.astype(np.float32),sr); measured=chromagram(f,m)
        scores={candidate:cosine_similarity(measured,c.template) for candidate,c in CHORDS.items()}
        assert max(scores,key=scores.get)==name
