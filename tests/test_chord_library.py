import numpy as np
from guitar_tutor.chord_library import CHORDS, PITCH_CLASSES

def test_required_chords_and_templates():
    assert set(CHORDS)=={"E Major","A Minor","C Major","G Major"}
    for chord in CHORDS.values():
        assert len(chord.template)==12
        assert np.isclose(np.linalg.norm(chord.template),1)
        assert all(chord.template[PITCH_CLASSES.index(pc)]>0 for pc in chord.pitch_classes)

def test_g_major_alternative():
    assert [len(x) for x in CHORDS["G Major"].fingerings]==[3,4]
