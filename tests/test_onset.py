import numpy as np
from guitar_tutor.audio.onset import StrumOnsetDetector

def test_quiet_attack_sustain_is_one_onset():
    detector=StrumOnsetDetector(cooldown=.6);quiet=np.zeros(64);attack=np.ones(64)
    detector.update(0.,quiet,1.0)
    first=detector.update(.1,attack,1.1)
    second=detector.update(.11,attack,1.2)
    assert first.detected and first.state=="attack"
    assert not second.detected
