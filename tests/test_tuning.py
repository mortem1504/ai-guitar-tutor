import numpy as np
from guitar_tutor.audio.tuning import estimate_tuning_offset,tuning_adjusted_frequency

def test_tuning_offset_estimate():
    factor=2**(-14/1200);peaks=tuple((f*factor,1.) for f in (110.,164.81,220.,261.63))
    assert abs(estimate_tuning_offset(peaks)+14)<1
    assert np.isclose(tuning_adjusted_frequency(440,-14),440*factor)
