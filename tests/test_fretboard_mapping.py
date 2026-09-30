import numpy as np
from guitar_tutor.config import FretboardCalibration
from guitar_tutor.vision.fretboard_mapper import FretboardMapper

def test_homography_roundtrip_and_assignment():
    mapper=FretboardMapper(FretboardCalibration(calibrated=True))
    for string in range(1,7):
        for fret in range(1,5):
            canonical=mapper.cell_center(string,fret);image=mapper.to_image(canonical)
            assert np.allclose(mapper.to_canonical(image),canonical,atol=1e-5)
            assignment=mapper.assign(image)
            assert (assignment.string,assignment.fret)==(string,fret)
            assert assignment.distance<1e-4
