import numpy as np
from guitar_tutor.config import FretBox
from guitar_tutor.vision.chord_shape import cell_center, point_to_cell
from guitar_tutor.vision.hand_geometry import joint_angle
from guitar_tutor.vision.chord_shape import ShapeClassifier
from guitar_tutor.chord_library import CHORDS
from guitar_tutor.vision.hand_geometry import FINGERS

def test_joint_angle():
    assert np.isclose(joint_angle(np.array([0,0]),np.array([1,0]),np.array([2,0])),180)
    assert np.isclose(joint_angle(np.array([0,0]),np.array([1,0]),np.array([1,1])),90)

def test_cells_round_trip():
    box=FretBox()
    for string in range(1,7):
        for fret in range(1,5): assert point_to_cell(cell_center(box,string,fret),box)==(string,fret)

def test_left_handed_cells_round_trip():
    box=FretBox()
    for string in range(1,7):
        for fret in range(1,5): assert point_to_cell(cell_center(box,string,fret,True),box,True)==(string,fret)

def test_visual_template_scores_exact_positions():
    box=FretBox(); chord=CHORDS["E Major"]; points=np.full((21,3),-.2,dtype=np.float32)
    for pos in chord.fingerings[0]:
        points[FINGERS[pos.finger][-1],:2]=cell_center(box,pos.string,pos.fret)
    result=ShapeClassifier(stable_frames=1).classify(points,chord,box,tracking=.95)
    assert result.confidence>.7
    assert result.match
