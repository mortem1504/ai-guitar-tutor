import numpy as np

from guitar_tutor.state import GuitarDetectionState
from guitar_tutor.vision.fretboard_detector import FretboardDetector, FretboardObservation, FretboardTracker
from guitar_tutor.vision.fretboard_geometry import geometry_from_polygon
from guitar_tutor.vision.guitar_detector import GuitarDetection, GuitarTracker


def test_guitar_tracker_requires_stable_presence_and_graceful_loss():
    tracker = GuitarTracker(stable_frames=3, lost_grace_seconds=.5)
    detection = GuitarDetection(.8, (.1, .1, .8, .8))
    assert not tracker.update(detection, now=1.).guitar_detected
    assert not tracker.update(detection, now=1.1).guitar_detected
    assert tracker.update(detection, now=1.2).guitar_detected
    assert tracker.update(None, now=1.5).guitar_detected
    assert not tracker.update(None, now=1.9).guitar_detected


def test_fretboard_geometry_round_trip_and_future_api():
    geometry = geometry_from_polygon(((.2, .2), (.8, .25), (.77, .65), (.17, .6)), .9)
    point = geometry.fretboard_to_camera((.5, .5))
    assert np.allclose(geometry.camera_to_fretboard(point), (.5, .5), atol=1e-4)
    mapped = geometry.point_to_fretboard(*point)
    assert mapped["inside_fretboard"]
    assert mapped["string"] is None
    assert mapped["fret"] == 2


def test_fretboard_tracker_stabilizes_candidate_and_handles_no_guitar():
    guitar = GuitarDetectionState(guitar_detected=True, stable_frames=5, bbox=(.1, .1, .9, .9))
    detector = FretboardDetector(min_confidence=0.)
    geometry = geometry_from_polygon(((.2, .2), (.8, .25), (.78, .6), (.18, .55)), .8)
    observation = FretboardObservation(geometry, .8, .6)
    tracker = FretboardTracker(alpha=.5, stable_frames=2)
    first, _ = tracker.update(observation, guitar, now=1.)
    second, result = tracker.update(observation, guitar, now=1.1)
    assert first.status == "CALIBRATING"
    assert second.status == "CALIBRATED"
    assert result is not None
    missing, result = tracker.update(None, GuitarDetectionState(), now=2.)
    assert missing.status == "NO_GUITAR"
    assert result is None
