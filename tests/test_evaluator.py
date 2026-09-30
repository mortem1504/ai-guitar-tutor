import time
from guitar_tutor.config import TutorConfig
from guitar_tutor.fusion.evaluator import Evaluator
from guitar_tutor.state import AudioResult, EvaluationState, VisionResult

def test_requires_both_modalities():
    now=time.monotonic(); ev=Evaluator(TutorConfig())
    v=VisionResult(timestamp=now,hand_detected=True,inside_fret_box=True,confidence=.9,match=True)
    a=AudioResult(timestamp=now,stable_chord="E Major",confidence=.9,match=False,dbfs=-20)
    assert ev.evaluate("E Major",v,a,now).state is EvaluationState.VISION_ONLY_CORRECT
    a.match=True
    assert ev.evaluate("E Major",v,a,now).state is EvaluationState.PERFECT

def test_stale_audio_not_perfect():
    now=time.monotonic(); ev=Evaluator(TutorConfig())
    v=VisionResult(timestamp=now,hand_detected=True,inside_fret_box=True,confidence=.9,match=True)
    a=AudioResult(timestamp=now-2,stable_chord="E Major",confidence=.9,match=True,dbfs=-20)
    assert ev.evaluate("E Major",v,a,now).state is EvaluationState.SHAPE_CORRECT_WAITING_FOR_STRUM

def test_transition_states_and_attempt_debounce():
    now=time.monotonic(); ev=Evaluator(TutorConfig())
    v=VisionResult(timestamp=now,hand_detected=True,inside_fret_box=False)
    a=AudioResult(timestamp=now,stable_chord="A Minor",dbfs=-20)
    assert ev.evaluate("E Major",v,a,now).state is EvaluationState.HAND_OUTSIDE_FRETBOARD
    v.inside_fret_box=True
    a.strum_timestamp=now+.05
    first=ev.evaluate("E Major",v,a,now+.1,strum_event=True)
    second=ev.evaluate("E Major",v,a,now+.2,strum_event=True)
    assert first.state is EvaluationState.AUDIO_MISMATCH
    assert first.attempt_event==second.attempt_event==1
