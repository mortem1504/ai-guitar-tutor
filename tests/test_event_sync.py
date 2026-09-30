from guitar_tutor.fusion.event_sync import EventSynchronizer
from guitar_tutor.state import AudioResult,VisionResult

def test_sync_selects_recent_pre_strum_vision_once():
    sync=EventSynchronizer();sync.add_vision(VisionResult(timestamp=10.0,match=True));sync.add_vision(VisionResult(timestamp=10.4,match=True))
    audio=AudioResult(timestamp=10.75,strum_timestamp=10.6)
    event=sync.synchronize(audio)
    assert event and event.vision.timestamp==10.4
    assert sync.synchronize(audio) is None

def test_sync_rejects_stale_vision():
    sync=EventSynchronizer(vision_lookback=.5);sync.add_vision(VisionResult(timestamp=1.0,match=True))
    assert sync.synchronize(AudioResult(timestamp=2.1,strum_timestamp=2.0)) is None
