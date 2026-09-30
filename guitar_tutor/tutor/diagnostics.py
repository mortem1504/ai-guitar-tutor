from __future__ import annotations

def event_timeline(snapshot)->dict:
    v,a,e=snapshot.vision,snapshot.audio,snapshot.evaluation
    return {"Hand stable":v.stable_since or None,"Strum detected":a.strum_timestamp or None,
      "Audio analyzed":a.timestamp or None,"Result":e.timestamp or None,
      "vision_to_strum_ms":round((a.strum_timestamp-v.stable_since)*1000,1) if a.strum_timestamp and v.stable_since else None,
      "strum_to_audio_ms":round((a.timestamp-a.strum_timestamp)*1000,1) if a.strum_timestamp else None}
