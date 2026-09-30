from __future__ import annotations
from dataclasses import dataclass
from guitar_tutor.chord_library import Chord
from guitar_tutor.state import AudioResult,VisionResult

STRING_NAMES={6:"low E",5:"A",4:"D",3:"G",2:"B",1:"high E"}

@dataclass(frozen=True)
class TutorFeedback:
    headline:str; details:tuple[str,...]; finger_rows:tuple[dict,...]; tone_rows:tuple[dict,...]

def build_feedback(chord:Chord,vision:VisionResult,audio:AudioResult)->TutorFeedback:
    details=[];rows=[]
    for finger,data in vision.finger_diagnostics.items():
        target=data["target"];detected=data["detected"];status=data["status"]
        rows.append({"Finger":finger.title(),"Target":f"{STRING_NAMES[target[0]]} fret {target[1]}",
          "Detected":"Missing" if not detected else f"{STRING_NAMES[detected[0]]} fret {detected[1]}",
          "Distance":"—" if data["distance"] is None else f"{data['distance']:.2f}","Status":status})
        if detected and status=="Incorrect":
            if detected[1]!=target[1]: details.append(f"{finger.title()} finger is {abs(detected[1]-target[1])} fret {'too high' if detected[1]>target[1] else 'too low'}.")
            elif detected[0]!=target[0]:details.append(f"{finger.title()} finger is closest to the {STRING_NAMES[detected[0]]} string; target the {STRING_NAMES[target[0]]} string.")
        elif not detected:details.append(f"{finger.title()} fingertip is not inside the calibrated fretboard.")
    tones=[]
    for pc in chord.pitch_classes:
        strength=audio.pitch_class_strengths.get(pc,0.); label="Strong" if strength>=.35 else "Present" if strength>=.18 else "Weak"
        tones.append({"Pitch class":pc,"Strength":label,"Energy":round(strength,3)})
        if label=="Weak" and audio.dbfs>-45:details.append(f"The {pc} chord tone is weak; a string may be muted.")
    if abs(audio.tuning_offset_cents)>=10:details.append(f"The guitar appears approximately {abs(audio.tuning_offset_cents):.0f} cents {'sharp' if audio.tuning_offset_cents>0 else 'flat'}.")
    return TutorFeedback(details[0] if details else "Measured placement and chord tones look balanced.",tuple(dict.fromkeys(details)),tuple(rows),tuple(tones))
