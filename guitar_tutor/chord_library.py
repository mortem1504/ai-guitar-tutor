from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class FingerPosition:
    finger: str
    string: int
    fret: int


@dataclass(frozen=True)
class Chord:
    display_name: str
    short_name: str
    pitch_classes: tuple[str, ...]
    fingerings: tuple[tuple[FingerPosition, ...], ...]
    sounding_frequencies: tuple[float, ...]
    visual_tolerance: float = .58
    minimum_visual_confidence: float = .67
    minimum_audio_confidence: float = .58

    @property
    def template(self) -> np.ndarray:
        values = np.zeros(12, dtype=np.float32)
        for i, pc in enumerate(self.pitch_classes):
            values[PITCH_CLASSES.index(pc)] = 1.18 if i == 0 else 1.0
        return values / np.linalg.norm(values)

    @property
    def expected_fingers(self) -> tuple[str, ...]:
        return tuple(p.finger for p in self.fingerings[0])

    @property
    def root_pitch_class(self) -> str: return self.pitch_classes[0]
    @property
    def third(self) -> str: return self.pitch_classes[1]
    @property
    def fifth(self) -> str: return self.pitch_classes[2]

    def ascii_diagram(self) -> str:
        frets = {p.string: p.fret for p in self.fingerings[0]}
        return "  ".join(f"{s}:{frets.get(s, 0)}" for s in range(6, 0, -1))


def _p(finger: str, string: int, fret: int) -> FingerPosition:
    return FingerPosition(finger, string, fret)


CHORDS: Mapping[str, Chord] = {
    "E Major": Chord("E Major", "E", ("E", "G#", "B"),
        ((_p("index",3,1), _p("middle",5,2), _p("ring",4,2)),),
        (82.41,123.47,164.81,207.65,246.94,329.63)),
    "A Minor": Chord("A Minor", "Am", ("A", "C", "E"),
        ((_p("index",2,1), _p("middle",4,2), _p("ring",3,2)),),
        (110.00,164.81,220.00,261.63,329.63)),
    "C Major": Chord("C Major", "C", ("C", "E", "G"),
        ((_p("index",2,1), _p("middle",4,2), _p("ring",5,3)),),
        (130.81,164.81,196.00,261.63,329.63)),
    "G Major": Chord("G Major", "G", ("G", "B", "D"),
        ((_p("index",5,2), _p("middle",6,3), _p("ring",1,3)),
         (_p("index",5,2), _p("middle",6,3), _p("ring",2,3), _p("pinky",1,3))),
        (98.00,123.47,146.83,196.00,246.94,392.00), visual_tolerance=.64),
}


def get_chord(name: str) -> Chord:
    try:
        return CHORDS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported chord {name!r}. Choose one of: {', '.join(CHORDS)}") from exc
