from __future__ import annotations
import html
import numpy as np
import streamlit as st
from guitar_tutor.chord_library import PITCH_CLASSES
from guitar_tutor.state import EvaluationState


def indicators(snap):
    def p(name,status,kind): return f'<span class="pill {kind}">{html.escape(name)} · {html.escape(status)}</span>'
    frame=snap.latest_frame
    if not snap.running:
        cam=("Ready","idle")
    elif frame is None:
        cam=("No frames","err")
    else:
        cam=("Active","ok")
    hand=("Active","ok") if snap.vision.hand_detected else (("Error","err") if snap.vision.error else ("Ready","idle"))
    guitar=("Detected","ok") if snap.vision.guitar.guitar_detected else (("Error","err") if snap.vision.guitar.error and not snap.vision.guitar.detector_available else (("Lost","err") if snap.vision.guitar.lost_frames else ("Searching","idle")))
    mic=("Active","ok") if snap.audio_active else (("Error","err") if snap.audio.error else ("Ready","idle"))
    dsp=("Active","ok") if snap.audio.timestamp else ("Ready","idle")
    st.markdown('<div class="status-row">'+p("Camera",*cam)+p("Guitar",*guitar)+p("Hand Tracking",*hand)+p("Microphone",*mic)+p("Audio DSP",*dsp)+"</div>",unsafe_allow_html=True)


def validation_card(ev):
    classes={EvaluationState.PERFECT:"perfect",EvaluationState.WAITING_FOR_STRUM:"waiting",
      EvaluationState.VISION_ONLY_CORRECT:"warning",EvaluationState.AUDIO_ONLY_CORRECT:"warning",
      EvaluationState.SYSTEM_ERROR:"incorrect",EvaluationState.AUDIO_MISMATCH:"incorrect",EvaluationState.VISION_MISMATCH:"incorrect"}
    cls=classes.get(ev.state,"waiting")
    st.markdown(f'<div class="validation {cls}"><h2>{html.escape(ev.state.value.replace("_"," "))}</h2><p>{html.escape(ev.message)}</p></div>',unsafe_allow_html=True)


def metrics(snap):
    v=snap.vision;a=snap.audio;e=snap.evaluation
    vals=(("Vision Match",f"{v.confidence:.0%}"),("Audio Match",f"{a.confidence:.0%}"),
          ("Combined",f"{e.combined_confidence:.0%}"),("Detected Chord",a.stable_chord),
          ("Audio Level",f"{a.dbfs:.1f} dBFS"),("Tuning Offset",f"{a.tuning_offset_cents:+.0f} cents"))
    st.markdown('<div class="metric-grid">'+''.join(f'<div class="metric"><small>{html.escape(k)}</small><strong>{html.escape(val)}</strong></div>' for k,val in vals)+"</div>",unsafe_allow_html=True)


def audio_charts(audio,target):
    st.caption("Microphone waveform")
    st.line_chart(audio.waveform, height=105)
    import pandas as pd
    data=np.asarray(audio.chroma); target_mask=np.array([1. if p in target.pitch_classes else 0. for p in PITCH_CLASSES])
    st.caption("12-bin chromagram · target tones are outlined by the Target series")
    st.bar_chart(pd.DataFrame({"Observed":data,"Target":target_mask*.12},index=PITCH_CLASSES),height=190)
