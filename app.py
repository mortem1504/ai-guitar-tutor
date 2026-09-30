from __future__ import annotations

import atexit
import logging
import streamlit as st

from guitar_tutor.audio.microphone import AudioManager, default_input_device, list_input_devices
from guitar_tutor.chord_library import CHORDS, get_chord
from guitar_tutor.config import FretboardCalibration, TutorConfig, load_fretboard_calibration, save_fretboard_calibration
from guitar_tutor.fusion.evaluator import FusionEngine
from guitar_tutor.state import EvaluationState, SharedState
from guitar_tutor.ui.components import audio_charts, indicators, metrics, validation_card
from guitar_tutor.ui.styles import CSS
from guitar_tutor.tutor.feedback import build_feedback
from guitar_tutor.tutor.diagnostics import event_timeline
from guitar_tutor.vision.camera_manager import CameraManager
from guitar_tutor.vision.vision_worker import VisionWorker

st.set_page_config(page_title="AI Guitar Chord Tutor",page_icon="🎸",layout="wide",initial_sidebar_state="collapsed")
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s · %(message)s")
logger=logging.getLogger("guitar_tutor.app")
st.markdown(CSS,unsafe_allow_html=True)


def initialize():
    if "core" not in st.session_state:
        config=TutorConfig(calibration=load_fretboard_calibration())
        state=SharedState(); mic=AudioManager(state,config)
        camera=CameraManager(config.camera_width,config.camera_height,config.camera_fps)
        vision=VisionWorker(state,config,camera); fusion=FusionEngine(state,config)
        st.session_state.core={"config":config,"state":state,"mic":mic,"camera":camera,"vision":vision,"fusion":fusion,"last_toast":0,
            "stats":{"attempts":0,"successes":0,"streak":0,"best":0,"last_attempt":0,"last_success":0}}
        atexit.register(vision.close); atexit.register(mic.stop); atexit.register(fusion.stop)
    return st.session_state.core


core=initialize(); config,state,mic,camera,vision,fusion=(core[k] for k in ("config","state","mic","camera","vision","fusion"))
st.markdown('<div class="hero"><h1>AI Guitar Chord Tutor</h1><p>See your fingers. Hear your guitar. Correct both in real time.</p><small>🔒 Camera and microphone processing stays on this device.</small></div>',unsafe_allow_html=True)

try: devices=list_input_devices()
except Exception: devices=[]
try: default_device=default_input_device()
except Exception: default_device=None

with st.sidebar:
    st.header("Tutor setup")
    target_name=st.selectbox("Target Chord",list(CHORDS),index=list(CHORDS).index(state.target))
    state.set_target(target_name)
    config.mirror=st.toggle("Mirror webcam",config.mirror)
    config.guitar_handedness=st.radio("Guitar mode",("Right-handed","Left-handed"),horizontal=True)
    camera_device=st.selectbox("Local camera",[0,1],index=0)
    camera.device=int(camera_device)
    labels=["System Default"]+[f"{d['index']} · {d['name']} · {d['sample_rate']} Hz" for d in devices]
    selected=st.selectbox("Local microphone",labels,index=0)
    device_index=None if selected=="System Default" else int(selected.split(" · ",1)[0])
    diagnostics=st.toggle("Developer Diagnostics",False)
    config.debug_performance=st.toggle("Performance diagnostics",config.debug_performance)
    current_snap=state.snapshot()
    auto_failed=current_snap.vision.fretboard.failures >= 30
    manual_mode=st.session_state.get("manual_calibration",False)
    if auto_failed or manual_mode:
        st.subheader("Manual calibration fallback")
        st.caption("Use only when automatic fretboard detection cannot find the first four frets.")
        st.session_state.manual_calibration=st.toggle("Manual calibration",True,key="manual_calibration_toggle")
        if st.session_state.manual_calibration:
            names=("Nut upper","Fret 4 upper","Fret 4 lower","Nut lower"); adjusted=[]
            with st.expander("Four fretboard corners",expanded=not config.calibration.calibrated):
                for i,(name,(px,py)) in enumerate(zip(names,config.calibration.corners)):
                    a,b=st.columns(2); adjusted.append((a.slider(f"{name} X",0.,1.,float(px),.01,key=f"cx{i}"),b.slider(f"{name} Y",0.,1.,float(py),.01,key=f"cy{i}")))
            try: pending=FretboardCalibration(tuple(adjusted),config.calibration.calibrated).validate(); calibration_error=None
            except ValueError as exc: pending=config.calibration; calibration_error=str(exc)
            config.calibration=pending
            if calibration_error: st.error(calibration_error)
            c1,c2=st.columns(2)
            if c1.button("Confirm & Save",use_container_width=True,disabled=bool(calibration_error)):
                config.calibration=FretboardCalibration(pending.corners,True);save_fretboard_calibration(config.calibration);st.toast("Fretboard calibration saved");st.rerun()
            if c2.button("Reset Calibration",use_container_width=True):
                config.calibration=FretboardCalibration();save_fretboard_calibration(config.calibration)
                for i in range(4): st.session_state.pop(f"cx{i}",None);st.session_state.pop(f"cy{i}",None)
                st.rerun()
    else:
        st.caption("Automatic fretboard calibration is active.")
    st.caption("Status: "+(current_snap.vision.fretboard.message if state.running else "Start the tutor to detect the guitar."))
    config.use_auto_tuning=st.toggle("Auto tuning calibration",config.use_auto_tuning)
    if st.button("Reset tuning calibration",use_container_width=True): mic.detector.tuning_offset=0.

start_col,stop_col,_=st.columns([1,1,5])
if start_col.button("Start Tutor",type="primary",use_container_width=True,disabled=state.running):
    try:
        vision.start(); mic.start(device_index); state.set_running(True); fusion.start()
    except Exception as exc:
        vision.stop(); mic.stop(); state.set_running(False); st.error(f"Could not start local hardware: {exc}")
    st.rerun()
if stop_col.button("Stop Tutor",use_container_width=True,disabled=not state.running):
    vision.stop(); mic.stop(); fusion.stop(); state.stop(); st.rerun()

snap=state.snapshot(); indicators(snap)
left,right=st.columns([1.65,1],gap="large")
with left:
    st.caption("LIVE FRET-HAND VIEW")
    frame_placeholder=st.empty()
    if state.running:
        current_frame=state.snapshot().latest_frame
        if current_frame is not None:
            frame_placeholder.image(current_frame,channels="BGR",use_container_width=True)
        else:
            frame_placeholder.warning(camera.error or "Camera is running but has not delivered a frame yet. Check camera permissions and the selected device.")
    else:
        if camera.error:
            st.error(f"Camera unavailable: {camera.error}")
        else:
            st.info("Press **Start Tutor** to open the local Mac camera and microphone.")

with right:
    chord=get_chord(target_name)
    st.subheader(f"Target Chord · {chord.display_name}")
    st.markdown(f'<div class="diagram">Strings 6→1<br>{chord.ascii_diagram()}<br>Tones: {", ".join(chord.pitch_classes)}</div>',unsafe_allow_html=True)
    def render_live_panel():
        current=state.snapshot()
        if current.latest_frame is not None:
            frame_placeholder.image(current.latest_frame,channels="BGR",use_container_width=True)
        elif current.running:
            frame_placeholder.warning(camera.error or "Camera is running but has not delivered a frame yet. Check camera permissions and the selected device.")
        session_stats=core["stats"]
        if current.evaluation.attempt_event>session_stats["last_attempt"]:
            session_stats["attempts"]+=current.evaluation.attempt_event-session_stats["last_attempt"]
            session_stats["last_attempt"]=current.evaluation.attempt_event
            if current.evaluation.success_event==session_stats["last_success"]: session_stats["streak"]=0
        if current.evaluation.success_event>session_stats["last_success"]:
            session_stats["successes"]+=current.evaluation.success_event-session_stats["last_success"]
            session_stats["last_success"]=current.evaluation.success_event
            session_stats["streak"]+=1; session_stats["best"]=max(session_stats["best"],session_stats["streak"])
        validation_card(current.evaluation); metrics(current); audio_charts(current.audio,chord)
        feedback=build_feedback(chord,current.vision,current.audio)
        st.caption("Tutor feedback");st.write(feedback.headline)
        if feedback.finger_rows: st.dataframe(feedback.finger_rows,use_container_width=True,hide_index=True)
        if feedback.tone_rows: st.dataframe(feedback.tone_rows,use_container_width=True,hide_index=True)
        accuracy=100*session_stats["successes"]/max(1,session_stats["attempts"])
        st.caption(f"Attempts {session_stats['attempts']} · Successful {session_stats['successes']} · Accuracy {accuracy:.0f}% · Streak {session_stats['streak']} · Best {session_stats['best']}")
        if diagnostics:
          with st.expander("Developer Diagnostics",expanded=True):
             st.json({"guitar":{"detected":current.vision.guitar.guitar_detected,"confidence":current.vision.guitar.confidence,
               "bbox":current.vision.guitar.bbox,"center":current.vision.guitar.center,"angle":current.vision.guitar.angle,
               "body_region":current.vision.guitar.body_region,"neck_region":current.vision.guitar.neck_region,
               "stable_frames":current.vision.guitar.stable_frames,"lost_frames":current.vision.guitar.lost_frames,
               "detector_available":current.vision.guitar.detector_available,"error":current.vision.guitar.error},
               "fretboard":{"status":current.vision.fretboard.status,"confidence":current.vision.fretboard.calibration_confidence,
               "polygon":current.vision.fretboard.polygon,"center":current.vision.fretboard.center,"angle":current.vision.fretboard.angle,
               "length":current.vision.fretboard.length,"width":current.vision.fretboard.width,"nut":current.vision.fretboard.nut,
               "frets":current.vision.fretboard.frets,"stable_frames":current.vision.fretboard.stable_frames,
               "failures":current.vision.fretboard.failures,"message":current.vision.fretboard.message},
               "vision":{"handedness":current.vision.handedness,"assignments":current.vision.assignments,
              "hand_count":current.vision.hand_count,"all_handedness":current.vision.all_handedness,
              "raw_landmarks":current.vision.landmarks,"joint_angles":current.vision.joint_angles,
              "confidence_components":current.vision.confidence_components,
              "cell_error":round(current.vision.average_cell_error,3),"stable_frames":current.vision.stable_frames,
              "fps":round(current.vision.fps,1),"latency_ms":round(current.vision.latency_ms,1),"error":current.vision.error},
              "audio":{"raw":current.audio.raw_chord,"stable":current.audio.stable_chord,
              "target_similarity":round(current.audio.target_similarity,3),"best_similarity":round(current.audio.best_similarity,3),
              "margin":round(current.audio.margin,3),"tone_coverage":round(current.audio.expected_tone_coverage,3),
              "similarities":current.audio.similarities,"chroma":current.audio.chroma.tolist(),
              "rms":current.audio.rms,"sample_rate":current.audio.sample_rate,"queue_fill":current.audio.queue_fill,
              "spectral_flux":current.audio.spectral_flux,"onset_state":current.audio.onset_state,
              "strum_timestamp":current.audio.strum_timestamp,"tuning_offset_cents":current.audio.tuning_offset_cents,
              "peaks_hz":[round(p[0],1) for p in current.audio.peaks],"latency_ms":round(current.audio.latency_ms,1),
              "overflows":mic.overflows,"error":current.audio.error},
              "fusion":{"vision_timestamp":current.vision.timestamp,"audio_timestamp":current.audio.timestamp,
              "evaluation_timestamp":current.evaluation.timestamp,"timeline":event_timeline(current)},
               "mediapipe_backend":"Tasks API primary; legacy only if available"})
        if config.debug_performance:
            st.caption("Performance")
            st.json({**current.performance, "yolo_device": getattr(vision.processor.guitar_detector, "device", "unknown"),
                "camera_ready": camera.is_running(), "microphone_active": current.audio_active})
        if current.evaluation.state is EvaluationState.PERFECT and current.evaluation.success_event>core["last_toast"]:
            core["last_toast"]=current.evaluation.success_event; st.toast("🎸 CHORD PERFECT!",icon="✅"); st.balloons()
    if state.running and hasattr(st,"fragment"):
        st.fragment(run_every=.25)(render_live_panel)()
    else:
        render_live_panel()
