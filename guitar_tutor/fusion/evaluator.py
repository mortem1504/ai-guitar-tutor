from __future__ import annotations

from threading import Event, Thread
import time

from guitar_tutor.config import TutorConfig
from guitar_tutor.state import Evaluation, EvaluationState, SharedState
from guitar_tutor.fusion.event_sync import EventSynchronizer


class Evaluator:
    def __init__(self, config: TutorConfig):
        self.config=config; self._last_success=-1e9; self._success_event=0; self._was_perfect=False
        self._attempt_event=0; self._last_attempt=-1e9; self._previous_audio="—"

    def evaluate(self,target,vision,audio,now=None,calibrated=True,strum_event=False) -> Evaluation:
        now=time.monotonic() if now is None else now
        vf=now-vision.timestamp <= self.config.fusion_freshness_seconds
        af=now-audio.timestamp <= self.config.fusion_freshness_seconds
        guitar_tracking_active = bool(vision.guitar.stable_frames or vision.guitar.lost_frames or vision.guitar.error)
        if guitar_tracking_active and not vision.guitar.guitar_detected:
            state,msg=EvaluationState.CALIBRATION_REQUIRED,"Hold your guitar inside the camera view."
            return Evaluation(state,msg,0.,audio.confidence,0.,target,audio.stable_chord,vision.detected_shape,now,
                self._success_event,self._attempt_event)
        vm=vf and vision.match; am=af and audio.match and (strum_event or audio.strum_timestamp==0)
        combined=.52*vision.confidence+.48*audio.confidence if vf and af else 0.
        if not calibrated:
            state,msg=EvaluationState.CALIBRATION_REQUIRED,"Calibrate the visible first four frets before starting the lesson."
        elif vision.error or audio.error:
            state,msg=EvaluationState.SYSTEM_ERROR,vision.error or audio.error or "Subsystem error"
        elif not vf or not vision.hand_detected:
            state,msg=EvaluationState.NO_HAND,"Place your fretting hand inside the Fret Box."
        elif not vision.inside_fret_box:
            state,msg=EvaluationState.HAND_OUTSIDE_FRETBOARD,"Align your fretting hand with the calibrated fretboard."
        elif vm and (not af or audio.dbfs < self.config.silence_dbfs):
            state,msg=EvaluationState.SHAPE_CORRECT_WAITING_FOR_STRUM,"Shape looks good — strum the chord."
        elif vm and am:
            state,msg=EvaluationState.PERFECT,"CHORD PERFECT!"
        elif vm:
            state,msg=EvaluationState.VISION_ONLY_CORRECT,"Fingers aligned! Strike your strings cleanly or tune your guitar."
        elif am:
            state,msg=EvaluationState.AUDIO_ONLY_CORRECT,"Audio pattern recognized, but optimize your hand placement to avoid bad habits."
        elif af and audio.dbfs < self.config.silence_dbfs:
            state,msg=EvaluationState.LOW_SIGNAL,"Adjust your fingering, then strum clearly."
        elif af and audio.stable_chord not in ("—",target):
            state,msg=EvaluationState.AUDIO_MISMATCH,f"I hear {audio.stable_chord}; the target is {target}."
        else:
            state,msg=EvaluationState.VISION_MISMATCH,"Move each fingertip toward its outlined target."
        perfect=state is EvaluationState.PERFECT
        meaningful=strum_event and af and audio.dbfs>=self.config.silence_dbfs
        if meaningful and now-self._last_attempt>=.5:
            self._attempt_event+=1; self._last_attempt=now
        self._previous_audio=audio.stable_chord if meaningful else "—"
        if perfect and not self._was_perfect and now-self._last_success >= self.config.success_cooldown_seconds:
            self._success_event+=1; self._last_success=now
        self._was_perfect=perfect
        return Evaluation(state,msg,vision.confidence,audio.confidence,float(combined),target,
            audio.stable_chord,vision.detected_shape,now,self._success_event,self._attempt_event)


class FusionEngine:
    def __init__(self,state: SharedState,config: TutorConfig):
        self.state,self.config,self.evaluator=state,config,Evaluator(config); self.sync=EventSynchronizer();self.stop_event=Event(); self.thread=None;self.event_vision=None
    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear(); self.thread=Thread(target=self._run,name="guitar-fusion",daemon=True); self.thread.start()
    def _run(self):
        while not self.stop_event.wait(.06):
            snap=self.state.snapshot();self.sync.add_vision(snap.vision);attempt=self.sync.synchronize(snap.audio)
            if attempt:self.event_vision=attempt.vision
            guitar_tracking_active=bool(snap.vision.guitar.stable_frames or snap.vision.guitar.lost_frames or snap.vision.guitar.error)
            guitar_ready=not guitar_tracking_active or snap.vision.guitar.guitar_detected
            fretboard_ready=snap.vision.fretboard.status in ("CALIBRATED", "TRACKING")
            event_active=bool(snap.audio.strum_timestamp and time.monotonic()-snap.audio.strum_timestamp<=.4
                and self.event_vision and guitar_ready and (not guitar_tracking_active or fretboard_ready))
            vision=self.event_vision if event_active else snap.vision
            live_calibrated = self.config.calibration.calibrated
            if snap.vision.guitar.stable_frames or snap.vision.guitar.lost_frames:
                live_calibrated = snap.vision.fretboard.status in ("CALIBRATED", "TRACKING")
            self.state.update_evaluation(self.evaluator.evaluate(self.state.target,vision,snap.audio,
                calibrated=live_calibrated,strum_event=event_active))
    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=1)
        self.thread=None
