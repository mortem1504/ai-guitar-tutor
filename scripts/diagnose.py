#!/usr/bin/env python3
from __future__ import annotations
import importlib, platform, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def version(name):
    try:
        module=importlib.import_module(name); return getattr(module,"__version__","installed")
    except Exception as exc: return f"ERROR: {exc}"


print(f"Python: {sys.version}")
print(f"Architecture: {platform.machine()} (ARM native: {platform.machine() in ('arm64','aarch64')})")
print(f"macOS: {platform.mac_ver()[0]}")
for name in ("mediapipe","cv2","numpy","sounddevice","streamlit","ultralytics","torch"):
    print(f"{name}: {version(name)}")
try:
    from guitar_tutor.vision.mediapipe_adapter import capability_report
    print(f"MediaPipe APIs: {capability_report()}")
except Exception as exc: print(f"MediaPipe inspection: ERROR: {exc}")
model=ROOT/"assets/models/hand_landmarker.task"
print(f"Model: {'READY' if model.exists() and model.stat().st_size>5_000_000 else 'MISSING'} · {model}")
print("I/O: local OpenCV camera + sounddevice microphone; browser permissions are not used")
try:
    import sounddevice as sd
    print("PortAudio devices:")
    for i,d in enumerate(sd.query_devices()):
        if d["max_input_channels"] or d["max_output_channels"]:
            print(f"  {i}: {d['name']} · inputs={d['max_input_channels']} outputs={d['max_output_channels']} rate={d['default_samplerate']}")
except Exception as exc: print(f"PortAudio: ERROR: {exc}")
try:
    import cv2
    print("Camera probe (indexes 0–4):")
    for i in range(5):
        cap=cv2.VideoCapture(i); ok=cap.isOpened()
        if ok: print(f"  {i}: available")
        cap.release()
except Exception as exc: print(f"Camera probe: ERROR: {exc}")
ready=(sys.version_info>=(3,13) and platform.machine() in ("arm64","aarch64") and model.exists())
print(f"System readiness: {'READY' if ready else 'CHECK ITEMS ABOVE'}")
