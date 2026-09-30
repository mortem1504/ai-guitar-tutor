# AI Guitar Chord Tutor

A local, real-time guitar coach with four independent engines: **EYE** maps the fretting hand through a calibrated perspective plane, **EAR** recognizes onset-gated browser microphone audio, **BRAIN** associates the stable hand state preceding each strum with its post-onset audio, and **TUTOR** explains measured finger and chord-tone problems. Success requires both modalities.

## What is implemented

- MediaPipe Tasks Hand Landmarker with legacy `solutions.hands` fallback and detailed compatibility errors.
- Separate WebRTC webcam and microphone processors. Browser media works even when Streamlit is hosted elsewhere; `sounddevice` remains an opt-in native fallback.
- A lightweight local guitar-trained Ultralytics `guitar-yolo11n.pt` detector recognizes its explicit guitar class with a cached, asynchronous inference worker. Standard COCO models do not contain a guitar category and are rejected. Guitar presence is stabilized over multiple frames, with a short loss grace period, bounding box, body/neck guidance regions, confidence, and orientation.
- Guitar-scoped OpenCV Hough/edge analysis finds paired neck edges, estimates a rotated fretboard quadrilateral, warps it with a perspective transform, checks candidate fret-wire peaks against progressive fret spacing, and smooths the result through a calibration state machine. The EYE tracks up to two hands with 21 landmarks each, draws both skeletons, and automatically selects the hand with the strongest detected-fretboard overlap as the fretting hand.
- The audio callback only copies into a bounded queue. A separate DSP thread performs DC/high-pass filtering, RMS gating, strum onset detection from energy slope plus spectral flux, overlapping 8192-point FFT analysis, harmonic-aware compressed chroma, chord-template comparison, tuning compensation, smoothing, and voting.
- Thread-safe shared state and a separate fusion thread. Observations expire after one second and success events are edge-triggered with a cooldown.
- E Major, A Minor, C Major, and two common G Major fingerings.
- A dark Streamlit dashboard, device selection, waveform/chromagram, calibration persistence, metrics, and diagnostics.
- Session-only training metrics for meaningful strum attempts, successful chords, accuracy, current streak, and best streak.
- Named Python loggers with concise UI errors and terminal tracebacks. Per-frame success is never logged.

## Install on Apple Silicon / Python 3.13

Requires macOS 11+ on arm64. The pinned MediaPipe 0.10.35 build supplies a native macOS arm64 wheel, so Rosetta is not required. MediaPipe 1.0.1 was deliberately not selected: its Tasks graph aborts during initialization on the tested M1 host, even with the CPU delegate; 0.10.35 exposes the modern Tasks API and initializes cleanly.

```bash
cd ai-guitar-tutor
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
python scripts/diagnose.py
streamlit run app.py
```

The model downloader uses the official MediaPipe Hand Landmarker `float16/1` asset, validates that the response is a plausible model, writes atomically, and never redownloads a valid local copy. Once present, it works offline.

## Using the tutor

1. Select a target and microphone in the sidebar.
2. Press **Start Tutor** and allow browser camera and microphone access. The camera first searches for a guitar, then locates and stabilizes the first four frets automatically. macOS may ask for browser permission. Terminal/Python microphone access is needed only for the native fallback.
4. Form the outlined shape, hold it steadily, then strum. A perfect event occurs only when the stable shape and stable acoustic chord agree.
5. Press **Stop Tutor** to close PortAudio and stop the fusion worker; stopping/removing the WebRTC component releases the browser camera track.

Supported lessons are E Major, A Minor, C Major, and G Major. G Major accepts both the common three-finger and four-finger beginner shapes. Enable **Developer Diagnostics** for raw landmarks, joint angles, grid assignments, score components, FFT peaks, chroma, every template similarity, sample rate, queue occupancy, latencies, and fusion timestamps.

The homography maps each fingertip into normalized neck coordinates before assigning a string/fret. `FretboardMapper.point_to_fretboard(x, y)` exposes normalized coordinates and a future-compatible `string: None` field. Display mirroring is separate from physical left/right-handed mapping. Perspective registration improves estimates but is not millimeter-accurate.

## macOS permissions and troubleshooting

- In **System Settings → Privacy & Security → Camera**, enable the browser used to open Streamlit.
- In **Privacy & Security → Microphone**, enable the terminal or Python host that launches Streamlit. Restart that app after changing permission.
- Safari can require per-site camera permission; Chrome shows it beside the address bar.
- Run `python scripts/diagnose.py` to report architecture, package versions, MediaPipe API availability, model status, PortAudio inputs, and camera indexes.
- If MediaPipe fails, use the pinned requirements and rerun `python scripts/download_models.py`; the in-app error includes Python, architecture, macOS, MediaPipe version, and API availability.
- If the mic is silent, choose the exact input in the sidebar, check macOS input level, and ensure another exclusive audio app is not holding it.
- `sounddevice` wheels normally provide what this app needs. If PortAudio cannot load on an unusual local Python distribution, install PortAudio using that distribution's documented platform procedure, then reinstall `sounddevice`; Homebrew is a recovery option, not a mandatory prerequisite.
- Camera and microphone access are local Python resources: OpenCV uses macOS AVFoundation (with a generic OpenCV fallback), and sounddevice uses a persistent PortAudio input stream. The browser only displays the processed camera frame and tutor UI.

## Tests

```bash
pytest -q
```

Tests cover chord knowledge/templates, synthetic polyphonic chroma, fret-grid geometry, joint angles, freshness, and the rule that one modality alone never produces success.

## Architecture

`guitar_tutor/vision`, `audio`, and `fusion` contain the EYE, EAR, and BRAIN respectively. `SharedState` is the only exchange point. Locks cover short snapshot/replacement operations only; MediaPipe, FFT, OpenCV rendering, and Streamlit rendering happen outside them. Buffers and histories are bounded.

- **EYE:** the WebRTC receiver timestamps frames, invokes MediaPipe, smooths 21 landmarks, calculates joint geometry and calibrated grid positions, scores alternative visual templates, annotates a frame, then publishes one immutable-style result.
- **EAR:** browser WebRTC PCM is primary. Its callback copies mono samples into a bounded deque and returns; a worker uses the frame's real sample rate for onset, FFT/chroma/template, expected-tone and tuning analysis. Native PortAudio follows the same DSP path when explicitly enabled.
- **BRAIN:** keeps a bounded vision history. Each debounced onset captures a strum timestamp, selects a stable vision observation from the preceding 0–1000 ms, and accepts audio only from the following 0–400 ms.
- **TUTOR:** compares per-finger calibrated assignments and per-pitch-class energy. It reports only evidence-backed missing/wrong fingers, weak chord tones, and tuning offsets; it does not claim exact audio string isolation.

## Limitations and future improvements

The calibrated overlay assumes the neck stays aligned; it does not detect frets or strings directly. Lighting, occlusion, unusual camera angles, amplification effects, room noise, and alternate voicings can reduce confidence. Chroma identifies harmonic content but cannot guarantee that every individual string rang cleanly. Future work could add automatic neck registration, per-string onset/separation, tuning calibration, more chord inversions, adaptive thresholds, lesson sequencing, and opt-in local recording/replay. No audio or video is written to disk in this version.

## Inspiration and attribution

The implementation is original Python code. Architectural inspiration came from [agneya-1402/AirGuitar](https://github.com/agneya-1402/AirGuitar) for hand tracking, [LucasSoaresRibeiro/chrome-extension-chord-detector](https://github.com/LucasSoaresRibeiro/chrome-extension-chord-detector) for browser-era chroma/template concepts, and [Guitariz/Guitariz](https://github.com/Guitariz/Guitariz) for separation of music-analysis structures. No source was copied.
