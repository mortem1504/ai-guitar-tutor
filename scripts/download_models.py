#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import ssl
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"assets"/"models"/"hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 5_000_000


def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if valid(DEST):
        print(f"Model ready: {DEST} ({DEST.stat().st_size:,} bytes)"); return
    temp=DEST.with_suffix(".task.part")
    try:
        print(f"Downloading official MediaPipe model to {DEST} ...")
        try:
            import certifi
            ssl_context=ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_context=ssl.create_default_context()
        with urllib.request.urlopen(URL,timeout=60,context=ssl_context) as response, temp.open("wb") as out:
            while block:=response.read(1024*1024): out.write(block)
        if not valid(temp): raise RuntimeError(f"download is unexpectedly small ({temp.stat().st_size} bytes)")
        temp.replace(DEST)
        digest=hashlib.sha256(DEST.read_bytes()).hexdigest()
        print(f"Verified size and SHA-256: {digest}")
    except Exception as exc:
        temp.unlink(missing_ok=True)
        raise SystemExit(f"Model download failed: {exc}\nConnect to the internet and retry. Existing valid models are always usable offline.")

if __name__=="__main__": main()
