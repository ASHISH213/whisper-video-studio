#!/usr/bin/env python3
"""
colab_studio.py - All-in-One Video2Audio & Whisper Transcription Studio
Built for Google Colab (GPU A100/L4/T4) and Local Workstations.

Combines:
  1. Fast FFmpeg Video to Audio Extraction (Speech MP3, WAV, AAC, FLAC, Opus)
  2. GPU-Accelerated faster-whisper Batch Transcriber (large-v3, distil-large-v3, etc.)
  3. End-to-End One-Click Pipeline (Video -> Audio -> .txt / .srt / .md Study Notes)
  4. Built-in Transcript Viewer & Markdown Study Notes Reader

Can be run:
  - Inside Google Colab (embedded in cell iframe or popped out in a window)
  - Locally via 'python colab_studio.py'
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import html
import json
import logging
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# --------------------------------------------------------------------------- Environment & Hardware
def is_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")

def get_gpu_info() -> dict:
    info = {"available": False, "name": "None (CPU Mode)", "total_mb": 0, "free_mb": 0}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                info = {
                    "available": True,
                    "name": parts[0],
                    "total_mb": int(float(parts[1])),
                    "free_mb": int(float(parts[2]))
                }
    except Exception:
        pass
    return info

def is_drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir()

def try_mount_drive() -> bool:
    if not is_colab():
        return False
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        return is_drive_mounted()
    except Exception as e:
        log(f"Drive mount error: {e}")
        return False

# --------------------------------------------------------------------------- Config & Presets
VIDEO_EXT = {
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".m4v", ".ts",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".3gp", ".3g2", ".ogv", ".vob",
    ".rm", ".rmvb", ".asf", ".divx", ".f4v", ".m2v", ".mpv", ".qt", ".tod", ".mod"
}
AUDIO_EXT = {
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".aiff", ".aif", ".aifc", ".alac", ".mka", ".ac3", ".dts", ".amr",
    ".caf", ".mp2", ".m4b", ".m4p", ".weba"
}
ALL_MEDIA_EXT = VIDEO_EXT | AUDIO_EXT

CLOUD_MASK = 0x400000 | 0x40000 | 0x1000
MAX_ITEMS = 5000
PROGRESS_KEYS = {"frame", "fps", "bitrate", "total_size", "out_time_us", "out_time_ms", "out_time",
                 "dup_frames", "drop_frames", "speed", "progress"}

AUDIO_PRESETS = [
    dict(id="1", title="MP3 \u00b7 Speech", ext="mp3", fmt="mp3", kbps=32,
         desc="Mono, 16 kHz, 32 kbps. Tiny files, optimal for faster-whisper.",
         tag="Best for Whisper", args=["-ac", "1", "-ar", "16000", "-b:a", "32k"]),
    dict(id="2", title="MP3 \u00b7 Music / High", ext="mp3", fmt="mp3", kbps=192,
         desc="Stereo, 192 kbps. High fidelity general purpose audio.", tag="",
         args=["-b:a", "192k"]),
    dict(id="3", title="M4A \u00b7 AAC", ext="m4a", fmt="ipod", kbps=128,
         desc="Stereo, 128 kbps. Universal playback on Apple & mobile devices.", tag="",
         args=["-c:a", "aac", "-b:a", "128k"]),
    dict(id="4", title="WAV \u00b7 Uncompressed", ext="wav", fmt="wav", kbps=256,
         desc="Mono, 16 kHz, 16-bit PCM. Fast processing, larger disk footprint.", tag="",
         args=["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]),
    dict(id="5", title="FLAC \u00b7 Lossless", ext="flac", fmt="flac", kbps=None,
         desc="Studio-grade master quality, bit-perfect compression.", tag="Lossless",
         args=["-c:a", "flac"]),
    dict(id="6", title="Opus \u00b7 Ultra-Compact", ext="opus", fmt="ogg", kbps=24,
         desc="Mono, 24 kbps. State-of-the-art voice compression.", tag="Smallest",
         args=["-ac", "1", "-c:a", "libopus", "-b:a", "24k"]),
]

WHISPER_MODELS = [
    dict(id="large-v3", name="large-v3", desc="Maximum accuracy, best punctuation & technical jargon", badge="Highest Accuracy", vram="~6 GB"),
    dict(id="distil-large-v3", name="distil-large-v3", desc="Distilled Large V3. 6x faster inference with minimal accuracy loss", badge="Recommended / 6x Speed", vram="~3 GB"),
    dict(id="large-v2", name="large-v2", desc="Previous generation large model, well tested", badge="Stable", vram="~6 GB"),
    dict(id="medium", name="medium", desc="Balanced speed and precision, great for English/code", badge="Balanced", vram="~3 GB"),
    dict(id="small", name="small", desc="Fast transcription for resource-constrained runtimes", badge="Fast", vram="~1.5 GB"),
    dict(id="base", name="base", desc="Ultra fast preview model", badge="Quick Preview", vram="~1 GB"),
]

DEFAULT_HOTWORDS = (
    "Naresh IT, Veera Babu, Kubernetes, kubectl, Terraform, AWS, EC2, S3, IAM, VPC, "
    "Docker, Jenkins, Helm, CI/CD, GitHub Actions, Ansible, Prometheus, Grafana, "
    "Azure, GCP, YAML, DevOps, SRE"
)

FFMPEG = shutil.which("ffmpeg") or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else None)
FFPROBE = shutil.which("ffprobe") or ("/usr/bin/ffprobe" if os.path.exists("/usr/bin/ffprobe") else None)
TOKEN = secrets.token_urlsafe(18)
LOCK = threading.Lock()
SERVER = None

# Global runtime state
SCAN = {"root": None, "mode": "video", "items": []}
PLAN = {"id": None, "state": "idle", "done": 0, "total": 0, "result": None, "error": None,
        "jobs": [], "preset": None, "out_dir": None}
RUN = {
    "state": "idle",           # idle | running | cancelled | finished
    "mode": "pipeline",        # pipeline | v2a | whisper
    "items": [],
    "log": [],
    "out_dir": "",
    "audio_dir": "",
    "transcripts_dir": "",
    "started": 0.0,
    "ended": 0.0,
    "cancel": False,
    "proc": None,              # ffmpeg subprocess if converting
    "current": None,           # current item id
    "stage": "",               # "Converting Video" | "Transcribing Audio" | ""
    "live_segment": None,      # {"ts": "[00:01:23]", "text": "..."}
    "whisper_model_name": "",
    "speed": "",
}

class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg

def log(msg: str):
    with LOCK:
        ts_str = f"{datetime.now():%H:%M:%S}"
        entry = f"{ts_str}  {msg}"
        RUN["log"].append(entry)
        del RUN["log"][:-400]
        print(f"[{ts_str}] {msg}", flush=True)

# --------------------------------------------------------------------------- Numbering & Natural Sort
KEYWORD_RE = re.compile(
    r"\b(?:session|day|class|lecture|lesson|part|episode|ep|video|module|chapter|week)"
    r"[\s._#-]*(\d+)", re.I
)

def file_number(stem: str) -> int | None:
    m = KEYWORD_RE.search(stem)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else None

def natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def item_key(it):
    return (it["num"] is None, it["num"] or 0, natural(os.path.join(it["rel"], it["name"])))

def ts(sec: float, srt: bool = False) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if srt:
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

# --------------------------------------------------------------------------- Media Helpers
def probe(path: str | Path):
    """Returns (duration_seconds | None, has_audio | None)"""
    if not FFPROBE:
        return None, None
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60
        )
        data = json.loads(r.stdout or "{}")
        dur = float(data.get("format", {}).get("duration") or 0) or None
        has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
        return dur, has_audio
    except Exception:
        return None, None

def ffmpeg_version_str() -> str:
    if not FFMPEG:
        return "ffmpeg not found"
    try:
        out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0].split(" Copyright")[0]
    except Exception:
        return "ffmpeg"

def places() -> list[dict]:
    p = []
    if is_colab():
        cands = [
            ("Colab MyDrive", Path("/content/drive/MyDrive")),
            ("Transcribe Folder", Path("/content/drive/MyDrive/transcribe")),
            ("Colab /content", Path("/content")),
            ("Drive Videos", Path("/content/drive/MyDrive/transcribe/videos")),
            ("Drive Audio", Path("/content/drive/MyDrive/transcribe/audio")),
            ("Drive Transcripts", Path("/content/drive/MyDrive/transcribe/transcripts_large_v3")),
        ]
    else:
        home = Path.home()
        cands = [
            ("Home", home),
            ("Desktop", home / "Desktop"),
            ("Desktop (OneDrive)", home / "OneDrive" / "Desktop"),
            ("Videos", home / "Videos"),
            ("Downloads", home / "Downloads"),
            ("Current Folder", Path.cwd()),
        ]
    seen = set()
    for name, path in cands:
        try:
            if path.is_dir() and str(path) not in seen:
                seen.add(str(path))
                p.append({"name": name, "path": str(path)})
        except OSError:
            pass
    return p

def roots() -> list[dict]:
    if os.name == "nt":
        return [{"name": f"{c}:\\", "path": f"{c}:\\"} for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    return [{"name": "/", "path": "/"}]

def preset_by_id(pid: str | int | None) -> dict | None:
    return next((p for p in AUDIO_PRESETS if str(p["id"]) == str(pid)), None)

# --------------------------------------------------------------------------- Faster-Whisper Lazy Loader
WHISPER_CACHE = {}

def get_whisper_model(model_name="large-v3", device=None, compute_type="float16"):
    global WHISPER_CACHE
    from faster_whisper import WhisperModel
    if device is None:
        gpu = get_gpu_info()
        device = "cuda" if gpu["available"] else "cpu"
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
    key = (model_name, device, compute_type)
    if key not in WHISPER_CACHE:
        log(f"Loading WhisperModel: {model_name} on {device} ({compute_type})...")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        WHISPER_CACHE[key] = model
        log(f"WhisperModel {model_name} loaded successfully!")
    return WHISPER_CACHE[key]

# --------------------------------------------------------------------------- API Implementations
def api_info(_q):
    gpu = get_gpu_info()
    fw_available = False
    try:
        import faster_whisper
        fw_available = True
    except ImportError:
        pass

    return {
        "ffmpeg": ffmpeg_version_str(),
        "faster_whisper": fw_available,
        "gpu": gpu,
        "is_colab": is_colab(),
        "drive_mounted": is_drive_mounted(),
        "audio_presets": [{k: v for k, v in p.items() if k not in ("args", "fmt")} for p in AUDIO_PRESETS],
        "whisper_models": WHISPER_MODELS,
        "default_hotwords": DEFAULT_HOTWORDS,
        "home": str(Path.home()),
        "cwd": str(Path.cwd()),
        "sep": os.sep,
        "places": places(),
        "scan_items": SCAN["items"] if SCAN["items"] else [],
        "scan_root": SCAN["root"] or "",
        "plan": PLAN["result"] if PLAN.get("result") else None,
    }

def api_mount_drive(_body):
    success = try_mount_drive()
    return {"mounted": success, "path": "/content/drive/MyDrive"}

def api_fs(q):
    raw = (q.get("path") or [""])[0]
    if not raw:
        return {"path": "", "parent": None, "dirs": roots(), "videos": 0, "audio": 0, "places": places()}
    d = Path(raw).expanduser()
    if not d.is_dir():
        raise ApiError(404, "Directory does not exist")
    dirs, vids, auds = [], 0, 0
    try:
        with os.scandir(d) as it:
            for e in it:
                try:
                    if e.name.startswith("."):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        dirs.append({"name": e.name, "path": os.path.join(str(d), e.name)})
                    else:
                        ext = os.path.splitext(e.name)[1].lower()
                        if ext in VIDEO_EXT:
                            vids += 1
                        elif ext in AUDIO_EXT:
                            auds += 1
                except OSError:
                    pass
    except PermissionError:
        raise ApiError(403, "Permission denied")
    dirs.sort(key=lambda x: natural(x["name"]))
    parent = str(d.parent) if d.parent != d else ("" if os.name == "nt" else None)
    return {"path": str(d), "parent": parent, "dirs": dirs, "videos": vids, "audio": auds}

def api_scan(body):
    d = Path(str(body.get("path", ""))).expanduser()
    if not d.is_dir():
        raise ApiError(404, "Directory does not exist")
    recursive = bool(body.get("recursive"))
    mode = str(body.get("scan_mode", "all"))  # "all" or "video" or "audio"
    valid_exts = VIDEO_EXT if mode == "video" else (AUDIO_EXT if mode == "audio" else ALL_MEDIA_EXT)
    
    items = []
    def add(dirpath, name):
        full = os.path.join(dirpath, name)
        st = os.stat(full)
        rel = os.path.relpath(dirpath, d)
        stem = os.path.splitext(name)[0]
        ext = os.path.splitext(name)[1].lower()
        is_aud = ext in AUDIO_EXT
        items.append(dict(
            path=full, name=name, rel="" if rel == "." else rel, size=st.st_size,
            num=file_number(stem),
            cloud=bool(getattr(st, "st_file_attributes", 0) & CLOUD_MASK),
            media_type="audio" if is_aud else "video",
            ext=ext.lstrip("."),
            duration=None, has_audio=None
        ))

    try:
        if recursive:
            for dirpath, dirnames, filenames in os.walk(d):
                dirnames[:] = [x for x in dirnames if not x.startswith(".")]
                for name in filenames:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in valid_exts and not name.startswith("._"):
                        try:
                            add(dirpath, name)
                        except OSError:
                            pass
                if len(items) >= MAX_ITEMS:
                    break
        else:
            for name in os.listdir(d):
                full = os.path.join(d, name)
                ext = os.path.splitext(name)[1].lower()
                if os.path.isfile(full) and ext in valid_exts and not name.startswith("._"):
                    try:
                        add(str(d), name)
                    except OSError:
                        pass
    except PermissionError:
        raise ApiError(403, "Permission denied reading folder")

    items.sort(key=item_key)
    items = items[:MAX_ITEMS]
    for i, it in enumerate(items):
        it["id"] = i

    with LOCK:
        SCAN["root"], SCAN["mode"], SCAN["items"] = str(d), mode, items
    return {
        "root": str(d),
        "total_size": sum(i["size"] for i in items),
        "counts": {
            "total": len(items),
            "video": sum(1 for i in items if i["media_type"] == "video"),
            "audio": sum(1 for i in items if i["media_type"] == "audio")
        },
        "items": [{k: it[k] for k in ("id", "name", "rel", "size", "num", "cloud", "media_type", "ext")} for it in items]
    }

# --------------------------------------------------------------------------- Dry Run / Planning
def build_plan_thread(plan_id, jobs, preset, out_dir, overwrite, do_probe, check_transcripts=False, transcripts_dir=None, mode="pipeline"):
    try:
        if do_probe:
            todo = [j for j in jobs if j["duration"] is None and j["has_audio"] is None]
            with LOCK:
                PLAN["total"], PLAN["done"] = len(todo), 0

            def work(j):
                j["duration"], j["has_audio"] = probe(j["path"])
                with LOCK:
                    PLAN["done"] += 1

            with ThreadPoolExecutor(max_workers=6) as ex:
                list(ex.map(work, todo))

        used, warnings = set(), []
        tr_dir = Path(transcripts_dir) if transcripts_dir else None
        
        for j in jobs:
            is_audio = j.get("media_type") == "audio"
            stem = os.path.splitext(j["name"])[0]
            
            # Check transcript completion if applicable
            has_txt = False
            if tr_dir and (tr_dir / f"{stem}.txt").exists():
                has_txt = True

            if is_audio:
                # Direct audio file: bypass extraction for transcription
                j["out"] = j["path"]
                if j["has_audio"] is False:
                    j["status"] = "noaudio"
                elif has_txt and not overwrite:
                    j["status"] = "done_transcript"
                elif mode in ("pipeline", "whisper"):
                    j["status"] = "direct"
                    j["note"] = "Direct transcribe (Audio file - no conversion needed)"
                else:
                    # In v2a mode, re-encode audio to preset format
                    out = out_dir / j["rel"] / (stem + "." + (preset["ext"] if preset else "mp3"))
                    j["out"] = str(out)
                    if out.exists() and out.stat().st_size > 0 and not overwrite:
                        j["status"] = "skip"
                    else:
                        j["status"] = "convert"
                        j["note"] = f"Re-encode to {preset['ext'].upper()}"
            else:
                # Video file: needs audio extraction
                out = out_dir / j["rel"] / (stem + "." + (preset["ext"] if preset else "mp3"))
                if str(out).lower() in used:
                    ext = os.path.splitext(j["name"])[1][1:].lower()
                    out = out.with_name(f"{stem}_{ext}.{preset['ext'] if preset else 'mp3'}")
                    j["note"] = "renamed (name collision)"
                used.add(str(out).lower())
                j["out"] = str(out)

                if j["has_audio"] is False:
                    j["status"] = "noaudio"
                elif has_txt and not overwrite:
                    j["status"] = "done_transcript"
                elif out.exists() and out.stat().st_size > 0 and not overwrite and mode == "pipeline":
                    # Video's audio was already extracted earlier, can directly transcribe
                    j["status"] = "direct"
                    j["note"] = "Audio already extracted - direct transcribe"
                elif out.exists() and out.stat().st_size > 0 and not overwrite and not tr_dir:
                    j["status"] = "skip"
                else:
                    j["status"] = "convert"
                    j["note"] = "Extract audio & transcribe" if mode == "pipeline" else "Extract audio"

        todo = [j for j in jobs if j["status"] in ("ready", "convert", "direct")]
        direct_cnt = sum(1 for j in jobs if j["status"] == "direct")
        convert_cnt = sum(1 for j in jobs if j["status"] == "convert")

        nums = sorted({j["num"] for j in jobs if j["num"] is not None})
        if len(nums) > 1:
            have = set(nums)
            gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in have]
            if gaps:
                warnings.append("No file found for session number(s): " + ", ".join(map(str, gaps[:12]))
                                + (" ..." if len(gaps) > 12 else ""))
        by_num = {}
        for j in jobs:
            if j["num"] is not None:
                by_num.setdefault(j["num"], []).append(j)
        for n, js in by_num.items():
            if len(js) > 1:
                warnings.append(f"{len(js)} files share the session/index number {n}")
        if any(j["status"] == "noaudio" for j in jobs):
            warnings.append("Some files have no audio track and will be skipped")
        if any(j.get("note") and "collision" in j.get("note") for j in jobs):
            warnings.append("Some output names were altered to resolve conflicts")
        if any(j.get("cloud") for j in todo):
            warnings.append("Some files are online-only (OneDrive). Processing will download them locally")

        total_dur = sum(j["duration"] or 0 for j in todo)
        convert_dur = sum(j["duration"] or 0 for j in todo if j["status"] == "convert")
        est = (preset["kbps"] * 1000 / 8 * convert_dur) if (preset and preset["kbps"] and convert_dur) else None
        
        probe_dir = out_dir
        while not probe_dir.exists() and probe_dir != probe_dir.parent:
            probe_dir = probe_dir.parent
        free = shutil.disk_usage(probe_dir).free
        if est and free < est * 1.2:
            warnings.append("Warning: Output destination has low remaining disk space")

        result = {
            "items": [dict({k: j.get(k) for k in ("id", "name", "rel", "size", "num", "cloud", "duration", "status", "note", "media_type", "ext")},
                           out=os.path.relpath(j["out"], out_dir) if j["out"] != j["path"] else j["name"]) for j in jobs],
            "warnings": warnings,
            "summary": {
                "todo": len(todo),
                "direct_audio": direct_cnt,
                "video_convert": convert_cnt,
                "skipped": sum(1 for j in jobs if j["status"] in ("skip", "done_transcript")),
                "noaudio": sum(1 for j in jobs if j["status"] == "noaudio"),
                "duration": total_dur, "est_bytes": est, "free_bytes": free,
                "out_dir": str(out_dir),
                "durations_known": all(j["duration"] for j in todo) if todo else True
            },
        }
        with LOCK:
            if PLAN["id"] == plan_id:
                PLAN.update(state="done", result=result, jobs=jobs, preset=preset, out_dir=str(out_dir))
    except Exception as e:
        with LOCK:
            if PLAN["id"] == plan_id:
                PLAN.update(state="error", error=f"{type(e).__name__}: {e}")

def api_plan(body):
    ids = set(body.get("ids") or [])
    if not ids:
        raise ApiError(400, "Please select at least one media file")
    pid = body.get("preset", "1")
    preset = preset_by_id(pid) or AUDIO_PRESETS[0]
    if not SCAN["root"]:
        raise ApiError(400, "Please scan a folder first")
    
    root = Path(SCAN["root"])
    out_dir = Path(str(body.get("out_dir") or "")).expanduser() if body.get("out_dir") else (
        Path("/content/drive/MyDrive/transcribe/audio") if is_colab() else root / "audio"
    )
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    transcripts_dir = Path(str(body.get("transcripts_dir") or "")).expanduser() if body.get("transcripts_dir") else None
    mode = str(body.get("mode", "pipeline"))

    jobs = [dict(it) for it in SCAN["items"] if it["id"] in ids]
    plan_id = secrets.token_hex(6)
    with LOCK:
        if RUN["state"] == "running":
            raise ApiError(409, "A processing job is already running")
        PLAN.update(id=plan_id, state="running", done=0, total=0, result=None, error=None, jobs=[],
                    preset=None, out_dir=str(out_dir))
    threading.Thread(
        target=build_plan_thread, daemon=True,
        args=(plan_id, jobs, preset, out_dir, bool(body.get("overwrite")), bool(body.get("probe", True)),
              bool(transcripts_dir), transcripts_dir, mode)
    ).start()
    return {"plan_id": plan_id}

def api_plan_status(_q):
    with LOCK:
        return {
            "id": PLAN["id"], "state": PLAN["state"], "done": PLAN["done"], "total": PLAN["total"],
            "error": PLAN["error"], "result": PLAN["result"] if PLAN["state"] == "done" else None
        }

# --------------------------------------------------------------------------- Execution Engine
def convert_single_video(job, preset, it):
    tmp = Path(job["out"] + ".part")
    Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
    if job.get("duration") is None:
        job["duration"], _ = probe(job["path"])
    dur = job["duration"]
    it["duration"] = dur

    cmd = [
        FFMPEG, "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
        "-i", job["path"], "-vn", "-map", "0:a:0", *preset["args"], "-f", preset["fmt"], str(tmp)
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1, creationflags=flags
    )
    with LOCK:
        RUN["proc"] = proc
    errs = []
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            k, _, v = line.partition("=")
            if k in PROGRESS_KEYS:
                if k in ("out_time_us", "out_time_ms") and v.lstrip("-").isdigit() and int(v) >= 0:
                    secs = int(v) / 1e6
                    if dur:
                        secs = min(secs, dur)
                        it["progress"] = secs / dur
                    it["secs"] = secs
                elif k == "speed":
                    it["speed"] = v
                    RUN["speed"] = v
            else:
                errs.append(line)
        rc = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        with LOCK:
            RUN["proc"] = None
    if RUN["cancel"]:
        tmp.unlink(missing_ok=True)
        return "cancelled", ""
    if rc == 0 and tmp.exists() and tmp.stat().st_size > 0:
        os.replace(tmp, job["out"])
        return "done", ""
    tmp.unlink(missing_ok=True)
    return "failed", " ".join(errs[-3:]) or f"ffmpeg exit code {rc}"

def transcribe_single_audio(audio_path: Path, out_dir: Path, model, batched_pipeline,
                            language: str, hotwords: str, batch_size: int, para_seconds: int, it: dict):
    stem = audio_path.stem
    txt_target = out_dir / f"{stem}.txt"
    srt_target = out_dir / f"{stem}.srt"
    md_target = out_dir / f"{stem}.md"

    # Fast local disk copy if on Colab to prevent Google Drive socket timeouts
    use_local_cache = is_colab() and "/content/drive" in str(audio_path)
    local_file = Path("/content") / audio_path.name if use_local_cache else audio_path
    if use_local_cache:
        try:
            shutil.copy(audio_path, local_file)
        except Exception:
            local_file = audio_path

    txt_tmp = out_dir / f"{stem}.txt.part"
    srt_tmp = out_dir / f"{stem}.srt.part"
    md_tmp = out_dir / f"{stem}.md.part"

    try:
        if batched_pipeline:
            segments, info = batched_pipeline.transcribe(
                str(local_file), language=language if language != "auto" else None,
                batch_size=batch_size, beam_size=5, initial_prompt=hotwords or None
            )
        else:
            segments, info = model.transcribe(
                str(local_file), language=language if language != "auto" else None,
                vad_filter=True, beam_size=5, condition_on_previous_text=False,
                initial_prompt=hotwords or None
            )

        it["duration"] = info.duration
        paras = []

        with open(txt_tmp, "w", encoding="utf-8") as ft, open(srt_tmp, "w", encoding="utf-8") as fs:
            for idx, s in enumerate(segments, 1):
                if RUN["cancel"]:
                    break
                text = s.text.strip()
                ft.write(f"[{ts(s.start)}] {text}\n")
                fs.write(f"{idx}\n{ts(s.start, True)} --> {ts(s.end, True)}\n{text}\n\n")

                if not paras or (s.start - paras[-1][0] >= para_seconds):
                    paras.append([s.start, []])
                paras[-1][1].append(text)

                with LOCK:
                    RUN["live_segment"] = {"ts": ts(s.start), "text": text}
                    dur = it.get("duration") or (info.duration if info and info.duration else 0)
                    if dur and dur > 0:
                        it["progress"] = min(0.99, max(0.01, s.start / dur))
                    else:
                        it["progress"] = min(0.95, it.get("progress", 0.0) + 0.01)
                    it["secs"] = s.start

        if RUN["cancel"]:
            txt_tmp.unlink(missing_ok=True)
            srt_tmp.unlink(missing_ok=True)
            md_tmp.unlink(missing_ok=True)
            return "cancelled", "Cancelled by user"

        with open(md_tmp, "w", encoding="utf-8") as fm:
            fm.write(f"# {stem}\n\n")
            fm.write(f"> **Audio Duration:** {ts(info.duration)} | **Transcribed with:** faster-whisper\n\n---\n\n")
            for start_sec, texts in paras:
                fm.write(f"**[{ts(start_sec)}]** " + " ".join(texts) + "\n\n")

        # Atomic commit: .txt is renamed last to mark completion
        srt_tmp.rename(srt_target)
        md_tmp.rename(md_target)
        txt_tmp.rename(txt_target)

        return "done", ""
    finally:
        if use_local_cache and local_file != audio_path and local_file.exists():
            local_file.unlink(missing_ok=True)

# --------------------------------------------------------------------------- Master Background Worker
def master_runner(jobs, config):
    mode = config["mode"]  # "pipeline" | "v2a" | "whisper"
    out_audio_dir = Path(config["audio_dir"])
    out_tr_dir = Path(config["transcripts_dir"])
    preset = config["preset"]
    overwrite = config["overwrite"]
    max_hours = config.get("max_hours") or 10.0
    
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    if mode in ("pipeline", "whisper"):
        out_tr_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale .part files
    for stale in out_audio_dir.rglob("*.part"):
        stale.unlink(missing_ok=True)
    if mode in ("pipeline", "whisper"):
        for stale in out_tr_dir.rglob("*.part"):
            stale.unlink(missing_ok=True)

    log(f"Job launched: Mode = {mode.upper()} | {len(jobs)} item(s)")
    t0_all = time.time()
    whisper_model = None
    batched_pipeline = None

    # Load Whisper Model if needed
    if mode in ("pipeline", "whisper"):
        with LOCK:
            RUN["stage"] = "Loading Whisper Engine..."
        try:
            m_name = config.get("whisper_model", "large-v3")
            c_type = config.get("compute_type", "float16")
            whisper_model = get_whisper_model(model_name=m_name, compute_type=c_type)
            if config.get("use_batched"):
                from faster_whisper import BatchedInferencePipeline
                batched_pipeline = BatchedInferencePipeline(model=whisper_model)
                log("BatchedInferencePipeline initialized.")
        except Exception as e:
            log(f"Whisper initialization failure: {e}")
            with LOCK:
                RUN.update(state="finished", ended=time.time(), stage=f"Whisper Init Error: {e}")
            return

    for job, it in zip(jobs, RUN["items"]):
        if RUN["cancel"]:
            break
        if max_hours and (time.time() - t0_all) > max_hours * 3600:
            log(f"Reached MAX_HOURS ({max_hours} h). Stopping batch cleanly.")
            break

        it["status"], RUN["current"] = "running", it["id"]
        t_item_start = time.time()
        audio_file_path = None
        is_audio = (job.get("media_type") == "audio")
        is_direct = (job.get("status") == "direct")

        # STAGE 1: Extract Video Audio OR Direct Audio Bypass
        if mode in ("pipeline", "whisper") and is_audio and (is_direct or job.get("status") != "convert"):
            # Audio file - direct transcribe, bypass FFmpeg conversion completely!
            audio_file_path = Path(job["path"])
            log(f"Direct audio input: {job['name']} (bypassing extraction conversion)")
            it["progress"] = 0.0
            it["status"] = "running"
        elif mode in ("pipeline", "whisper") and is_direct:
            # Video whose audio was pre-extracted
            audio_file_path = Path(job["out"])
            log(f"Pre-extracted audio found: {audio_file_path.name} (bypassing extraction)")
            it["progress"] = 0.0
            it["status"] = "running"
        elif mode in ("pipeline", "v2a"):
            with LOCK:
                RUN["stage"] = f"Extracting audio: {job['name']}"
            target_audio = Path(job["out"])
            if target_audio.exists() and target_audio.stat().st_size > 0 and not overwrite:
                log(f"Audio already exists: {target_audio.name} (skipping extraction)")
                audio_file_path = target_audio
                it["progress"] = 1.0
                it["status"] = "done"
            else:
                status, err = convert_single_video(job, preset, it)
                if status == "done":
                    audio_file_path = target_audio
                    log(f"Extracted audio: {target_audio.name}")
                else:
                    it["status"], it["err"] = status, err
                    log(f"Extraction failed: {job['name']} - {err}")
                    continue
        else:
            # Direct audio file input
            audio_file_path = Path(job["path"])

        # STAGE 2: Transcribe with Whisper if needed
        if mode in ("pipeline", "whisper") and audio_file_path and audio_file_path.exists():
            stem = audio_file_path.stem
            final_txt = out_tr_dir / f"{stem}.txt"
            if final_txt.exists() and not overwrite:
                log(f"Transcript already complete: {stem}.txt (skipped)")
                it["status"] = "done"
                it["progress"] = 1.0
                continue

            with LOCK:
                RUN["stage"] = f"Transcribing ({config.get('whisper_model', 'large-v3')}): {audio_file_path.name}"
                it["status"] = "running"
                it["progress"] = 0.0

            log(f"Starting Whisper transcription: {audio_file_path.name}")
            try:
                status, err = transcribe_single_audio(
                    audio_path=audio_file_path,
                    out_dir=out_tr_dir,
                    model=whisper_model,
                    batched_pipeline=batched_pipeline,
                    language=config.get("language", "en"),
                    hotwords=config.get("hotwords", DEFAULT_HOTWORDS),
                    batch_size=int(config.get("batch_size", 16)),
                    para_seconds=int(config.get("para_seconds", 45)),
                    it=it
                )
                it["status"], it["err"] = status, err
                if status == "done":
                    took = time.time() - t_item_start
                    dur = it.get("duration") or 1.0
                    mult = dur / max(took, 0.1)
                    log(f"Finished transcript: {stem}.md / .srt / .txt in {took/60:.1f} min ({mult:.1f}x speed)")
                else:
                    log(f"Whisper failed for {stem}: {err}")
            except Exception as e:
                it["status"], it["err"] = "failed", str(e)
                log(f"Whisper exception for {stem}: {e}")

        # Update output size if audio was created
        if audio_file_path and audio_file_path.exists():
            try:
                it["out_size"] = audio_file_path.stat().st_size
            except OSError:
                pass

    with LOCK:
        for it in RUN["items"]:
            if it["status"] == "queued":
                it["status"] = "cancelled"
        RUN.update(
            state="cancelled" if RUN["cancel"] else "finished",
            ended=time.time(),
            current=None,
            stage="Done" if not RUN["cancel"] else "Cancelled"
        )
    log("Job finished successfully!" if not RUN["cancel"] else "Job cancelled by user.")

# --------------------------------------------------------------------------- API Start & Control
def api_start(body):
    with LOCK:
        if RUN["state"] == "running":
            raise ApiError(409, "A processing batch is already running")
        if PLAN["id"] != body.get("plan_id") or PLAN["state"] != "done":
            raise ApiError(400, "Please review the dry-run plan first")
        
        jobs = [j for j in PLAN["jobs"] if j["status"] not in ("noaudio",)]
        if not jobs:
            raise ApiError(400, "No valid files ready to process")
        
        mode = str(body.get("mode", "pipeline"))
        out_dir = PLAN["out_dir"]
        root = Path(SCAN["root"])
        
        audio_dir = out_dir if mode in ("v2a", "pipeline") else str(root)
        transcripts_dir = str(body.get("transcripts_dir") or (
            Path("/content/drive/MyDrive/transcribe/transcripts_large_v3") if is_colab() else root / "transcripts"
        ))

        config = {
            "mode": mode,
            "audio_dir": audio_dir,
            "transcripts_dir": transcripts_dir,
            "preset": PLAN["preset"] or AUDIO_PRESETS[0],
            "overwrite": bool(body.get("overwrite")),
            "whisper_model": str(body.get("whisper_model", "large-v3")),
            "compute_type": str(body.get("compute_type", "float16")),
            "language": str(body.get("language", "en")),
            "hotwords": str(body.get("hotwords", DEFAULT_HOTWORDS)),
            "use_batched": bool(body.get("use_batched", False)),
            "batch_size": int(body.get("batch_size", 16)),
            "para_seconds": int(body.get("para_seconds", 45)),
            "max_hours": float(body.get("max_hours", 10.0)),
        }

        RUN.update(
            state="running",
            mode=mode,
            log=[],
            out_dir=out_dir,
            audio_dir=audio_dir,
            transcripts_dir=transcripts_dir,
            started=time.time(),
            ended=0.0,
            cancel=False,
            current=None,
            stage="Initializing...",
            live_segment=None,
            whisper_model_name=config["whisper_model"],
            items=[dict(
                id=j["id"], name=j["name"], status="queued", progress=0.0, secs=0.0,
                duration=j.get("duration"), speed="", err="", out_size=0
            ) for j in jobs]
        )

    threading.Thread(target=master_runner, args=(jobs, config), daemon=True).start()
    return {"ok": True}

def api_status(_q):
    with LOCK:
        items = [dict(i) for i in RUN["items"]]
        state = RUN["state"]
        started = RUN["started"]
        elapsed = (RUN["ended"] or time.time()) - started if started else 0
        total_dur = sum(i["duration"] or 0 for i in items)
        
        if items and total_dur > 0:
            frac = sum((i["duration"] if i["status"] == "done" else i["secs"]) for i in items) / total_dur
        elif items:
            frac = sum(1.0 if i["status"] == "done" else i["progress"] for i in items) / len(items)
        else:
            frac = 0.0

        processed = sum((i["duration"] or 0) if i["status"] == "done" else i["secs"] for i in items)
        eta = elapsed * (1 - frac) / frac if (state == "running" and frac > 0.01) else None
        
        return {
            "state": state,
            "mode": RUN["mode"],
            "stage": RUN["stage"],
            "items": items,
            "fraction": max(0.0, min(frac, 1.0)),
            "elapsed": elapsed,
            "eta": eta,
            "x": (processed / elapsed) if elapsed > 2 and processed else None,
            "speed": RUN.get("speed", ""),
            "ok": sum(1 for i in items if i["status"] == "done"),
            "failed": sum(1 for i in items if i["status"] == "failed"),
            "out_dir": RUN["out_dir"],
            "transcripts_dir": RUN["transcripts_dir"],
            "current": RUN["current"],
            "live_segment": RUN["live_segment"],
            "log": RUN["log"][-80:],
            "out_size": sum(i["out_size"] for i in items),
        }

def api_cancel(_body):
    with LOCK:
        RUN["cancel"] = True
        proc = RUN["proc"]
    if proc:
        try:
            proc.kill()
        except Exception:
            pass
    return {"ok": True}

def api_transcripts(q):
    """List or read files from the transcripts directory."""
    raw_dir = (q.get("dir") or [""])[0] or RUN["transcripts_dir"] or (
        "/content/drive/MyDrive/transcribe/transcripts_large_v3" if is_colab() else ""
    )
    if not raw_dir or not os.path.isdir(raw_dir):
        return {"files": [], "content": None}

    d = Path(raw_dir)
    file_arg = (q.get("file") or [""])[0]
    if file_arg:
        target = d / file_arg
        if target.exists() and target.is_file():
            try:
                with open(target, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read(200000)  # read first 200KB
                return {"files": [], "file": file_arg, "content": txt}
            except Exception as e:
                return {"error": str(e)}

    files = []
    for p in sorted(d.iterdir(), key=lambda x: natural(x.name)):
        if p.is_file() and p.suffix.lower() in (".md", ".srt", ".txt"):
            files.append({
                "name": p.name,
                "stem": p.stem,
                "ext": p.suffix.lower(),
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime
            })
    return {"files": files, "dir": str(d)}

def api_open(body):
    p = str(body.get("path", ""))
    if not os.path.isdir(p):
        raise ApiError(400, "Directory not found")
    if is_colab():
        return {"ok": True, "msg": "Running in cloud VM"}
    if os.name == "nt":
        os.startfile(p)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])
    return {"ok": True}

def api_quit(_body):
    api_cancel({})
    threading.Thread(target=lambda: (time.sleep(0.3), SERVER.shutdown()), daemon=True).start()
    return {"ok": True}

GET_ROUTES = {
    "/api/info": api_info,
    "/api/fs": api_fs,
    "/api/plan_status": api_plan_status,
    "/api/status": api_status,
    "/api/transcripts": api_transcripts,
}

POST_ROUTES = {
    "/api/scan": api_scan,
    "/api/plan": api_plan,
    "/api/start": api_start,
    "/api/cancel": api_cancel,
    "/api/open": api_open,
    "/api/quit": api_quit,
    "/api/mount_drive": api_mount_drive,
}

# --------------------------------------------------------------------------- HTTP Request Handler
class StudioHandler(BaseHTTPRequestHandler):
    server_version = "WhisperStudio/2.0"

    def log_message(self, *a):
        pass

    def _host_ok(self):
        # Allow localhost, 127.0.0.1, or Google Colab proxy domains (*.googleusercontent.com)
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        if host in ("127.0.0.1", "localhost") or host.endswith(".googleusercontent.com"):
            return True
        if is_colab():
            return True
        return False

    def _send(self, code: int, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # Allow framing inside Google Colab notebook cell iframe
            self.send_header("Content-Security-Policy", "frame-ancestors *")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _dispatch(self, routes, arg):
        url = urlparse(self.path)
        fn = routes.get(url.path)
        if not fn:
            return self._send(404, {"error": "Endpoint not found"})
        # Verify Token
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, {"error": "Invalid session token. Please reload the studio."})
        try:
            self._send(200, fn(arg(url)))
        except ApiError as e:
            self._send(e.code, {"error": e.msg})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "Forbidden host"})
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, STUDIO_HTML.replace("__TOKEN__", TOKEN), "text/html; charset=utf-8")
        self._dispatch(GET_ROUTES, lambda u: parse_qs(u.query))

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"error": "Forbidden host"})

        def body(_u):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        try:
            self._dispatch(POST_ROUTES, body)
        except json.JSONDecodeError:
            self._send(400, {"error": "Invalid JSON body"})

# --------------------------------------------------------------------------- Studio Single Page HTML/CSS/JS
STUDIO_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whisper & Video Studio &middot; Colab & Local</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%237c5cff'/%3E%3Cpath d='M14 32h4M22 20v24M30 12v40M38 18v28M46 26v12M52 32h0' stroke='%2322d3ee' stroke-width='5' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #070b14; --card: rgba(255,255,255,.05); --card2: rgba(255,255,255,.085); --card-hover: rgba(255,255,255,.12);
  --line: rgba(255,255,255,.1); --line-strong: rgba(255,255,255,.18);
  --txt: #f1f5f9; --mut: #94a3b8; --acc: #7c5cff; --acc2: #22d3ee; --acc3: #ec4899;
  --ok: #10b981; --warn: #f59e0b; --bad: #f43f5e;
  --shadow: 0 24px 64px -20px rgba(0,0,0,.7); --glow: 0 0 35px -5px rgba(124,92,255,.35);
  --r: 18px; color-scheme: dark;
}
[data-theme=light] {
  --bg: #f8fafc; --card: rgba(255,255,255,.85); --card2: #ffffff; --card-hover: #f1f5f9;
  --line: rgba(15,23,42,.08); --line-strong: rgba(15,23,42,.15);
  --txt: #0f172a; --mut: #64748b;
  --shadow: 0 20px 50px -20px rgba(71,85,105,.25); --glow: 0 0 30px -5px rgba(124,92,255,.2);
  color-scheme: light;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--txt);
  font: 14.5px/1.55 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  min-height: 100vh; overflow-x: hidden; -webkit-font-smoothing: antialiased;
}
.blobs { position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none; }
.blobs i { position: absolute; border-radius: 50%; filter: blur(100px); opacity: .24; animation: float 26s ease-in-out infinite; }
.blobs i:nth-child(1) { width: 55vmax; height: 55vmax; background: var(--acc); left: -15vmax; top: -20vmax; }
.blobs i:nth-child(2) { width: 45vmax; height: 45vmax; background: var(--acc2); right: -15vmax; top: 15vmax; animation-delay: -8s; opacity: .2; }
.blobs i:nth-child(3) { width: 50vmax; height: 50vmax; background: var(--acc3); left: 20vmax; bottom: -25vmax; animation-delay: -16s; opacity: .14; }
@keyframes float { 50% { transform: translate(5vmax, 4vmax) scale(1.08); } }

.wrap { max-width: 1140px; margin: 0 auto; padding: 18px 20px 60px; }

/* Header & Nav */
header { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
.logo-box {
  width: 44px; height: 44px; border-radius: 14px;
  background: linear-gradient(135deg, var(--acc), var(--acc2));
  display: grid; place-items: center; box-shadow: 0 10px 28px -10px var(--acc); flex: none;
}
.logo-box svg { width: 26px; height: 26px; }
.logo-bars rect { animation: eq 1.2s ease-in-out infinite; transform-origin: center; }
.logo-bars rect:nth-child(2) { animation-delay: -0.3s; }
.logo-bars rect:nth-child(3) { animation-delay: -0.6s; }
.logo-bars rect:nth-child(4) { animation-delay: -0.9s; }
@keyframes eq { 50% { transform: scaleY(0.35); } }

h1 { font-size: 20px; font-weight: 800; margin: 0; letter-spacing: -0.4px; }
.sub { color: var(--mut); font-size: 12.5px; }
.sp { flex: 1; }
.badge {
  font-size: 11.5px; font-weight: 600; padding: 4px 10px; border-radius: 99px;
  border: 1px solid var(--line); background: var(--card); color: var(--txt);
  display: inline-flex; align-items: center; gap: 6px;
}
.badge.ok { border-color: rgba(16,185,129,.3); color: var(--ok); background: rgba(16,185,129,.1); }
.badge.warn { border-color: rgba(245,158,11,.3); color: var(--warn); background: rgba(245,158,11,.1); }
.badge .pulse { width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }

.icon-btn {
  border: 1px solid var(--line); background: var(--card); color: var(--txt);
  width: 36px; height: 36px; border-radius: 11px; cursor: pointer; display: grid; place-items: center;
  transition: .2s;
}
.icon-btn:hover { background: var(--card2); border-color: var(--line-strong); transform: translateY(-1px); }
.icon-btn svg { width: 17px; height: 17px; }

/* Workflow Mode Tabs */
.modes-bar {
  display: flex; gap: 8px; margin-bottom: 18px; padding: 4px;
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  overflow-x: auto;
}
.mode-btn {
  flex: 1; min-width: 140px; padding: 9px 14px; border: 0; background: transparent; color: var(--mut);
  font: inherit; font-size: 13.5px; font-weight: 600; border-radius: 10px; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center; gap: 8px; transition: .2s;
  white-space: nowrap;
}
.mode-btn:hover { color: var(--txt); background: rgba(255,255,255,.04); }
.mode-btn.active {
  background: linear-gradient(135deg, var(--acc), #5b8cff 60%, var(--acc2));
  color: #fff; box-shadow: 0 8px 24px -8px var(--acc);
}
.mode-btn svg { width: 17px; height: 17px; }

/* Stepper */
.stepper { display: flex; gap: 6px; margin: 0 0 20px; padding: 0; list-style: none; }
.stepper li {
  flex: 1; display: flex; align-items: center; gap: 8px; padding: 9px 12px; border-radius: 13px;
  border: 1px solid var(--line); background: var(--card); color: var(--mut); cursor: default;
  transition: .25s; min-width: 0;
}
.stepper li b {
  width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
  font-size: 11px; background: var(--card2); flex: none; transition: .25s;
}
.stepper li span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12.5px; font-weight: 600; }
.stepper li.reach { cursor: pointer; color: var(--txt); }
.stepper li.done b { background: var(--ok); color: #042416; }
.stepper li.active {
  color: var(--txt); border: 1px solid transparent;
  background: linear-gradient(var(--bg), var(--bg)) padding-box, linear-gradient(120deg, var(--acc), var(--acc2)) border-box;
  box-shadow: 0 10px 28px -14px var(--acc);
}
.stepper li.active b { background: linear-gradient(135deg, var(--acc), var(--acc2)); color: #fff; }

/* Panels */
.panel {
  display: none; background: var(--card); border: 1px solid var(--line); border-radius: var(--r);
  padding: 24px; box-shadow: var(--shadow); backdrop-filter: blur(16px);
}
.panel.active { display: block; animation: panelIn .35s cubic-bezier(.2,.8,.2,1); }
@keyframes panelIn { from { opacity: 0; transform: translateY(12px); } }

h2 { margin: 0 0 4px; font-size: 19px; font-weight: 700; letter-spacing: -0.3px; }
.lead { color: var(--mut); margin: 0 0 18px; font-size: 13.5px; }

/* Inputs & Form elements */
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
input[type=text], input[type=number], select, textarea {
  background: var(--card2); border: 1px solid var(--line); color: var(--txt); border-radius: 12px;
  padding: 10px 14px; font: inherit; outline: none; transition: .2s; min-width: 0;
}
input[type=text]:focus, input[type=number]:focus, select:focus, textarea:focus {
  border-color: var(--acc); box-shadow: 0 0 0 3px rgba(124,92,255,.2);
}
.grow { flex: 1; }

.btn {
  border: 1px solid var(--line); background: var(--card2); color: var(--txt); border-radius: 12px;
  padding: 9px 16px; font: inherit; font-size: 13.5px; font-weight: 600; cursor: pointer;
  transition: .2s; display: inline-flex; align-items: center; justify-content: center; gap: 8px;
}
.btn:hover:not(:disabled) { transform: translateY(-1px); border-color: var(--acc); }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btn.primary {
  background: linear-gradient(135deg, var(--acc), #5b8cff 60%, var(--acc2));
  border: 0; color: #fff; box-shadow: 0 12px 28px -12px var(--acc);
}
.btn.primary:hover:not(:disabled) { box-shadow: 0 16px 36px -10px var(--acc); }
.btn.danger { color: var(--bad); border-color: rgba(244,63,94,.3); }
.btn.danger:hover:not(:disabled) { background: rgba(244,63,94,.1); }
.btn.sm { padding: 6px 11px; font-size: 12.5px; border-radius: 9px; }
.btn svg { width: 16px; height: 16px; }

.chips { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 12px; }
.chips button {
  font: inherit; font-size: 12px; color: var(--mut); background: var(--card); border: 1px solid var(--line);
  border-radius: 99px; padding: 4px 11px; cursor: pointer; transition: .2s;
}
.chips button:hover { color: var(--txt); border-color: var(--acc); background: var(--card2); }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 0 0 16px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 13px; padding: 12px 14px; }
.tile small { color: var(--mut); display: block; font-size: 11.5px; margin-bottom: 2px; }
.tile b { font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
.tile.ok b { color: var(--ok); }
.tile.acc b { color: var(--acc2); }

.tablewrap { border: 1px solid var(--line); border-radius: 13px; max-height: 380px; overflow: auto; background: var(--card); }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th {
  position: sticky; top: 0; background: var(--bg); text-align: left; font-size: 12px;
  color: var(--mut); font-weight: 600; padding: 9px 12px; z-index: 1; border-bottom: 1px solid var(--line);
}
td { padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: middle; }
tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: var(--card2); }
td.n { color: var(--mut); text-align: right; width: 50px; font-variant-numeric: tabular-nums; }
td.sz, th.sz { text-align: right; white-space: nowrap; color: var(--mut); font-variant-numeric: tabular-nums; }
td.nm { max-width: 0; width: 100%; }
td.nm div { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
td.nm small { color: var(--mut); display: block; overflow: hidden; text-overflow: ellipsis; font-size: 11.5px; }

input[type=checkbox].cb {
  appearance: none; width: 18px; height: 18px; border-radius: 6px; border: 1.5px solid var(--mut);
  cursor: pointer; display: grid; place-items: center; transition: .15s; margin: 0; background: transparent;
}
input[type=checkbox].cb:checked { background: var(--acc); border-color: var(--acc); }
input[type=checkbox].cb:checked:after {
  content: ""; width: 8px; height: 4.5px; border: 2px solid #fff; border-top: 0; border-right: 0;
  transform: rotate(-45deg) translate(1px, -1px);
}

.switch { display: inline-flex; align-items: center; gap: 9px; cursor: pointer; user-select: none; font-size: 13.5px; }
.switch input { display: none; }
.switch span {
  width: 38px; height: 22px; border-radius: 99px; background: var(--card2); border: 1px solid var(--line);
  position: relative; transition: .25s; flex: none;
}
.switch span:after {
  content: ""; position: absolute; left: 3px; top: 3px; width: 14px; height: 14px; border-radius: 50%;
  background: var(--mut); transition: .25s;
}
.switch input:checked + span { background: var(--acc); border-color: var(--acc); }
.switch input:checked + span:after { left: 19px; background: #fff; }

.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-bottom: 16px; }
.pcard {
  position: relative; text-align: left; font: inherit; color: inherit; background: var(--card);
  border: 1.5px solid var(--line); border-radius: 14px; padding: 14px 16px; cursor: pointer; transition: .2s;
}
.pcard:hover { transform: translateY(-2px); border-color: var(--acc); background: var(--card2); }
.pcard.sel {
  border-color: var(--acc);
  background: linear-gradient(160deg, rgba(124,92,255,.16), rgba(34,211,238,.06));
  box-shadow: 0 12px 30px -15px var(--acc);
}
.pcard h3 { margin: 0 0 3px; font-size: 15px; font-weight: 700; }
.pcard p { margin: 0; color: var(--mut); font-size: 12.5px; }
.pcard .meta { margin-top: 8px; font-size: 11.5px; color: var(--mut); }
.tag {
  display: inline-block; font-size: 10.5px; font-weight: 700; color: #042416; background: var(--ok);
  border-radius: 99px; padding: 2px 8px; margin-left: 6px; vertical-align: middle;
}
.tag.blue { background: var(--acc2); color: #04202c; }
.pcard .tick {
  position: absolute; right: 12px; top: 12px; width: 20px; height: 20px; border-radius: 50%;
  background: var(--acc); display: grid; place-items: center; transform: scale(0);
  transition: .22s cubic-bezier(.3,1.6,.5,1);
}
.pcard.sel .tick { transform: scale(1); }
.pcard .tick svg { width: 12px; height: 12px; stroke: #fff; }

.field { margin-top: 16px; }
.field label.t { display: block; font-size: 12.5px; color: var(--mut); margin-bottom: 5px; font-weight: 600; }
.warns { display: grid; gap: 8px; margin: 0 0 14px; }
.warn {
  display: flex; gap: 10px; background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.25);
  color: var(--warn); border-radius: 11px; padding: 8px 12px; font-size: 13px;
}
.pill { font-size: 11px; font-weight: 700; border-radius: 99px; padding: 2.5px 9px; white-space: nowrap; }
.pill.ready { background: rgba(16,185,129,.15); color: var(--ok); }
.pill.direct { background: rgba(16,185,129,.18); color: var(--ok); border: 1px solid rgba(16,185,129,.35); }
.pill.convert { background: rgba(124,92,255,.18); color: #a78bfa; border: 1px solid rgba(124,92,255,.35); }
.pill.skip, .pill.done_transcript { background: var(--card2); color: var(--mut); }
.pill.noaudio { background: rgba(244,63,94,.15); color: var(--bad); }

.tag-media {
  font-size: 10px; font-weight: 700; border-radius: 6px; padding: 2px 7px;
  margin-right: 7px; display: inline-block; vertical-align: middle; letter-spacing: 0.2px;
}
.tag-media.vid { background: rgba(124,92,255,.16); color: #a78bfa; border: 1px solid rgba(124,92,255,.32); }
.tag-media.aud { background: rgba(16,185,129,.16); color: #34d399; border: 1px solid rgba(16,185,129,.32); }

.filter-btn {
  border: 1px solid var(--line); background: var(--card); color: var(--mut); border-radius: 9px;
  padding: 5px 12px; font-size: 12px; font-weight: 600; cursor: pointer; transition: .2s;
}
.filter-btn:hover { color: var(--txt); border-color: var(--line-strong); }
.filter-btn.active {
  background: var(--card2); color: var(--txt); border-color: var(--acc);
  box-shadow: 0 4px 14px -6px var(--acc);
}

/* Stage Banner */
.stage-banner {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  background: var(--card); border: 1px solid var(--line); border-radius: 13px;
  padding: 10px 16px; margin-bottom: 16px; box-shadow: 0 2px 8px -4px rgba(0,0,0,.15);
}
.stage-left { display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1; }
.pulse-beacon {
  width: 9px; height: 9px; border-radius: 50%; background: var(--acc2);
  box-shadow: 0 0 0 0 rgba(34,211,238,.7); animation: pulseBeacon 1.8s infinite; flex-shrink: 0;
}
@keyframes pulseBeacon {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34,211,238,.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34,211,238,0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34,211,238,0); }
}
.stage-banner.done .pulse-beacon { background: var(--ok); animation: none; box-shadow: 0 0 6px var(--ok); }
.stage-badge {
  font-size: 10px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase;
  padding: 3px 8px; border-radius: 6px; background: rgba(124,92,255,.16); color: var(--acc);
  border: 1px solid rgba(124,92,255,.3); flex-shrink: 0;
}
.stage-badge.run { background: rgba(34,211,238,.14); color: var(--acc2); border-color: rgba(34,211,238,.3); }
.stage-badge.ok { background: rgba(16,185,129,.14); color: var(--ok); border-color: rgba(16,185,129,.3); }
.stage-badge.warn { background: rgba(244,63,94,.14); color: var(--bad); border-color: rgba(244,63,94,.3); }
.stage-text { font-size: 13.5px; font-weight: 600; color: var(--txt); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stage-right { font-size: 12px; color: var(--mut); flex-shrink: 0; font-weight: 500; }

/* Progress & Monitor */
.hero { display: grid; grid-template-columns: 200px 1fr; gap: 24px; align-items: center; margin-bottom: 18px; }
.ring {
  position: relative; width: 196px; height: 196px; margin: 0 auto;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.ring svg { width: 100%; height: 100%; transform: rotate(-90deg); overflow: visible; }
.ring circle { fill: none; stroke-width: 9; stroke-linecap: round; }
.ring .bg { stroke: rgba(255,255,255,.07); }
[data-theme=light] .ring .bg { stroke: rgba(0,0,0,.08); }
.ring .spinner-arc {
  stroke: url(#spinGrad); stroke-dasharray: 85 254; stroke-dashoffset: 0; opacity: 0;
  transform-origin: 60px 60px; pointer-events: none; transition: opacity .3s ease;
}
.ring.running .spinner-arc { opacity: 1; animation: ringArcSpin 1.4s linear infinite; }
@keyframes ringArcSpin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.ring .fg {
  stroke: url(#g); stroke-dasharray: 339.292; stroke-dashoffset: 339.292;
  transition: stroke-dashoffset .4s cubic-bezier(0.4, 0, 0.2, 1), stroke .3s ease;
}
.ring.running .fg { filter: drop-shadow(0 0 6px rgba(34,211,238,.45)); }
.ring.done .fg { stroke: var(--ok); filter: drop-shadow(0 0 8px rgba(16,185,129,.4)); }
.ring.done .spinner-arc { display: none; }

.ring .pc {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; text-align: center;
  pointer-events: none; padding: 10px; box-sizing: border-box;
}
.pc-tag {
  font-size: 10px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--acc2); line-height: 1; height: 12px; margin-bottom: 2px;
  max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; transition: color .2s;
}
.ring.done .pc-tag { color: var(--ok); }
.pc-num-wrap { display: flex; align-items: baseline; justify-content: center; line-height: 1; margin: 1px 0 3px; }
.pc-num {
  font-size: 40px; font-weight: 800; letter-spacing: -1.5px; line-height: 1;
  color: var(--txt); font-variant-numeric: tabular-nums; font-family: inherit;
}
.pc-pct { font-size: 17px; font-weight: 700; color: var(--acc2); margin-left: 2px; opacity: .9; }
.ring.done .pc-pct { color: var(--ok); }
.pc-sub {
  font-size: 11.5px; font-weight: 600; color: var(--mut); line-height: 1.2; height: 14px;
  max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block;
}

.bar { height: 8px; border-radius: 99px; background: var(--card2); overflow: hidden; position: relative; }
.bar i {
  display: block; height: 100%; width: 0; border-radius: 99px;
  background: linear-gradient(90deg, var(--acc), var(--acc2)); transition: width .4s ease;
  position: relative; overflow: hidden;
}
.bar i:after {
  content: ""; position: absolute; inset: 0;
  background: repeating-linear-gradient(115deg, rgba(255,255,255,.25) 0 10px, transparent 10px 20px);
  background-size: 40px 100%; animation: barSlide 1s linear infinite; opacity: .5;
}
@keyframes barSlide { to { background-position: 40px 0; } }
.bar.idle i:after { animation: none; opacity: 0; }
.bar.ind i { width: 35%!important; animation: ind 1.3s ease-in-out infinite; }
@keyframes ind { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }

.cur { background: var(--card); border: 1px solid var(--line); border-radius: 13px; padding: 13px 16px; margin-bottom: 14px; }
.cur .top { display: flex; gap: 10px; justify-content: space-between; margin-bottom: 8px; font-size: 13.5px; }
.cur .top b { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.cur .top span { color: var(--mut); white-space: nowrap; font-variant-numeric: tabular-nums; font-size: 12.5px; }

.live-speech {
  background: rgba(124,92,255,.1); border: 1px solid rgba(124,92,255,.25); border-radius: 11px;
  padding: 10px 14px; margin-top: 10px; font-size: 13px; display: flex; gap: 10px; align-items: flex-start;
}
.live-speech .mic { color: var(--acc2); font-weight: 700; flex: none; }
.live-speech .text { flex: 1; color: var(--txt); font-style: italic; }

.queue { border: 1px solid var(--line); border-radius: 13px; max-height: 260px; overflow: auto; background: var(--card); }
.qi { display: flex; gap: 10px; align-items: center; padding: 8px 12px; border-bottom: 1px solid var(--line); font-size: 13.5px; }
.qi:last-child { border-bottom: 0; }
.qi .nm { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.qi .st { color: var(--mut); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.qi.running { background: rgba(124,92,255,.1); }
.dot { width: 18px; height: 18px; flex: none; display: grid; place-items: center; }
.dot:before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--mut); opacity: .5; }
.qi.running .dot:before {
  width: 14px; height: 14px; background: none; border: 2px solid var(--acc);
  border-top-color: transparent; opacity: 1; animation: spin .8s linear infinite;
}
.qi.done .dot:before { content: "\2713"; width: auto; height: auto; background: none; color: var(--ok); font-weight: 900; }
.qi.failed .dot:before { content: "\2715"; width: auto; height: auto; background: none; color: var(--bad); font-weight: 900; }
@keyframes spin { to { transform: rotate(360deg); } }

pre.log {
  margin: 8px 0 0; background: var(--card); border: 1px solid var(--line); border-radius: 11px;
  padding: 11px; max-height: 180px; overflow: auto; font: 12px/1.45 'JetBrains Mono', Consolas, monospace;
  color: var(--mut);
}

/* Transcripts Viewer Tab */
.tr-viewer { display: grid; grid-template-columns: 280px 1fr; gap: 16px; min-height: 480px; }
.tr-list { border: 1px solid var(--line); border-radius: 13px; background: var(--card); overflow: auto; max-height: 520px; }
.tr-item {
  padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; transition: .15s;
}
.tr-item:hover { background: var(--card2); }
.tr-item.active { background: rgba(124,92,255,.14); border-left: 3px solid var(--acc); }
.tr-item b { display: block; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tr-item small { color: var(--mut); font-size: 11px; }

.tr-content {
  border: 1px solid var(--line); border-radius: 13px; background: var(--card); padding: 18px;
  display: flex; flex-direction: column; overflow: hidden; max-height: 520px;
}
.tr-tabs { display: flex; gap: 6px; margin-bottom: 12px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
.tr-tab {
  background: transparent; border: 0; color: var(--mut); font: inherit; font-size: 12.5px;
  font-weight: 600; padding: 5px 12px; border-radius: 8px; cursor: pointer;
}
.tr-tab.active { background: var(--card2); color: var(--txt); }
.tr-body { flex: 1; overflow: auto; font-size: 13.5px; line-height: 1.6; }
.tr-body pre { font: 12.5px/1.5 'JetBrains Mono', monospace; margin: 0; white-space: pre-wrap; color: var(--txt); }

.modal {
  position: fixed; inset: 0; background: rgba(3,6,15,.65); backdrop-filter: blur(8px);
  display: none; place-items: center; z-index: 50; padding: 20px;
}
.modal.show { display: grid; animation: fadeIn .2s; }
@keyframes fadeIn { from { opacity: 0; } }
.sheet {
  width: min(680px, 100%); max-height: 85vh; display: flex; flex-direction: column;
  background: var(--bg); border: 1px solid var(--line-strong); border-radius: 20px;
  box-shadow: var(--shadow); overflow: hidden;
}
[data-theme=light] .sheet { background: #fff; }
.sheet .hd { padding: 14px 16px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: center; }
.crumbs { display: flex; flex-wrap: wrap; gap: 2px; font-size: 12.5px; flex: 1; min-width: 0; }
.crumbs button { background: none; border: 0; color: var(--mut); font: inherit; cursor: pointer; padding: 3px 6px; border-radius: 6px; }
.crumbs button:hover { background: var(--card2); color: var(--txt); }
.crumbs button:last-child { color: var(--txt); font-weight: 700; }
.sheet .bd { display: grid; grid-template-columns: 160px 1fr; min-height: 0; flex: 1; }
.side { border-right: 1px solid var(--line); padding: 8px; overflow: auto; }
.dirs { padding: 8px; overflow: auto; }
.side button, .dirs button {
  display: flex; align-items: center; gap: 8px; width: 100%; text-align: left; background: none;
  border: 0; color: var(--txt); font: inherit; font-size: 13.5px; padding: 7px 9px; border-radius: 8px; cursor: pointer;
}
.side button:hover, .dirs button:hover { background: var(--card2); }
.sheet .ft { padding: 12px 16px; border-top: 1px solid var(--line); display: flex; gap: 10px; align-items: center; justify-content: space-between; }

.toasts { position: fixed; right: 18px; bottom: 18px; display: grid; gap: 8px; z-index: 90; }
.toast {
  background: var(--bg); border: 1px solid var(--bad); color: var(--txt); border-radius: 12px;
  padding: 10px 16px; box-shadow: var(--shadow); max-width: 360px; animation: panelIn .3s; font-size: 13px;
}
.toast.info { border-color: var(--acc); }
[data-theme=light] .toast { background: #fff; }

.confetti { position: fixed; inset: 0; pointer-events: none; overflow: hidden; z-index: 80; }
.confetti i { position: absolute; top: -20px; width: 8px; height: 13px; border-radius: 2px; animation: fall linear forwards; }
@keyframes fall { to { transform: translateY(110vh) rotate(720deg); opacity: .8; } }

@media(max-width: 760px) {
  .stepper li span { display: none; }
  .hero { grid-template-columns: 1fr; justify-items: center; }
  .tr-viewer { grid-template-columns: 1fr; }
  .sheet .bd { grid-template-columns: 1fr; }
  .side { display: none; }
}
</style>
</head>
<body>
<div class="blobs"><i></i><i></i><i></i></div>
<div class="wrap">
  <header>
    <div class="logo-box">
      <svg viewBox="0 0 26 26" fill="#fff" class="logo-bars">
        <rect x="2" y="9" width="3" height="8" rx="1.5"/>
        <rect x="8" y="4" width="3" height="18" rx="1.5"/>
        <rect x="14" y="7" width="3" height="12" rx="1.5"/>
        <rect x="20" y="10" width="3" height="6" rx="1.5"/>
      </svg>
    </div>
    <div>
      <h1>Whisper & Video Studio</h1>
      <div class="sub">Colab Pro GPU &middot; Video2Audio &middot; faster-whisper Batch Pipeline</div>
    </div>
    <div class="sp"></div>
    <span class="badge" id="gpuBadge"><span class="pulse"></span><span id="gpuText">Detecting GPU...</span></span>
    <span class="badge" id="driveBadge" style="cursor:pointer;" title="Click to mount Google Drive"><span id="driveText">Drive</span></span>
    <button class="icon-btn" id="themeBtn" title="Toggle Light/Dark Theme">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    </button>
  </header>

  <!-- Mode Selector -->
  <div class="modes-bar">
    <button class="mode-btn active" data-m="pipeline">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      All-in-One Pipeline
    </button>
    <button class="mode-btn" data-m="v2a">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/></svg>
      Video to Audio
    </button>
    <button class="mode-btn" data-m="whisper">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
      Whisper Transcribe
    </button>
    <button class="mode-btn" data-m="viewer">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      Transcripts Studio
    </button>
  </div>

  <!-- Stepper for Pipeline / Conversion -->
  <ol class="stepper" id="stepper">
    <li data-s="1"><b>1</b><span>Input Folder</span></li>
    <li data-s="2"><b>2</b><span>Select Files</span></li>
    <li data-s="3"><b>3</b><span>Config & Model</span></li>
    <li data-s="4"><b>4</b><span>Review Plan</span></li>
    <li data-s="5"><b>5</b><span>Live Monitor</span></li>
  </ol>

  <!-- STEP 1: Folder Picker -->
  <section class="panel active" id="p1">
    <h2>Where are your media files?</h2>
    <p class="lead" id="p1Lead">Choose the folder containing your videos or audio. Files are processed in session-number order.</p>
    <div class="row">
      <input type="text" id="path" class="grow" placeholder="Enter path (e.g. /content/drive/MyDrive/transcribe/videos)" spellcheck="false">
      <button class="btn" id="browseBtn">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
        Browse
      </button>
    </div>
    <div class="row" style="margin-top:14px; gap:16px; flex-wrap:wrap">
      <label class="switch"><input type="checkbox" id="recursive"><span></span>Search sub-folders recursively</label>
      <div class="row" style="gap:6px">
        <span class="sub" style="font-weight:600">Scan Filter:</span>
        <button type="button" class="filter-btn active" data-scan="all" id="scanAllBtn">All Media (Videos + Audio)</button>
        <button type="button" class="filter-btn" data-scan="video" id="scanVidBtn">Videos Only</button>
        <button type="button" class="filter-btn" data-scan="audio" id="scanAudBtn">Audio Only</button>
      </div>
    </div>
    <div class="chips" id="recents"></div>
    <div class="row" style="margin-top:20px; justify-content:space-between">
      <button class="btn sm" id="quickColabBtn" style="display:none">Set Colab Defaults</button>
      <button class="btn primary" id="scanBtn">
        Scan Files
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
      </button>
    </div>
  </section>

  <!-- STEP 2: Choose Videos/Audios -->
  <section class="panel" id="p2">
    <h2>Select Files to Process</h2>
    <p class="lead">Videos and audio files sorted by session number. Audio files directly transcribe without conversion.</p>
    <div class="tiles">
      <div class="tile"><small>Total Files</small><b id="tFound">0</b></div>
      <div class="tile"><small>Source Size</small><b id="tSize">0</b></div>
      <div class="tile ok"><small>Selected</small><b id="tSel">0</b></div>
      <div class="tile acc"><small>Selected Size</small><b id="tSelSize">0</b></div>
    </div>
    <div class="row" style="margin-bottom:10px">
      <input type="text" id="q" class="grow" placeholder="Search by filename or session keyword...">
      <div class="row" style="gap:6px">
        <span class="sub">Session Range:</span>
        <input type="number" id="rFrom" placeholder="From" min="0" style="width:75px">
        <input type="number" id="rTo" placeholder="To" min="0" style="width:75px">
        <button class="btn sm" id="rApply">Apply Range</button>
      </div>
    </div>
    <div class="row" style="margin-bottom:12px; justify-content:space-between; flex-wrap:wrap; gap:10px">
      <div class="row" style="gap:6px">
        <button class="btn sm" id="selAll">All</button>
        <button class="btn sm" id="selNone">None</button>
        <button class="btn sm" id="selInv">Invert</button>
      </div>
      <div class="row" style="gap:6px">
        <button class="filter-btn active" id="fAll">All Media (<span id="cntAll">0</span>)</button>
        <button class="filter-btn" id="fVid">🎬 Videos (<span id="cntVid">0</span>)</button>
        <button class="filter-btn" id="fAud">🎵 Audio (<span id="cntAud">0</span>)</button>
      </div>
      <span class="sub" id="visInfo"></span>
    </div>
    <div class="tablewrap">
      <table>
        <thead>
          <tr>
            <th style="width:36px"><input type="checkbox" class="cb" id="hdrCb"></th>
            <th style="width:50px">No.</th>
            <th>Filename</th>
            <th class="sz">Size</th>
          </tr>
        </thead>
        <tbody id="vbody"></tbody>
      </table>
    </div>
    <div class="row" style="margin-top:20px; justify-content:space-between">
      <button class="btn" data-go="1">Back</button>
      <button class="btn primary" id="to3">Configure Settings</button>
    </div>
  </section>

  <!-- STEP 3: Config & Model Settings -->
  <section class="panel" id="p3">
    <h2>Processing Settings & Model Config</h2>
    <p class="lead">Fine-tune audio extraction and faster-whisper parameters.</p>

    <!-- Audio Presets Section (for Pipeline & V2A) -->
    <div id="audioPresetSection">
      <label class="t" style="display:block;margin-bottom:8px;font-weight:600">Audio Extraction Format</label>
      <div class="cards" id="presets"></div>
      <div class="field">
        <label class="t">Save Extracted Audio to</label>
        <div class="row">
          <input type="text" id="outdir" class="grow" spellcheck="false">
          <button class="btn sm" id="outBrowse">Browse</button>
          <button class="btn sm" id="outReset">Default</button>
        </div>
      </div>
    </div>

    <!-- Whisper Settings Section (for Pipeline & Whisper) -->
    <div id="whisperSection" style="margin-top:20px">
      <label class="t" style="display:block;margin-bottom:8px;font-weight:600">faster-whisper Model</label>
      <div class="cards" id="models"></div>

      <div class="field">
        <label class="t">Save Transcripts (.txt, .srt, .md) to</label>
        <div class="row">
          <input type="text" id="trOutDir" class="grow" spellcheck="false">
          <button class="btn sm" id="trBrowse">Browse</button>
          <button class="btn sm" id="trReset">Default</button>
        </div>
      </div>

      <div class="row" style="margin-top:14px; gap:16px">
        <div style="flex:1">
          <label class="t" style="display:block;margin-bottom:4px;font-size:12.5px;color:var(--mut);font-weight:600">Compute Type</label>
          <select id="computeType" style="width:100%">
            <option value="float16" selected>float16 (Standard GPU - Best)</option>
            <option value="int8_float16">int8_float16 (Low VRAM)</option>
            <option value="int8">int8 (CPU Mode)</option>
            <option value="bfloat16">bfloat16 (A100 / L4)</option>
          </select>
        </div>
        <div style="flex:1">
          <label class="t" style="display:block;margin-bottom:4px;font-size:12.5px;color:var(--mut);font-weight:600">Spoken Language</label>
          <select id="language" style="width:100%">
            <option value="en" selected>English (en)</option>
            <option value="auto">Auto-detect</option>
            <option value="hi">Hindi</option>
            <option value="te">Telugu</option>
            <option value="ta">Tamil</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
          </select>
        </div>
      </div>

      <div class="field">
        <div class="row" style="justify-content:space-between">
          <label class="t">Hotwords / Initial Prompt (Technical vocabulary)</label>
          <button class="btn sm" id="resetHotwordsBtn" style="padding:2px 8px;font-size:11px">Reset Default</button>
        </div>
        <textarea id="hotwords" rows="3" style="width:100%;resize:vertical;font-size:13px"></textarea>
        <div class="chips" id="hotwordChips"></div>
      </div>

      <div class="row" style="margin-top:14px; gap:20px">
        <label class="switch">
          <input type="checkbox" id="useBatched">
          <span></span>
          <b>Batched Inference</b> (2x-4x faster on GPU, uses BatchedInferencePipeline)
        </label>
        <div id="batchSizeBox" style="display:none; align-items:center; gap:8px" class="row">
          <span class="sub">Batch Size:</span>
          <select id="batchSize">
            <option value="8">8</option>
            <option value="16" selected>16 (Default)</option>
            <option value="24">24 (A100/L4)</option>
          </select>
        </div>
      </div>

      <div class="row" style="margin-top:12px; gap:20px">
        <div class="row" style="gap:6px">
          <span class="sub">Paragraph Duration:</span>
          <input type="number" id="paraSeconds" value="45" min="15" max="180" style="width:70px">
          <span class="sub">sec (for .md study notes)</span>
        </div>
        <div class="row" style="gap:6px">
          <span class="sub">Max Runtime:</span>
          <input type="number" id="maxHours" value="10" min="1" max="24" style="width:65px">
          <span class="sub">hours guard</span>
        </div>
      </div>
    </div>

    <div class="field" style="margin-top:16px">
      <label class="switch"><input type="checkbox" id="overwrite"><span></span>Overwrite files that already exist (Default: off, skips done files)</label>
    </div>

    <div class="row" style="margin-top:24px; justify-content:space-between">
      <button class="btn" data-go="2">Back</button>
      <button class="btn primary" id="to4">Review Dry Run</button>
    </div>
  </section>

  <!-- STEP 4: Review Dry Run -->
  <section class="panel" id="p4">
    <h2>Review Before Starting</h2>
    <p class="lead">Dry run summary. Existing files will be safely skipped to resume progress.</p>
    <div id="planLoad" style="padding:40px;text-align:center">
      <div class="logo-box" style="margin:0 auto 12px;animation:spin 1s linear infinite"></div>
      <div id="planMsg">Analyzing media files & probe metadata...</div>
      <div class="bar" style="max-width:320px;margin:14px auto 0"><i id="planBar"></i></div>
    </div>
    <div id="planBody" style="display:none">
      <div class="tiles" id="planTiles"></div>
      <div class="warns" id="planWarns"></div>
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              <th>No.</th>
              <th>Filename</th>
              <th class="sz">Size</th>
              <th class="sz">Audio Length</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="planRows"></tbody>
        </table>
      </div>
    </div>
    <div class="row" style="margin-top:20px; justify-content:space-between">
      <button class="btn" data-go="3">Back</button>
      <button class="btn primary" id="startBtn" disabled>
        Start Processing
        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>
      </button>
    </div>
  </section>

  <!-- STEP 5: Live Execution Dashboard -->
  <section class="panel" id="p5">
    <div class="stage-banner" id="stageBanner">
      <div class="stage-left">
        <span class="pulse-beacon"></span>
        <span class="stage-badge" id="stageBadge">INITIALIZING</span>
        <span class="stage-text" id="stageText">Preparing execution batch...</span>
      </div>
      <div class="stage-right" id="stageModeText"></div>
    </div>

    <div class="hero">
      <div class="ring" id="ring">
        <svg viewBox="0 0 120 120">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#7c5cff"/>
              <stop offset="100%" stop-color="#22d3ee"/>
            </linearGradient>
            <linearGradient id="spinGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#22d3ee" stop-opacity="0.9"/>
              <stop offset="45%" stop-color="#7c5cff" stop-opacity="0.25"/>
              <stop offset="100%" stop-color="#7c5cff" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <circle class="bg" cx="60" cy="60" r="54"/>
          <circle class="spinner-arc" id="ringSpinner" cx="60" cy="60" r="54"/>
          <circle class="fg" id="ringFg" cx="60" cy="60" r="54"/>
        </svg>
        <div class="pc">
          <span class="pc-tag" id="pcTag">PROGRESS</span>
          <div class="pc-num-wrap">
            <span class="pc-num" id="pct">0</span><span class="pc-pct">%</span>
          </div>
          <span class="pc-sub" id="pcSub">Ready</span>
        </div>
      </div>
      <div style="flex:1; min-width:0">
        <div class="tiles">
          <div class="tile"><small>Processed</small><b id="sDone">0/0</b></div>
          <div class="tile"><small>Time Left (ETA)</small><b id="sEta">--</b></div>
          <div class="tile"><small>Elapsed</small><b id="sEl">0:00</b></div>
          <div class="tile acc"><small>Speed</small><b id="sX">--</b></div>
          <div class="tile"><small>Audio Processed</small><b id="sOut">0 MB</b></div>
        </div>
        <div class="cur" id="curBox">
          <div class="top">
            <b id="curName">Waiting...</b>
            <span id="curInfo"></span>
          </div>
          <div class="bar" id="curBar"><i id="curFill"></i></div>
          <div class="live-speech" id="liveSpeechBox" style="display:none">
            <span class="mic">&#127908; <span id="liveTs">[00:00]</span>:</span>
            <span class="text" id="liveText">...</span>
          </div>
        </div>
      </div>
    </div>

    <div class="queue" id="queue"></div>

    <details style="margin-top:14px">
      <summary style="cursor:pointer;color:var(--mut);font-size:12.5px;font-weight:600">Live Terminal Logs</summary>
      <pre class="log" id="logBox"></pre>
    </details>

    <div class="row" style="margin-top:18px; justify-content:space-between">
      <button class="btn danger" id="cancelBtn">Cancel Run</button>
      <div class="row">
        <button class="btn" id="openTranscriptsBtn" style="display:none">Open Transcripts Studio</button>
        <button class="btn primary" id="runAgainBtn" style="display:none">Process More</button>
      </div>
    </div>
  </section>

  <!-- TRANSCRIPTS VIEWER TAB -->
  <section class="panel" id="pViewer">
    <div class="row" style="justify-content:space-between;margin-bottom:14px">
      <div>
        <h2>Transcripts Studio & Markdown Reader</h2>
        <div class="sub">Read study notes, inspect subtitles (.srt), or copy raw text</div>
      </div>
      <div class="row">
        <button class="btn sm" id="refreshTrBtn">Refresh</button>
        <button class="btn sm" id="copyTrBtn">Copy Text</button>
      </div>
    </div>
    <div class="tr-viewer">
      <div class="tr-list" id="trFileList"></div>
      <div class="tr-content">
        <div class="tr-tabs">
          <button class="tr-tab active" data-view="md">Study Notes (.md)</button>
          <button class="tr-tab" data-view="srt">Subtitles (.srt)</button>
          <button class="tr-tab" data-view="txt">Raw Lines (.txt)</button>
        </div>
        <div class="tr-body" id="trBodyView">
          <div style="color:var(--mut);text-align:center;padding:40px">Select a transcript on the left to read</div>
        </div>
      </div>
    </div>
  </section>
</div>

<!-- Modal Directory Browser -->
<div class="modal" id="modal">
  <div class="sheet">
    <div class="hd">
      <div class="crumbs" id="crumbs"></div>
      <button class="icon-btn" id="mClose"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
    </div>
    <div class="bd">
      <div class="side" id="side"></div>
      <div class="dirs" id="dirs"></div>
    </div>
    <div class="ft">
      <span class="sub" id="mInfo"></span>
      <button class="btn primary" id="mPick">Select Folder</button>
    </div>
  </div>
</div>

<div class="toasts" id="toasts"></div>
<div class="confetti" id="confetti"></div>

<script>
const TOKEN = "__TOKEN__";
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const S = {
  mode: "pipeline", // pipeline | v2a | whisper | viewer
  step: 1, maxStep: 1,
  info: null, root: "", items: [], sel: new Set(),
  preset: "1", whisperModel: "large-v3",
  planId: null, plan: null,
  running: false, timer: null,
  activeTranscriptStem: null, activeTranscriptExt: "md",
  cachedTranscripts: {}
};

const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const human = n => { if (n == null) return "--"; const u = ["B","KB","MB","GB","TB"]; let i = 0; while(n >= 1024 && i < 4){ n /= 1024; i++; } return (i < 2 ? Math.round(n) : n.toFixed(1)) + " " + u[i]; };
const hms = s => { if (s == null || !isFinite(s)) return "--"; s = Math.round(s); const h = Math.floor(s/3600), m = Math.floor(s%3600/60), x = s%60; return h ? `${h}:${String(m).padStart(2,"0")}:${String(x).padStart(2,"0")}` : `${m}:${String(x).padStart(2,"0")}`; };
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function api(path, body) {
  const opt = body === undefined ? { headers: { "X-Token": TOKEN } } : {
    method: "POST", headers: { "X-Token": TOKEN, "Content-Type": "application/json" },
    body: JSON.stringify(body)
  };
  let r;
  try { r = await fetch("/api/" + path, opt); }
  catch(e) { throw new Error("Cannot reach studio server. Check Colab session."); }
  let j = {}; try { j = await r.json(); } catch(e){}
  if (!r.ok) throw new Error(j.error || ("Error (" + r.status + ")"));
  return j;
}

function toast(msg, info) {
  const t = document.createElement("div");
  t.className = "toast" + (info ? " info" : "");
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 5000);
}

/* ---------- Navigation & Mode Switch ---------- */
function go(n) {
  S.step = n; S.maxStep = Math.max(S.maxStep, n);
  $$(".panel").forEach(p => p.classList.remove("active"));
  $("#p" + n).classList.add("active");
  $$("#stepper li").forEach(li => {
    const s = +li.dataset.s;
    li.classList.toggle("active", s === n);
    li.classList.toggle("done", s < n);
    li.classList.toggle("reach", s <= S.maxStep && !S.running);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$$("#stepper li").forEach(li => li.onclick = () => {
  const s = +li.dataset.s;
  if (S.running || s > S.maxStep) return;
  go(s);
});
$$("[data-go]").forEach(b => b.onclick = () => go(+b.dataset.go));

$$(".mode-btn").forEach(btn => btn.onclick = () => setMode(btn.dataset.m));

function setMode(m) {
  S.mode = m;
  $$(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.m === m));
  if (m === "viewer") {
    $("#stepper").style.display = "none";
    $$(".panel").forEach(p => p.classList.remove("active"));
    $("#pViewer").classList.add("active");
    loadTranscripts();
    return;
  }
  $("#stepper").style.display = "flex";
  $("#audioPresetSection").style.display = (m === "pipeline" || m === "v2a") ? "block" : "none";
  $("#whisperSection").style.display = (m === "pipeline" || m === "whisper") ? "block" : "none";
  
  if (m === "pipeline") {
    $("#p1Lead").textContent = "All-in-One: converts videos to audio and immediately runs faster-whisper.";
  } else if (m === "v2a") {
    $("#p1Lead").textContent = "Video to Audio mode: extracts high-quality speech or music audio tracks.";
  } else {
    $("#p1Lead").textContent = "Whisper Transcribe mode: transcribes existing audio files into .txt, .srt, .md.";
  }
  go(S.step);
}

/* ---------- Theme & Colab Hardware ---------- */
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("wstudio-theme", t); } catch(e){}
}
$("#themeBtn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

async function updateHardware() {
  if (!S.info) return;
  const g = S.info.gpu;
  const gb = $("#gpuBadge");
  if (g.available) {
    gb.className = "badge ok";
    $("#gpuText").textContent = `${g.name} &middot; ${(g.total_mb/1024).toFixed(1)} GB`;
  } else {
    gb.className = "badge warn";
    $("#gpuText").textContent = S.info.is_colab ? "Colab CPU (Enable GPU in Runtime!)" : "CPU Mode";
  }

  const db = $("#driveBadge");
  if (S.info.drive_mounted) {
    db.className = "badge ok";
    $("#driveText").textContent = "Drive Mounted";
  } else if (S.info.is_colab) {
    db.className = "badge warn";
    $("#driveText").textContent = "Mount Drive";
    db.onclick = async () => {
      $("#driveText").textContent = "Mounting...";
      const r = await api("mount_drive", {});
      if (r.mounted) {
        toast("Google Drive mounted successfully!", true);
        S.info.drive_mounted = true;
        updateHardware();
      } else {
        toast("Failed to mount drive. Check notebook permissions.");
      }
    };
  } else {
    db.style.display = "none";
  }
}

/* ---------- Folder Browser Modal ---------- */
let browseCb = null, browsePath = "";
const ICON_DIR = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>';

async function openBrowser(start, cb) {
  browseCb = cb;
  $("#modal").classList.add("show");
  await loadDir(start || S.info.home);
}

async function loadDir(p) {
  try {
    const d = await api("fs?path=" + encodeURIComponent(p || ""));
    browsePath = d.path;
    const parts = [`<button data-p="">Roots</button>`];
    if (d.path) {
      const sep = S.info.sep;
      let acc = "";
      const segs = d.path.split(sep).filter(Boolean);
      segs.forEach((sg, i) => {
        acc = (i === 0 ? (sep === "\\" ? sg + "\\" : "/" + sg) : acc + (acc.endsWith(sep) ? "" : sep) + sg);
        parts.push(`<span style="color:var(--mut)">&rsaquo;</span><button data-p="${esc(acc)}">${esc(sg)}</button>`);
      });
    }
    $("#crumbs").innerHTML = parts.join("");
    $("#side").innerHTML = (S.info.places || []).map(x => `<button data-p="${esc(x.path)}">${ICON_DIR}<span>${esc(x.name)}</span></button>`).join("");
    
    let rows = [];
    if (d.path && d.parent !== null) rows.push(`<button data-p="${esc(d.parent)}">${ICON_DIR}<span>.. (Up)</span></button>`);
    rows = rows.concat(d.dirs.map(x => `<button data-p="${esc(x.path)}">${ICON_DIR}<span>${esc(x.name)}</span></button>`));
    $("#dirs").innerHTML = rows.length ? rows.join("") : '<div style="color:var(--mut);padding:14px;text-align:center">Empty directory</div>';
    
    $("#mInfo").textContent = d.path ? `${d.videos} video(s), ${d.audio} audio file(s)` : "Pick a drive";
    $("#mPick").disabled = !d.path;
    $$("#crumbs button, #side button, #dirs button").forEach(b => b.onclick = () => loadDir(b.dataset.p));
    $("#dirs").scrollTop = 0;
  } catch(e) { toast(e.message); }
}

$("#mClose").onclick = () => $("#modal").classList.remove("show");
$("#modal").onclick = e => { if (e.target.id === "modal") $("#modal").classList.remove("show"); };
$("#mPick").onclick = () => { $("#modal").classList.remove("show"); if (browseCb) browseCb(browsePath); };

/* ---------- Step 1: Scanning ---------- */
S.scanFilter = "all";
S.typeFilter = "all";

function getRecents() { let r = []; try { r = JSON.parse(localStorage.getItem("wstudio-recents")||"[]"); } catch(e){} return r; }
function saveRecent(p) { let r = getRecents().filter(x=>x!==p); r.unshift(p); try{ localStorage.setItem("wstudio-recents",JSON.stringify(r.slice(0,5))); }catch(e){} }
function renderRecents() {
  const r = getRecents();
  $("#recents").innerHTML = r.length ? r.map(p => `<button title="${esc(p)}" data-p="${esc(p)}">${esc(p)}</button>`).join("") : "";
  $$("#recents button").forEach(b => b.onclick = () => $("#path").value = b.dataset.p);
}

$$(".filter-btn[data-scan]").forEach(b => {
  b.onclick = () => {
    $$(".filter-btn[data-scan]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    S.scanFilter = b.dataset.scan;
  };
});

$("#browseBtn").onclick = () => openBrowser($("#path").value.trim(), p => $("#path").value = p);
$("#quickColabBtn").onclick = () => {
  $("#path").value = "/content/drive/MyDrive/transcribe/videos";
  $("#outdir").value = "/content/drive/MyDrive/transcribe/audio";
  $("#trOutDir").value = "/content/drive/MyDrive/transcribe/transcripts_large_v3";
};

$("#scanBtn").onclick = scanFiles;
async function scanFiles() {
  const p = $("#path").value.trim().replace(/^["']|["']$/g, "");
  if (!p) { toast("Please choose or enter a directory"); return; }
  const b = $("#scanBtn"); b.disabled = true; b.textContent = "Scanning...";
  try {
    const d = await api("scan", { path: p, recursive: $("#recursive").checked, scan_mode: S.scanFilter });
    if (!d.items.length) { toast(`No media files matching '${S.scanFilter}' found in this folder`); return; }
    
    S.root = d.root; S.items = d.items; S.sel = new Set(d.items.map(i => i.id));
    saveRecent(d.root); renderRecents();
    $("#path").value = d.root;
    
    // Set smart output defaults
    const sep = S.info.sep;
    if (S.info.is_colab) {
      if (!$("#outdir").value) $("#outdir").value = "/content/drive/MyDrive/transcribe/audio";
      if (!$("#trOutDir").value) $("#trOutDir").value = "/content/drive/MyDrive/transcribe/transcripts_large_v3";
    } else {
      if (!$("#outdir").value) $("#outdir").value = d.root + (d.root.endsWith(sep) ? "" : sep) + "audio";
      if (!$("#trOutDir").value) $("#trOutDir").value = d.root + (d.root.endsWith(sep) ? "" : sep) + "transcripts_large_v3";
    }

    S.typeFilter = "all";
    $$("#fAll, #fVid, #fAud").forEach(x => x.classList.toggle("active", x.id === "fAll"));
    renderItems();
    S.maxStep = 2; go(2);
  } catch(e) { toast(e.message); }
  finally { b.disabled = false; b.textContent = "Scan Files"; }
}

/* ---------- Step 2: Selection & Media Filters ---------- */
$$("#fAll, #fVid, #fAud").forEach(b => {
  b.onclick = () => {
    $$("#fAll, #fVid, #fAud").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    S.typeFilter = (b.id === "fVid" ? "video" : (b.id === "fAud" ? "audio" : "all"));
    renderItems();
  };
});

function getVisibleItems() {
  const q = $("#q").value.trim().toLowerCase();
  return S.items.filter(i => {
    const matchQ = !q || (i.rel + "/" + i.name).toLowerCase().includes(q);
    const matchType = (S.typeFilter === "all" || i.media_type === S.typeFilter);
    return matchQ && matchType;
  });
}

function renderItems() {
  const vis = getVisibleItems();
  $("#vbody").innerHTML = vis.map(i => {
    const isAud = (i.media_type === "audio");
    const badge = `<span class="tag-media ${isAud ? "aud" : "vid"}">${isAud ? "🎵 " : "🎬 "}${esc((i.ext || "").toUpperCase())}</span>`;
    return `<tr>
      <td><input type="checkbox" class="cb row-cb" data-id="${i.id}" ${S.sel.has(i.id)?"checked":""}></td>
      <td class="n">${i.num == null ? "-" : i.num}</td>
      <td class="nm"><div title="${esc(i.name)}">${badge}${esc(i.name)}${i.cloud?" &#9729;":""}</div>${i.rel?`<small>${esc(i.rel)}</small>`:""}</td>
      <td class="sz">${human(i.size)}</td>
    </tr>`;
  }).join("");

  $$(".row-cb").forEach(cb => cb.onchange = () => {
    const id = +cb.dataset.id;
    cb.checked ? S.sel.add(id) : S.sel.delete(id);
    updateCounts();
  });
  
  $("#visInfo").textContent = vis.length < S.items.length ? `Showing ${vis.length} of ${S.items.length}` : "";
  $("#tFound").textContent = S.items.length;
  $("#tSize").textContent = human(S.items.reduce((a,i) => a + i.size, 0));
  $("#cntAll").textContent = S.items.length;
  $("#cntVid").textContent = S.items.filter(i => i.media_type === "video").length;
  $("#cntAud").textContent = S.items.filter(i => i.media_type === "audio").length;
  updateCounts();
}

function updateCounts() {
  const chosen = S.items.filter(i => S.sel.has(i.id));
  $("#tSel").textContent = chosen.length;
  $("#tSelSize").textContent = human(chosen.reduce((a,i) => a + i.size, 0));
  const vis = getVisibleItems();
  $("#hdrCb").checked = vis.length > 0 && vis.every(i => S.sel.has(i.id));
  $("#to3").disabled = (chosen.length === 0);
}

$("#q").oninput = renderItems;
$("#hdrCb").onchange = e => {
  getVisibleItems().forEach(i => e.target.checked ? S.sel.add(i.id) : S.sel.delete(i.id));
  renderItems();
};
$("#selAll").onclick = () => { S.items.forEach(i => S.sel.add(i.id)); renderItems(); };
$("#selNone").onclick = () => { S.sel.clear(); renderItems(); };
$("#selInv").onclick = () => { S.items.forEach(i => S.sel.has(i.id) ? S.sel.delete(i.id) : S.sel.add(i.id)); renderItems(); };

$("#rApply").onclick = () => {
  const a = $("#rFrom").value === "" ? -Infinity : +$("#rFrom").value;
  const b = $("#rTo").value === "" ? Infinity : +$("#rTo").value;
  if (a === -Infinity && b === Infinity) { toast("Please specify start or end number"); return; }
  S.sel = new Set(S.items.filter(i => i.num != null && i.num >= a && i.num <= b).map(i => i.id));
  renderItems();
  toast(`${S.sel.size} file(s) selected for range [${a === -Infinity ? 0 : a} - ${b === Infinity ? "max" : b}]`, true);
};

$("#to3").onclick = () => { renderPresets(); renderModels(); go(3); };

/* ---------- Step 3: Config & Models ---------- */
function renderPresets() {
  $("#presets").innerHTML = S.info.audio_presets.map(p => `
    <button class="pcard ${p.id === S.preset ? "sel" : ""}" data-id="${p.id}">
      <span class="tick"><svg viewBox="0 0 24 24" fill="none" stroke-width="3.5" stroke-linecap="round"><path d="M5 12l5 5 9-10"/></svg></span>
      <h3>${esc(p.title)}${p.tag ? `<span class="tag">${esc(p.tag)}</span>` : ""}</h3>
      <p>${esc(p.desc)}</p>
      <div class="meta">.${p.ext} &middot; ${p.kbps ? "~" + human(p.kbps*1000/8*3600) + "/hr" : "Lossless"}</div>
    </button>
  `).join("");
  $$("#presets .pcard").forEach(c => c.onclick = () => { S.preset = c.dataset.id; renderPresets(); });
}

function renderModels() {
  $("#models").innerHTML = S.info.whisper_models.map(m => `
    <button class="pcard ${m.id === S.whisperModel ? "sel" : ""}" data-id="${m.id}">
      <span class="tick"><svg viewBox="0 0 24 24" fill="none" stroke-width="3.5" stroke-linecap="round"><path d="M5 12l5 5 9-10"/></svg></span>
      <h3>${esc(m.name)}${m.badge ? `<span class="tag blue">${esc(m.badge)}</span>` : ""}</h3>
      <p>${esc(m.desc)}</p>
      <div class="meta">VRAM: ${m.vram}</div>
    </button>
  `).join("");
  $$("#models .pcard").forEach(c => c.onclick = () => { S.whisperModel = c.dataset.id; renderModels(); });
}

function renderHotwordChips() {
  const tags = ["Kubernetes", "kubectl", "Terraform", "AWS", "Docker", "Jenkins", "Helm", "Ansible", "DevOps", "Naresh IT", "Veera Babu"];
  $("#hotwordChips").innerHTML = tags.map(t => `<button type="button" data-tag="${t}">+ ${t}</button>`).join("");
  $$("#hotwordChips button").forEach(b => b.onclick = () => {
    const cur = $("#hotwords").value.trim();
    const tag = b.dataset.tag;
    if (!cur.includes(tag)) {
      $("#hotwords").value = cur ? cur + ", " + tag : tag;
    }
  });
}
$("#resetHotwordsBtn").onclick = () => $("#hotwords").value = S.info.default_hotwords;
$("#useBatched").onchange = e => $("#batchSizeBox").style.display = e.target.checked ? "flex" : "none";

$("#outBrowse").onclick = () => openBrowser(S.root, p => $("#outdir").value = p);
$("#outReset").onclick = () => $("#outdir").value = S.info.is_colab ? "/content/drive/MyDrive/transcribe/audio" : S.root + "/audio";
$("#trBrowse").onclick = () => openBrowser(S.root, p => $("#trOutDir").value = p);
$("#trReset").onclick = () => $("#trOutDir").value = S.info.is_colab ? "/content/drive/MyDrive/transcribe/transcripts_large_v3" : S.root + "/transcripts";

$("#to4").onclick = runPlan;

/* ---------- Step 4: Dry Run Review ---------- */
async function runPlan() {
  const ids = S.items.filter(i => S.sel.has(i.id)).map(i => i.id);
  go(4);
  $("#planLoad").style.display = ""; $("#planBody").style.display = "none"; $("#startBtn").disabled = true;
  $("#planMsg").textContent = "Inspecting media streams & checking transcripts...";
  $("#planBar").style.width = "0";

  try {
    const r = await api("plan", {
      ids,
      preset: S.preset,
      out_dir: $("#outdir").value.trim(),
      transcripts_dir: (S.mode !== "v2a") ? $("#trOutDir").value.trim() : null,
      overwrite: $("#overwrite").checked,
      probe: true,
      mode: S.mode
    });
    S.planId = r.plan_id;
    for (;;) {
      await sleep(300);
      const st = await api("plan_status");
      if (st.id !== S.planId) return;
      if (st.state === "running") {
        $("#planBar").style.width = (st.total ? (st.done / st.total * 100) : 10) + "%";
        if (st.total) $("#planMsg").textContent = `Probing media files... ${st.done}/${st.total}`;
      } else if (st.state === "error") {
        throw new Error(st.error);
      } else {
        renderPlanResult(st.result);
        return;
      }
    }
  } catch(e) {
    toast(e.message);
    go(3);
  }
}

function renderPlanResult(r) {
  S.plan = r; const s = r.summary;
  $("#planLoad").style.display = "none"; $("#planBody").style.display = "";
  $("#planTiles").innerHTML = `
    <div class="tile ok"><small>Ready to Process</small><b>${s.todo}</b></div>
    <div class="tile acc"><small>Direct Audio (No Convert)</small><b>${s.direct_audio || 0}</b></div>
    <div class="tile"><small>Videos to Extract</small><b>${s.video_convert || 0}</b></div>
    <div class="tile"><small>Already Completed</small><b>${s.skipped}</b></div>
    <div class="tile"><small>Total Duration</small><b>${s.durations_known && s.duration ? hms(s.duration) : "--"}</b></div>
  `;
  const warns = r.warnings || r.warns || [];
  $("#planWarns").innerHTML = warns.map(w => `<div class="warn"><span>&#9888;</span><span>${esc(w)}</span></div>`).join("");
  
  const statusLabels = {
    direct: "⚡ Direct Transcribe",
    convert: "🎬 Extract & Transcribe",
    ready: "Ready",
    skip: "Skipped (Exists)",
    done_transcript: "✓ Done (.txt Exists)",
    noaudio: "⚠️ No Audio Track"
  };

  const items = r.items || [];
  $("#planRows").innerHTML = items.map(i => `<tr>
    <td class="n">${i.num == null ? "-" : i.num}</td>
    <td class="nm"><div title="${esc(i.name)}">${esc(i.name)}${i.cloud ? " &#9729;" : ""}</div>
      <small>${esc(i.out)}${i.note ? " &middot; " + esc(i.note) : ""}</small></td>
    <td class="sz">${human(i.size)}</td>
    <td class="sz">${i.duration ? hms(i.duration) : "--"}</td>
    <td><span class="pill ${i.status}">${statusLabels[i.status] || i.status}</span></td>
  </tr>`).join("");

  const b = $("#startBtn");
  b.disabled = (s.todo === 0);
  b.textContent = s.todo ? `Start Processing (${s.todo} file${s.todo > 1 ? "s" : ""})` : "All files already completed";
}

$("#startBtn").onclick = async () => {
  try {
    $("#startBtn").disabled = true;
    const ringEl = $("#ring");
    ringEl.classList.remove("done");
    ringEl.classList.add("running");
    $("#ringFg").style.strokeDashoffset = "339.292";
    $("#pct").textContent = "0";
    $("#pcTag").textContent = "STARTING";
    $("#pcSub").textContent = "Preparing";
    if ($("#stageText")) $("#stageText").textContent = "Launching job...";
    if ($("#stageBadge")) { $("#stageBadge").textContent = "STARTING"; $("#stageBadge").className = "stage-badge run"; }

    await api("start", {
      plan_id: S.planId,
      mode: S.mode,
      transcripts_dir: $("#trOutDir").value.trim(),
      overwrite: $("#overwrite").checked,
      whisper_model: S.whisperModel,
      compute_type: $("#computeType").value,
      language: $("#language").value,
      hotwords: $("#hotwords").value.trim(),
      use_batched: $("#useBatched").checked,
      batch_size: +$("#batchSize").value,
      para_seconds: +$("#paraSeconds").value,
      max_hours: +$("#maxHours").value,
    });
    S.running = true;
    startPolling();
    go(5);
  } catch(e) {
    $("#startBtn").disabled = false;
    $("#ring").classList.remove("running");
    toast(e.message);
  }
};

/* ---------- Step 5: Execution Monitoring ---------- */
let pollFailures = 0;
let pollingActive = false;
let pollTimer = null;

function startPolling() {
  stopPolling();
  pollFailures = 0;
  S.running = true;
  pollTimer = setInterval(pollStatus, 500);
  pollStatus();
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (S.timer) {
    clearInterval(S.timer);
    S.timer = null;
  }
}

async function pollStatus() {
  if (pollingActive) return;
  pollingActive = true;
  try {
    const st = await api("status");
    pollFailures = 0;
    try {
      renderDashboard(st);
    } catch(renderErr) {
      console.error("renderDashboard error:", renderErr);
    }
    if (st.state !== "running" && st.state !== "idle") {
      stopPolling();
      S.running = false;
      finishDashboard(st);
    }
  } catch(e) {
    pollFailures++;
    console.warn("Poll status failed (" + pollFailures + "):", e.message);
    if (pollFailures >= 40) {
      stopPolling();
      S.running = false;
      toast("Lost connection to studio server. Please check your notebook.");
      const ringEl = $("#ring");
      if (ringEl) ringEl.classList.remove("running");
    }
  } finally {
    pollingActive = false;
  }
}

function renderDashboard(st) {
  if (!st) return;
  const items = st.items || [];
  const ringEl = $("#ring");

  // 1. Ring state classes for continuous active motion
  if (st.state === "running") {
    ringEl.classList.add("running");
    ringEl.classList.remove("done");
  } else if (st.state === "finished" || st.state === "done") {
    ringEl.classList.remove("running");
    ringEl.classList.add("done");
  } else {
    ringEl.classList.remove("running");
    ringEl.classList.remove("done");
  }

  // 2. Numeric percentage & SVG stroke offset
  const frac = Math.max(0, Math.min(1, Number(st.fraction) || 0));
  const pct = Math.round(frac * 100);
  $("#pct").textContent = pct;
  $("#ringFg").style.strokeDashoffset = (339.292 * (1 - frac)).toFixed(2);

  // 3. Tag (Upper tracking label inside circle)
  let tagText = "PROGRESS";
  if (st.state === "running") {
    const stageLower = (st.stage || "").toLowerCase();
    if (stageLower.includes("loading") || stageLower.includes("init")) {
      tagText = "LOADING";
    } else if (stageLower.includes("extracting")) {
      tagText = "EXTRACT";
    } else if (stageLower.includes("transcribing")) {
      tagText = "TRANSCRIBE";
    } else {
      tagText = "RUNNING";
    }
  } else if (st.state === "finished") {
    tagText = "COMPLETE";
  } else if (st.state === "cancelled") {
    tagText = "STOPPED";
  }
  $("#pcTag").textContent = tagText;

  // 4. Sub-label inside circle (Always strictly bounded and aesthetic)
  const done = items.filter(i => i.status === "done" || i.status === "failed").length;
  if (items.length > 0) {
    if (st.state === "running") {
      $("#pcSub").textContent = `${done} of ${items.length} done`;
    } else if (st.state === "finished") {
      $("#pcSub").textContent = `${done}/${items.length} finished`;
    } else {
      $("#pcSub").textContent = `${done}/${items.length} files`;
    }
  } else {
    $("#pcSub").textContent = st.state === "running" ? "Active" : st.state;
  }

  // 5. Active Stage Banner outside the circle (Displays full details without disrupting the ring)
  if ($("#stageText")) {
    $("#stageText").textContent = st.stage || (st.state === "running" ? "Processing..." : st.state);
  }
  if ($("#stageBadge")) {
    if (st.state === "running") {
      const sLow = (st.stage || "").toLowerCase();
      $("#stageBadge").textContent = sLow.includes("loading") ? "LOADING MODEL" : "RUNNING";
      $("#stageBadge").className = "stage-badge run";
    } else if (st.state === "finished") {
      $("#stageBadge").textContent = "FINISHED";
      $("#stageBadge").className = "stage-badge ok";
    } else {
      $("#stageBadge").textContent = (st.state || "IDLE").toUpperCase();
      $("#stageBadge").className = "stage-badge warn";
    }
  }
  if ($("#stageModeText")) {
    const modeMap = {
      pipeline: "Pipeline Mode (Audio + Whisper)",
      v2a: "Video \u2192 Audio Extraction",
      whisper: "Whisper Transcribe Only"
    };
    $("#stageModeText").textContent = modeMap[st.mode] || (st.mode ? st.mode.toUpperCase() : "");
  }

  // 6. Metrics Tiles
  $("#sDone").textContent = items.length ? `${done}/${items.length}` : "0/0";
  $("#sEta").textContent = st.state === "running" ? (st.eta != null ? hms(st.eta) : "...") : "0:00";
  $("#sEl").textContent = hms(st.elapsed);
  $("#sX").textContent = st.x ? (st.x.toFixed(1) + "x") : (st.speed || "--");
  $("#sOut").textContent = human(st.out_size);

  // 7. Active Current Item Bar
  const cur = items.find(i => i.status === "running");
  if (cur) {
    $("#curName").textContent = cur.name;
    $("#curBar").className = "bar" + (cur.duration ? "" : " ind");
    $("#curFill").style.width = (cur.duration ? Math.min(100, (cur.progress * 100)) : 35) + "%";
    $("#curInfo").textContent = (cur.speed || "") + " " + hms(cur.secs) + (cur.duration ? " / " + hms(cur.duration) : "");
  } else {
    $("#curName").textContent = st.state === "running" ? "Starting next file..." : "Complete";
    $("#curBar").className = "bar idle";
    $("#curFill").style.width = st.state === "running" ? "0%" : "100%";
    $("#curInfo").textContent = "";
  }

  // 8. Live Speech Segment Streamer
  if (st.live_segment && st.live_segment.text) {
    $("#liveSpeechBox").style.display = "flex";
    $("#liveTs").textContent = st.live_segment.ts || "";
    $("#liveText").textContent = '"' + st.live_segment.text + '"';
  }

  // 9. Queue List
  $("#queue").innerHTML = items.map(i => `
    <div class="qi ${i.status}" data-id="${i.id}">
      <span class="dot"></span>
      <span class="nm" title="${esc(i.name)}">${esc(i.name)}</span>
      <span class="st">${i.status === "running" ? Math.round((i.progress || 0)*100) + "%" : (i.status === "done" ? human(i.out_size || 0) : i.status)}</span>
    </div>
  `).join("");

  // 10. Logs
  if (st.log && Array.isArray(st.log)) {
    $("#logBox").textContent = st.log.join("\n");
    $("#logBox").scrollTop = $("#logBox").scrollHeight;
  }
}

function finishDashboard(st) {
  stopPolling();
  $("#cancelBtn").style.display = "none";
  $("#openTranscriptsBtn").style.display = "inline-flex";
  $("#runAgainBtn").style.display = "inline-flex";
  const ringEl = $("#ring");
  ringEl.classList.remove("running");
  ringEl.classList.add("done");
  $("#ringFg").style.strokeDashoffset = "0";
  $("#pct").textContent = "100";
  $("#pcTag").textContent = "COMPLETE";
  const items = st.items || [];
  const done = items.filter(i => i.status === "done").length;
  $("#pcSub").textContent = `${done}/${items.length} finished`;
  if ($("#stageBanner")) $("#stageBanner").classList.add("done");
  if ($("#stageBadge")) {
    $("#stageBadge").textContent = "COMPLETED";
    $("#stageBadge").className = "stage-badge ok";
  }
  if ($("#stageText")) {
    $("#stageText").textContent = st.failed === 0 ? "Batch processing finished successfully!" : `Batch completed with ${st.failed} issue(s).`;
  }
  if (st.state === "finished" && (st.failed || 0) === 0) {
    launchConfetti();
    toast("Batch processing finished successfully!", true);
  }
}

$("#cancelBtn").onclick = async () => {
  if (!confirm("Stop processing? Finished files will be preserved.")) return;
  $("#cancelBtn").disabled = true;
  if ($("#stageText")) $("#stageText").textContent = "Stopping batch...";
  try {
    await api("cancel", {});
  } catch(e) {
    toast("Cancel failed: " + e.message);
  }
};

$("#runAgainBtn").onclick = () => {
  stopPolling();
  S.running = false;
  const ringEl = $("#ring");
  ringEl.classList.remove("running", "done");
  $("#cancelBtn").style.display = "inline-flex";
  $("#cancelBtn").disabled = false;
  $("#openTranscriptsBtn").style.display = "none";
  $("#runAgainBtn").style.display = "none";
  S.maxStep = 2;
  go(2);
};
$("#openTranscriptsBtn").onclick = () => setMode("viewer");

/* ---------- Transcripts Studio Viewer ---------- */
async function loadTranscripts() {
  const dir = S.info?.is_colab ? "/content/drive/MyDrive/transcribe/transcripts_large_v3" : (S.plan?.out_dir || "");
  try {
    const r = await api("transcripts?dir=" + encodeURIComponent(dir));
    if (!r.files || !r.files.length) {
      $("#trFileList").innerHTML = '<div style="color:var(--mut);padding:20px;text-align:center">No transcripts found yet</div>';
      return;
    }
    const stems = [...new Set(r.files.map(f => f.stem))];
    $("#trFileList").innerHTML = stems.map((st, i) => `
      <div class="tr-item ${i === 0 ? "active" : ""}" data-stem="${esc(st)}">
        <b>${esc(st)}</b>
        <small>.md &middot; .srt &middot; .txt</small>
      </div>
    `).join("");

    $$(".tr-item").forEach(item => item.onclick = () => {
      $$(".tr-item").forEach(x => x.classList.remove("active"));
      item.classList.add("active");
      S.activeTranscriptStem = item.dataset.stem;
      showTranscript();
    });

    if (stems.length) {
      S.activeTranscriptStem = stems[0];
      showTranscript();
    }
  } catch(e) { toast(e.message); }
}

$$(".tr-tab").forEach(tab => tab.onclick = () => {
  $$(".tr-tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  S.activeTranscriptExt = tab.dataset.view;
  showTranscript();
});

async function showTranscript() {
  if (!S.activeTranscriptStem) return;
  const fileName = `${S.activeTranscriptStem}.${S.activeTranscriptExt}`;
  const dir = S.info?.is_colab ? "/content/drive/MyDrive/transcribe/transcripts_large_v3" : (S.plan?.out_dir || "");
  const view = $("#trBodyView");
  view.innerHTML = '<div style="color:var(--mut);text-align:center;padding:30px">Loading transcript...</div>';
  try {
    const r = await api(`transcripts?dir=${encodeURIComponent(dir)}&file=${encodeURIComponent(fileName)}`);
    if (r.content != null) {
      if (S.activeTranscriptExt === "md") {
        // Render markdown with nice study paragraphs
        let mdHtml = esc(r.content)
          .replace(/^# (.*$)/gm, '<h2 style="font-size:18px;margin:0 0 10px;color:var(--acc2)">$1</h2>')
          .replace(/\*\*\[(\d{2}:\d{2}(?::\d{2})?)\]\*\*/g, '<span class="pill ready" style="margin-right:6px">[$1]</span>')
          .replace(/\n\n/g, '<p style="margin:0 0 14px"></p>');
        view.innerHTML = `<div style="max-width:800px;margin:0 auto">${mdHtml}</div>`;
      } else {
        view.innerHTML = `<pre>${esc(r.content)}</pre>`;
      }
    } else {
      view.innerHTML = `<div style="color:var(--warn);padding:20px;text-align:center">File not found: ${fileName}</div>`;
    }
  } catch(e) { view.innerHTML = `<div style="color:var(--bad)">${e.message}</div>`; }
}

$("#refreshTrBtn").onclick = loadTranscripts;
$("#copyTrBtn").onclick = () => {
  const text = $("#trBodyView").innerText;
  if (!text) return;
  navigator.clipboard.writeText(text);
  toast("Transcript copied to clipboard!", true);
};

/* ---------- Confetti Animation ---------- */
function launchConfetti() {
  const box = $("#confetti"), cols = ["#7c5cff","#22d3ee","#ec4899","#10b981","#f59e0b"];
  for (let i = 0; i < 75; i++) {
    const e = document.createElement("i");
    e.style.left = Math.random() * 100 + "%";
    e.style.background = cols[i % cols.length];
    e.style.animationDuration = (2 + Math.random() * 2.2) + "s";
    e.style.animationDelay = (Math.random() * 0.5) + "s";
    box.appendChild(e);
  }
  setTimeout(() => box.innerHTML = "", 5000);
}

/* ---------- App Initialization ---------- */
(async function init() {
  try {
    const t = localStorage.getItem("wstudio-theme");
    if (t) setTheme(t);
  } catch(e){}

  try {
    S.info = await api("info");
    updateHardware();
    renderRecents();
    renderHotwordChips();
    $("#hotwords").value = S.info.default_hotwords;

    if (S.info.is_colab) {
      $("#quickColabBtn").style.display = "inline-flex";
      $("#path").value = "/content/drive/MyDrive/transcribe/videos";
      $("#outdir").value = "/content/drive/MyDrive/transcribe/audio";
      $("#trOutDir").value = "/content/drive/MyDrive/transcribe/transcripts_large_v3";
    } else {
      const r = getRecents();
      $("#path").value = r[0] || S.info.cwd;
    }

    renderPresets();
    renderModels();

    if (S.info.scan_items && S.info.scan_items.length) {
      S.items = S.info.scan_items;
      S.sel = new Set(S.items.map(i => i.id));
      renderItems();
    }
    if (S.info.plan) {
      renderPlanResult(S.info.plan);
    }

    const urlParams = new URLSearchParams(window.location.search);
    const reqStep = urlParams.get("step");
    const reqMode = urlParams.get("mode");

    if (reqMode === "viewer") {
      setMode("viewer");
      return;
    } else if (reqMode) {
      setMode(reqMode);
    }

    const st = await api("status");
    if (st.items && st.items.length && (st.state === "running" || st.state === "finished")) {
      S.maxStep = 5;
      S.running = (st.state === "running");
      renderDashboard(st);
      go(reqStep ? +reqStep : 5);
      if (S.running) startPolling(); else finishDashboard(st);
    } else if (reqStep) {
      S.maxStep = Math.max(S.maxStep, +reqStep);
      go(+reqStep);
    } else {
      go(1);
    }
  } catch(e) {
    go(1);
    toast(e.message);
  }
})();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------- Colab Launcher & CLI
def run_server(port: int = 8765, open_browser: bool = True):
    global SERVER
    host = "127.0.0.1" if not is_colab() else "0.0.0.0"

    for p in range(port, port + 20):
        try:
            SERVER = ThreadingHTTPServer((host, p), StudioHandler)
            break
        except OSError:
            continue
    else:
        print(f"Could not bind to any port in range {port}-{port+20}")
        sys.exit(1)

    actual_port = SERVER.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"\n=======================================================")
    print(f"  Whisper & Video Studio is LIVE at: {url}")
    print(f"  Colab Environment: {'YES' if is_colab() else 'NO (Local)'}")
    print(f"=======================================================\n")

    if open_browser and not is_colab():
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    return actual_port

def launch_in_colab(port: int = 8765, height: int = 950):
    """Entry point for Google Colab notebooks."""
    actual_port = run_server(port=port, open_browser=False)
    
    # Run server loop in background thread
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()
    time.sleep(0.5)

    if is_colab():
        from google.colab import output
        print("Rendering interactive Studio inside your Google Colab notebook cell...")
        print("Tip: You can also open the Studio in a separate browser tab below:")
        try:
            output.serve_kernel_port_as_window(actual_port)
        except Exception:
            pass
        return output.serve_kernel_port_as_iframe(actual_port, height=height)
    else:
        print(f"Open your browser at: http://127.0.0.1:{actual_port}/")

def main():
    ap = argparse.ArgumentParser(description="Whisper & Video Studio Web App")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="Do not launch browser automatically")
    args = ap.parse_args()

    actual_port = run_server(port=args.port, open_browser=not args.no_browser)
    try:
        SERVER.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        api_cancel({})
        SERVER.server_close()
        print("\nStudio server stopped.")

if __name__ == "__main__":
    main()
