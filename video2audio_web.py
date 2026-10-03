#!/usr/bin/env python3
"""
video2audio_web.py - a modern local web app that converts videos to audio with ffmpeg.

    python video2audio_web.py

It starts a small server on your own computer (127.0.0.1 only) and opens the app in your
browser. Your videos never leave your machine and nothing needs to be uploaded: the app
reads the folder you choose directly from disk.

Needs: Python 3.8+ and ffmpeg (ffmpeg + ffprobe on PATH). No other packages.
"""
from __future__ import annotations

import argparse
import json
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

# --------------------------------------------------------------------------- config
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".m4v", ".ts",
             ".mts", ".m2ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob"}
CLOUD_MASK = 0x400000 | 0x40000 | 0x1000      # Windows online-only (OneDrive etc.) attributes
MAX_ITEMS = 5000
PROGRESS_KEYS = {"frame", "fps", "bitrate", "total_size", "out_time_us", "out_time_ms", "out_time",
                 "dup_frames", "drop_frames", "speed", "progress"}

PRESETS = [
    dict(id="1", title="MP3 \u00b7 Speech", ext="mp3", fmt="mp3", kbps=32,
         desc="Mono, 16 kHz, 32 kbps. Tiny files, ideal for Whisper and transcripts.",
         tag="Best for transcription", args=["-ac", "1", "-ar", "16000", "-b:a", "32k"]),
    dict(id="2", title="MP3 \u00b7 Music", ext="mp3", fmt="mp3", kbps=192,
         desc="Stereo, 192 kbps. Good all-round quality.", tag="", args=["-b:a", "192k"]),
    dict(id="3", title="M4A \u00b7 AAC", ext="m4a", fmt="ipod", kbps=128,
         desc="Stereo, 128 kbps. Efficient and plays everywhere.", tag="",
         args=["-c:a", "aac", "-b:a", "128k"]),
    dict(id="4", title="WAV \u00b7 Whisper-ready", ext="wav", fmt="wav", kbps=256,
         desc="Mono, 16 kHz, 16-bit. Uncompressed, so files are large.", tag="",
         args=["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]),
    dict(id="5", title="FLAC \u00b7 Lossless", ext="flac", fmt="flac", kbps=None,
         desc="Original quality, usually half the size of WAV.", tag="Lossless",
         args=["-c:a", "flac"]),
    dict(id="6", title="Opus \u00b7 Tiny speech", ext="opus", fmt="ogg", kbps=24,
         desc="Mono, 24 kbps. The smallest files for voice.", tag="Smallest",
         args=["-ac", "1", "-c:a", "libopus", "-b:a", "24k"]),
]

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
TOKEN = secrets.token_urlsafe(18)
LOCK = threading.Lock()
SERVER = None


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


# --------------------------------------------------------------------------- helpers
KEYWORD_RE = re.compile(r"\b(?:session|day|class|lecture|lesson|part|episode|ep|video|module|chapter|week)"
                        r"[\s._#-]*(\d+)", re.I)


def file_number(stem):
    m = KEYWORD_RE.search(stem)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else None


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def item_key(it):
    return (it["num"] is None, it["num"] or 0, natural(os.path.join(it["rel"], it["name"])))


def probe(path):
    """(duration_seconds | None, has_audio | None)"""
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                            "-of", "json", str(path)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        data = json.loads(r.stdout or "{}")
        dur = float(data.get("format", {}).get("duration") or 0) or None
        return dur, any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    except Exception:
        return None, None


def ffmpeg_version():
    try:
        out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=15).stdout
        return out.splitlines()[0].split(" Copyright")[0]
    except Exception:
        return "ffmpeg"


def preset_by_id(pid):
    return next((p for p in PRESETS if p["id"] == str(pid)), None)


def places():
    home = Path.home()
    cands = [("Home", home), ("Desktop", home / "Desktop"), ("Desktop (OneDrive)", home / "OneDrive" / "Desktop"),
             ("Videos", home / "Videos"), ("Movies", home / "Movies"), ("Downloads", home / "Downloads"),
             ("Documents", home / "Documents"), ("Current folder", Path.cwd())]
    seen, out = set(), []
    for name, p in cands:
        try:
            if p.is_dir() and str(p) not in seen:
                seen.add(str(p))
                out.append({"name": name, "path": str(p)})
        except OSError:
            pass
    return out


def roots():
    if os.name == "nt":
        return [{"name": f"{c}:\\", "path": f"{c}:\\"} for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    return [{"name": "/", "path": "/"}]


# --------------------------------------------------------------------------- state
SCAN = {"root": None, "items": []}
PLAN = {"id": None, "state": "idle", "done": 0, "total": 0, "result": None, "error": None,
        "jobs": [], "preset": None, "out_dir": None}
RUN = {"state": "idle", "items": [], "log": [], "out_dir": "", "started": 0.0, "ended": 0.0,
       "cancel": False, "proc": None, "current": None}


def log(msg):
    with LOCK:
        RUN["log"].append(f"{datetime.now():%H:%M:%S}  {msg}")
        del RUN["log"][:-300]


# --------------------------------------------------------------------------- API: info / folders / scan
def api_info(_q):
    return {"ffmpeg": ffmpeg_version(), "presets": [{k: v for k, v in p.items() if k not in ("args", "fmt")}
                                                      for p in PRESETS],
            "home": str(Path.home()), "cwd": str(Path.cwd()), "sep": os.sep,
            "places": places()}


def api_fs(q):
    raw = (q.get("path") or [""])[0]
    if not raw:
        return {"path": "", "parent": None, "dirs": roots(), "videos": 0, "places": places()}
    d = Path(raw).expanduser()
    if not d.is_dir():
        raise ApiError(404, "That folder does not exist")
    dirs, vids = [], 0
    try:
        with os.scandir(d) as it:
            for e in it:
                try:
                    if e.name.startswith("."):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        dirs.append({"name": e.name, "path": os.path.join(str(d), e.name)})
                    elif os.path.splitext(e.name)[1].lower() in VIDEO_EXT:
                        vids += 1
                except OSError:
                    pass
    except PermissionError:
        raise ApiError(403, "No permission to open this folder")
    dirs.sort(key=lambda x: natural(x["name"]))
    parent = str(d.parent) if d.parent != d else ("" if os.name == "nt" else None)
    return {"path": str(d), "parent": parent, "dirs": dirs, "videos": vids}


def api_scan(body):
    d = Path(str(body.get("path", ""))).expanduser()
    if not d.is_dir():
        raise ApiError(404, "That folder does not exist")
    recursive = bool(body.get("recursive"))
    items = []

    def add(dirpath, name):
        full = os.path.join(dirpath, name)
        st = os.stat(full)
        rel = os.path.relpath(dirpath, d)
        items.append(dict(path=full, name=name, rel="" if rel == "." else rel, size=st.st_size,
                          num=file_number(os.path.splitext(name)[0]),
                          cloud=bool(getattr(st, "st_file_attributes", 0) & CLOUD_MASK),
                          duration=None, has_audio=None))

    try:
        if recursive:
            for dirpath, dirnames, filenames in os.walk(d):
                dirnames[:] = [x for x in dirnames if not x.startswith(".")]
                for name in filenames:
                    if os.path.splitext(name)[1].lower() in VIDEO_EXT and not name.startswith("._"):
                        try:
                            add(dirpath, name)
                        except OSError:
                            pass
                if len(items) >= MAX_ITEMS:
                    break
        else:
            for name in os.listdir(d):
                if os.path.isfile(os.path.join(d, name)) and os.path.splitext(name)[1].lower() in VIDEO_EXT \
                        and not name.startswith("._"):
                    try:
                        add(str(d), name)
                    except OSError:
                        pass
    except PermissionError:
        raise ApiError(403, "No permission to read this folder")

    items.sort(key=item_key)
    items = items[:MAX_ITEMS]
    for i, it in enumerate(items):
        it["id"] = i
    with LOCK:
        SCAN["root"], SCAN["items"] = str(d), items
    return {"root": str(d), "total_size": sum(i["size"] for i in items),
            "items": [{k: it[k] for k in ("id", "name", "rel", "size", "num", "cloud")} for it in items]}


# --------------------------------------------------------------------------- API: plan (dry run)
def build_plan_thread(plan_id, jobs, preset, out_dir, overwrite, do_probe):
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
            for j in jobs:                      # remember results for next time
                src = SCAN["items"][j["id"]] if j["id"] < len(SCAN["items"]) else None
                if src is not None and src["path"] == j["path"]:
                    src["duration"], src["has_audio"] = j["duration"], j["has_audio"]

        used, warnings = set(), []
        for j in jobs:
            out = out_dir / j["rel"] / (os.path.splitext(j["name"])[0] + "." + preset["ext"])
            if str(out).lower() in used:
                ext = os.path.splitext(j["name"])[1][1:].lower()
                out = out.with_name(f"{os.path.splitext(j['name'])[0]}_{ext}.{preset['ext']}")
                j["note"] = "renamed (name clash)"
            used.add(str(out).lower())
            j["out"] = str(out)
            if j["has_audio"] is False:
                j["status"] = "noaudio"
            elif out.exists() and out.stat().st_size > 0 and not overwrite:
                j["status"] = "skip"
            else:
                j["status"] = "convert"

        todo = [j for j in jobs if j["status"] == "convert"]
        nums = sorted({j["num"] for j in jobs if j["num"] is not None})
        if len(nums) > 1:
            have = set(nums)
            gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in have]
            if gaps:
                warnings.append("No video found for number(s): " + ", ".join(map(str, gaps[:15]))
                                + (" ..." if len(gaps) > 15 else ""))
        by_num = {}
        for j in jobs:
            if j["num"] is not None:
                by_num.setdefault(j["num"], []).append(j)
        for n, js in by_num.items():
            if len(js) > 1:
                warnings.append(f"{len(js)} files share the number {n}")
        if any(j["status"] == "noaudio" for j in jobs):
            warnings.append("Some videos have no audio track and will be skipped")
        if any(j.get("note") for j in jobs):
            warnings.append("Some output names were changed because two videos have the same name")
        if any(j["cloud"] for j in todo):
            warnings.append("Some files are online-only (OneDrive etc.). Converting will download them first")

        total_dur = sum(j["duration"] or 0 for j in todo)
        est = preset["kbps"] * 1000 / 8 * total_dur if preset["kbps"] and total_dur else None
        probe_dir = out_dir
        while not probe_dir.exists() and probe_dir != probe_dir.parent:
            probe_dir = probe_dir.parent
        free = shutil.disk_usage(probe_dir).free
        if est and free < est * 1.2:
            warnings.append("Low disk space for the output folder")

        result = {
            "items": [dict({k: j.get(k) for k in ("id", "name", "rel", "size", "num", "cloud", "duration", "status", "note")},
                           out=os.path.relpath(j["out"], out_dir)) for j in jobs],
            "warnings": warnings,
            "summary": {"todo": len(todo), "skipped": sum(1 for j in jobs if j["status"] == "skip"),
                        "noaudio": sum(1 for j in jobs if j["status"] == "noaudio"),
                        "duration": total_dur, "est_bytes": est, "free_bytes": free,
                        "out_dir": str(out_dir), "durations_known": all(j["duration"] for j in todo) if todo else True},
        }
        with LOCK:
            if PLAN["id"] == plan_id:
                PLAN.update(state="done", result=result, jobs=jobs, preset=preset, out_dir=str(out_dir))
    except Exception as e:                      # noqa: BLE001
        with LOCK:
            if PLAN["id"] == plan_id:
                PLAN.update(state="error", error=f"{type(e).__name__}: {e}")


def api_plan(body):
    ids = set(body.get("ids") or [])
    if not ids:
        raise ApiError(400, "Select at least one video first")
    preset = preset_by_id(body.get("preset"))
    if not preset:
        raise ApiError(400, "Unknown audio format")
    if not SCAN["root"]:
        raise ApiError(400, "Scan a folder first")
    root = Path(SCAN["root"])
    out_dir = Path(str(body.get("out_dir") or "")).expanduser() if body.get("out_dir") else root / "audio"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    jobs = [dict(it) for it in SCAN["items"] if it["id"] in ids]
    plan_id = secrets.token_hex(6)
    with LOCK:
        if RUN["state"] == "running":
            raise ApiError(409, "A conversion is running right now")
        PLAN.update(id=plan_id, state="running", done=0, total=0, result=None, error=None, jobs=[],
                    preset=None, out_dir=str(out_dir))
    threading.Thread(target=build_plan_thread, daemon=True,
                     args=(plan_id, jobs, preset, out_dir, bool(body.get("overwrite")), bool(body.get("probe", True)))).start()
    return {"plan_id": plan_id}


def api_plan_status(_q):
    with LOCK:
        return {"id": PLAN["id"], "state": PLAN["state"], "done": PLAN["done"], "total": PLAN["total"],
                "error": PLAN["error"], "result": PLAN["result"] if PLAN["state"] == "done" else None}


# --------------------------------------------------------------------------- API: run
def convert_one(job, preset, it):
    tmp = Path(job["out"] + ".part")
    Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
    if job["duration"] is None:
        job["duration"], _ = probe(job["path"])
    dur = job["duration"]
    it["duration"] = dur
    cmd = [FFMPEG, "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
           "-i", job["path"], "-vn", "-map", "0:a:0", *preset["args"], "-f", preset["fmt"], str(tmp)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
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


def run_worker(jobs, preset, out_dir):
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.rglob("*.part"):
            stale.unlink(missing_ok=True)
    except OSError as e:
        log(f"Cannot use the output folder: {e}")
        with LOCK:
            RUN.update(state="finished", ended=time.time())
        return
    log(f"Started: {len(jobs)} file(s) to {preset['ext'].upper()}")
    logfile = out_dir / "convert_log.txt"
    for job, it in zip(jobs, RUN["items"]):
        if RUN["cancel"]:
            break
        it["status"], RUN["current"] = "running", it["id"]
        try:
            status, err = convert_one(job, preset, it)
        except Exception as e:                  # noqa: BLE001
            status, err = "failed", f"{type(e).__name__}: {e}"
        it["status"], it["err"] = status, err
        if status == "done":
            it["progress"], it["secs"] = 1.0, it["duration"] or it.get("secs", 0)
            try:
                it["out_size"] = os.path.getsize(job["out"])
            except OSError:
                pass
            log(f"Done: {job['name']}")
        elif status == "failed":
            log(f"FAILED: {job['name']} - {err}")
        try:
            with open(logfile, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {status.upper():9} {job['path']} {err}\n")
        except OSError:
            pass
    with LOCK:
        for it in RUN["items"]:
            if it["status"] == "queued":
                it["status"] = "cancelled"
        RUN.update(state="cancelled" if RUN["cancel"] else "finished", ended=time.time(), current=None)
    log("Cancelled" if RUN["cancel"] else "All finished")


def api_start(body):
    with LOCK:
        if RUN["state"] == "running":
            raise ApiError(409, "A conversion is already running")
        if PLAN["id"] != body.get("plan_id") or PLAN["state"] != "done":
            raise ApiError(400, "Please review the plan first")
        jobs = [j for j in PLAN["jobs"] if j["status"] == "convert"]
        if not jobs:
            raise ApiError(400, "Nothing to convert")
        preset, out_dir = PLAN["preset"], PLAN["out_dir"]
        RUN.update(state="running", log=[], out_dir=out_dir, started=time.time(), ended=0.0, cancel=False,
                   current=None, items=[dict(id=j["id"], name=j["name"], status="queued", progress=0.0, secs=0.0,
                                             duration=j["duration"], speed="", err="", out_size=0) for j in jobs])
    threading.Thread(target=run_worker, args=(jobs, preset, out_dir), daemon=True).start()
    return {"ok": True}


def api_status(_q):
    with LOCK:
        items = [dict(i) for i in RUN["items"]]
        state = RUN["state"]
        started = RUN["started"]
        elapsed = (RUN["ended"] or time.time()) - started if started else 0
        total_dur = sum(i["duration"] or 0 for i in items)
        known = items and all(i["duration"] for i in items)
        if known:
            frac = sum((i["duration"] if i["status"] == "done" else i["secs"]) for i in items) / total_dur
        elif items:
            frac = sum(1.0 if i["status"] == "done" else i["progress"] for i in items) / len(items)
        else:
            frac = 0.0
        processed = sum((i["duration"] or 0) if i["status"] == "done" else i["secs"] for i in items)
        eta = elapsed * (1 - frac) / frac if state == "running" and frac > 0.01 else None
        return {"state": state, "items": items, "fraction": min(frac, 1.0), "elapsed": elapsed, "eta": eta,
                "x": (processed / elapsed) if elapsed > 2 and processed else None,
                "ok": sum(1 for i in items if i["status"] == "done"),
                "failed": sum(1 for i in items if i["status"] == "failed"),
                "out_dir": RUN["out_dir"], "current": RUN["current"], "log": RUN["log"][-60:],
                "out_size": sum(i["out_size"] for i in items)}


def api_cancel(_body):
    with LOCK:
        RUN["cancel"] = True
        proc = RUN["proc"]
    if proc:
        try:
            proc.kill()
        except Exception:                       # noqa: BLE001
            pass
    return {"ok": True}


def api_open(body):
    p = str(body.get("path", ""))
    if p not in (RUN["out_dir"], PLAN["out_dir"]) or not os.path.isdir(p):
        raise ApiError(400, "Folder not available")
    if os.name == "nt":
        os.startfile(p)                         # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])
    return {"ok": True}


def api_quit(_body):
    api_cancel({})
    threading.Thread(target=lambda: (time.sleep(0.3), SERVER.shutdown()), daemon=True).start()
    return {"ok": True}


GET_ROUTES = {"/api/info": api_info, "/api/fs": api_fs, "/api/plan_status": api_plan_status, "/api/status": api_status}
POST_ROUTES = {"/api/scan": api_scan, "/api/plan": api_plan, "/api/start": api_start,
               "/api/cancel": api_cancel, "/api/open": api_open, "/api/quit": api_quit}


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "V2AWeb/1.0"

    def log_message(self, *a):
        pass

    def _host_ok(self):
        return (self.headers.get("Host") or "").split(":")[0].lower() in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _dispatch(self, routes, arg):
        url = urlparse(self.path)
        fn = routes.get(url.path)
        if not fn:
            return self._send(404, {"error": "Not found"})
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, {"error": "Bad token - reload the page"})
        try:
            self._send(200, fn(arg(url)))
        except ApiError as e:
            self._send(e.code, {"error": e.msg})
        except Exception as e:                  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "Forbidden host"})
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.replace("__TOKEN__", TOKEN), "text/html; charset=utf-8")
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
            self._send(400, {"error": "Bad request"})


# --------------------------------------------------------------------------- the page
PAGE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video to Audio</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%237c5cff'/%3E%3Cpath d='M14 32h4M22 22v20M30 14v36M38 22v20M46 28v8M52 32h0' stroke='white' stroke-width='5' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root{--bg:#070b14;--card:rgba(255,255,255,.05);--card2:rgba(255,255,255,.085);--line:rgba(255,255,255,.1);
--txt:#e9edf7;--mut:#8e98b0;--acc:#7c5cff;--acc2:#22d3ee;--acc3:#ec4899;--ok:#34d399;--warn:#fbbf24;--bad:#fb7185;
--shadow:0 20px 60px -20px rgba(0,0,0,.6);--r:18px;color-scheme:dark}
[data-theme=light]{--bg:#f2f5fb;--card:rgba(255,255,255,.8);--card2:#fff;--line:rgba(15,23,42,.1);--txt:#0f172a;
--mut:#64748b;--shadow:0 20px 50px -25px rgba(30,41,90,.35);color-scheme:light}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--txt);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;min-height:100vh;
overflow-x:hidden}
.blobs{position:fixed;inset:0;z-index:-1;overflow:hidden;pointer-events:none}
.blobs i{position:absolute;width:55vmax;height:55vmax;border-radius:50%;filter:blur(90px);opacity:.28;animation:float 22s ease-in-out infinite}
.blobs i:nth-child(1){background:var(--acc);left:-15vmax;top:-20vmax}
.blobs i:nth-child(2){background:var(--acc2);right:-20vmax;top:10vmax;animation-delay:-7s}
.blobs i:nth-child(3){background:var(--acc3);left:20vmax;bottom:-30vmax;animation-delay:-13s;opacity:.18}
[data-theme=light] .blobs i{opacity:.16}
@keyframes float{50%{transform:translate(6vmax,4vmax) scale(1.1)}}
.wrap{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
header{display:flex;align-items:center;gap:14px;margin-bottom:22px}
.logo{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:grid;place-items:center;
box-shadow:0 10px 30px -10px var(--acc)}
.logo svg{width:26px;height:26px}
.logo rect{transform-origin:center;animation:eq 1.2s ease-in-out infinite}
.logo rect:nth-child(2){animation-delay:-.3s}.logo rect:nth-child(3){animation-delay:-.6s}.logo rect:nth-child(4){animation-delay:-.9s}
@keyframes eq{50%{transform:scaleY(.35)}}
h1{font-size:21px;margin:0;letter-spacing:-.3px}
.sub{color:var(--mut);font-size:13px}
.sp{flex:1}
.chip{font-size:12px;color:var(--mut);border:1px solid var(--line);border-radius:99px;padding:4px 10px;background:var(--card)}
.icon-btn{border:1px solid var(--line);background:var(--card);color:var(--txt);width:38px;height:38px;border-radius:12px;cursor:pointer;
display:grid;place-items:center;transition:.2s}
.icon-btn:hover{background:var(--card2);transform:translateY(-1px)}
.icon-btn svg{width:18px;height:18px}
.stepper{display:flex;gap:6px;margin:0 0 20px;padding:0;list-style:none}
.stepper li{flex:1;display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:14px;border:1px solid var(--line);
background:var(--card);color:var(--mut);cursor:default;transition:.25s;min-width:0}
.stepper li b{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-size:12px;background:var(--card2);flex:none;transition:.25s}
.stepper li span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:13px;font-weight:600}
.stepper li.reach{cursor:pointer;color:var(--txt)}
.stepper li.done b{background:var(--ok);color:#052e1f}
.stepper li.active{color:var(--txt);border-color:transparent;background:linear-gradient(var(--bg),var(--bg)) padding-box,
linear-gradient(120deg,var(--acc),var(--acc2)) border-box;border:1px solid transparent;box-shadow:0 10px 30px -15px var(--acc)}
.stepper li.active b{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#fff}
[data-theme=light] .stepper li.active{background:linear-gradient(#fff,#fff) padding-box,linear-gradient(120deg,var(--acc),var(--acc2)) border-box}
.panel{display:none;background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:26px;box-shadow:var(--shadow);
backdrop-filter:blur(14px)}
.panel.active{display:block;animation:in .45s cubic-bezier(.2,.8,.2,1)}
@keyframes in{from{opacity:0;transform:translateY(14px)}}
h2{margin:0 0 4px;font-size:20px;letter-spacing:-.2px}
.lead{color:var(--mut);margin:0 0 20px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=text],input[type=number]{background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:12px;padding:11px 14px;
font:inherit;outline:none;transition:.2s;min-width:0}
input[type=text]:focus,input[type=number]:focus{border-color:var(--acc);box-shadow:0 0 0 4px rgba(124,92,255,.18)}
.grow{flex:1}
.btn{border:1px solid var(--line);background:var(--card2);color:var(--txt);border-radius:12px;padding:10px 16px;font:inherit;font-weight:600;
cursor:pointer;transition:.2s;display:inline-flex;align-items:center;gap:8px}
.btn:hover:not(:disabled){transform:translateY(-1px);border-color:var(--acc)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.primary{background:linear-gradient(135deg,var(--acc),#5b8cff 55%,var(--acc2));border:0;color:#fff;box-shadow:0 12px 30px -12px var(--acc)}
.btn.primary:hover:not(:disabled){box-shadow:0 16px 36px -10px var(--acc)}
.btn.danger{color:var(--bad)}
.btn svg{width:17px;height:17px}
.foot{display:flex;justify-content:space-between;gap:10px;margin-top:22px;flex-wrap:wrap}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.chips button{font:inherit;font-size:12.5px;color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:99px;padding:5px 12px;
cursor:pointer;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;transition:.2s}
.chips button:hover{color:var(--txt);border-color:var(--acc)}
.switch{display:inline-flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.switch input{display:none}
.switch span{width:42px;height:24px;border-radius:99px;background:var(--card2);border:1px solid var(--line);position:relative;transition:.25s;flex:none}
.switch span:after{content:"";position:absolute;left:3px;top:3px;width:16px;height:16px;border-radius:50%;background:var(--mut);transition:.25s}
.switch input:checked+span{background:var(--acc);border-color:var(--acc)}
.switch input:checked+span:after{left:21px;background:#fff}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.tile small{color:var(--mut);display:block;font-size:12px}
.tile b{font-size:22px;letter-spacing:-.4px}
.tile.ok b{color:var(--ok)}
.tablewrap{border:1px solid var(--line);border-radius:14px;max-height:400px;overflow:auto;background:var(--card)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{position:sticky;top:0;background:var(--bg);text-align:left;font-size:12px;color:var(--mut);font-weight:600;padding:10px 12px;z-index:1;
border-bottom:1px solid var(--line)}
[data-theme=light] th{background:#eef2f9}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:0}
tbody tr{transition:background .15s}
tbody tr:hover{background:var(--card2)}
td.n{color:var(--mut);text-align:right;width:56px}
td.sz,th.sz{text-align:right;white-space:nowrap;color:var(--mut)}
td.nm{max-width:0;width:100%}
td.nm div{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
td.nm small{color:var(--mut);display:block;overflow:hidden;text-overflow:ellipsis}
input[type=checkbox].cb{appearance:none;width:19px;height:19px;border-radius:6px;border:1.5px solid var(--mut);cursor:pointer;display:grid;
place-items:center;transition:.15s;margin:0;background:transparent}
input[type=checkbox].cb:checked{background:var(--acc);border-color:var(--acc)}
input[type=checkbox].cb:checked:after{content:"";width:9px;height:5px;border:2px solid #fff;border-top:0;border-right:0;transform:rotate(-45deg) translate(1px,-1px)}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.toolbar .rng{display:flex;align-items:center;gap:6px;color:var(--mut);font-size:13px}
.toolbar input[type=number]{width:78px;padding:9px 10px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.pcard{position:relative;text-align:left;font:inherit;color:inherit;background:var(--card);border:1.5px solid var(--line);border-radius:16px;
padding:16px;cursor:pointer;transition:.22s}
.pcard:hover{transform:translateY(-2px);border-color:var(--acc)}
.pcard.sel{border-color:var(--acc);background:linear-gradient(160deg,rgba(124,92,255,.18),rgba(34,211,238,.06));box-shadow:0 14px 34px -18px var(--acc)}
.pcard h3{margin:0 0 4px;font-size:16px}
.pcard p{margin:0;color:var(--mut);font-size:13px}
.pcard .meta{margin-top:10px;font-size:12px;color:var(--mut)}
.tag{display:inline-block;font-size:11px;font-weight:700;color:#06281c;background:var(--ok);border-radius:99px;padding:2px 9px;margin-left:8px;vertical-align:middle}
.pcard .tick{position:absolute;right:14px;top:14px;width:22px;height:22px;border-radius:50%;background:var(--acc);display:grid;place-items:center;
transform:scale(0);transition:.25s cubic-bezier(.3,1.6,.5,1)}
.pcard.sel .tick{transform:scale(1)}
.pcard .tick svg{width:13px;height:13px;stroke:#fff}
.field{margin-top:20px}
.field label.t{display:block;font-size:13px;color:var(--mut);margin-bottom:6px;font-weight:600}
.warns{display:grid;gap:8px;margin:0 0 16px}
.warn{display:flex;gap:10px;background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);color:var(--warn);border-radius:12px;padding:9px 13px;font-size:13.5px}
[data-theme=light] .warn{color:#92600a}
.pill{font-size:12px;font-weight:700;border-radius:99px;padding:3px 10px;white-space:nowrap}
.pill.convert{background:rgba(52,211,153,.15);color:var(--ok)}
.pill.skip{background:var(--card2);color:var(--mut)}
.pill.noaudio{background:rgba(251,113,133,.15);color:var(--bad)}
.loader{padding:50px 10px;text-align:center}
.bar{height:10px;border-radius:99px;background:var(--card2);overflow:hidden;position:relative}
.bar i{display:block;height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .45s ease;position:relative;overflow:hidden}
.bar i:after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(115deg,rgba(255,255,255,.28) 0 10px,transparent 10px 22px);
background-size:44px 100%;animation:slide 1s linear infinite;opacity:.55}
.bar.idle i:after{animation:none;opacity:0}
.bar.ind i{width:35%!important;animation:ind 1.3s ease-in-out infinite}
@keyframes slide{to{background-position:44px 0}}
@keyframes ind{0%{margin-left:-35%}100%{margin-left:100%}}
.hero{display:grid;grid-template-columns:200px 1fr;gap:26px;align-items:center;margin-bottom:20px}
.ring{position:relative;width:190px;height:190px}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:11;stroke-linecap:round}
.ring .bg{stroke:var(--card2)}
.ring .fg{stroke:url(#g);stroke-dasharray:339.3;stroke-dashoffset:339.3;transition:stroke-dashoffset .5s ease}
.ring .pc{position:absolute;inset:0;display:grid;place-content:center;text-align:center}
.ring .pc b{font-size:38px;letter-spacing:-1px;line-height:1}
.ring .pc small{color:var(--mut);margin-top:4px}
.ring.done .fg{stroke:var(--ok)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}
.cur{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:16px}
.cur .top{display:flex;gap:10px;justify-content:space-between;margin-bottom:10px;font-size:14px}
.cur .top b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cur .top span{color:var(--mut);white-space:nowrap;font-variant-numeric:tabular-nums}
.queue{border:1px solid var(--line);border-radius:14px;max-height:300px;overflow:auto;position:relative;background:var(--card)}
.qi{display:flex;gap:12px;align-items:center;padding:9px 14px;border-bottom:1px solid var(--line);font-size:14px;transition:background .3s}
.qi:last-child{border-bottom:0}
.qi .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.qi .st{color:var(--mut);font-size:12.5px;white-space:nowrap;font-variant-numeric:tabular-nums}
.qi.running{background:rgba(124,92,255,.12)}
.qi.failed .st{color:var(--bad)}
.dot{width:20px;height:20px;flex:none;display:grid;place-items:center}
.dot:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--mut);opacity:.5}
.qi.running .dot:before{width:16px;height:16px;background:none;border:2.5px solid var(--acc);border-top-color:transparent;opacity:1;animation:spin .8s linear infinite}
.qi.done .dot:before{content:"\2713";width:auto;height:auto;background:none;color:var(--ok);font-weight:900;opacity:1;border-radius:0}
.qi.failed .dot:before{content:"\2715";width:auto;height:auto;background:none;color:var(--bad);font-weight:900;opacity:1;border-radius:0}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{width:34px;height:34px;border:3.5px solid var(--card2);border-top-color:var(--acc);border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 14px}
details{margin-top:14px}
summary{cursor:pointer;color:var(--mut);font-size:13px}
pre.log{margin:8px 0 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;max-height:180px;overflow:auto;font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--mut)}
.banner{display:none;align-items:center;gap:14px;border-radius:14px;padding:14px 18px;margin-bottom:16px;border:1px solid var(--line)}
.banner.show{display:flex;animation:in .5s}
.banner.ok{background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.35)}
.banner.warn2{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.3)}
.banner .big{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;font-size:22px;font-weight:900;flex:none}
.banner.ok .big{background:var(--ok);color:#052e1f}
.banner.warn2 .big{background:var(--warn);color:#3b2a00}
.banner.ok .big svg{width:22px;stroke-dasharray:30;stroke-dashoffset:30;animation:draw .6s .15s forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.modal{position:fixed;inset:0;background:rgba(3,6,15,.6);backdrop-filter:blur(6px);display:none;place-items:center;z-index:50;padding:20px}
.modal.show{display:grid;animation:fade .2s}
@keyframes fade{from{opacity:0}}
.sheet{width:min(640px,100%);max-height:86vh;display:flex;flex-direction:column;background:var(--bg);border:1px solid var(--line);border-radius:20px;
box-shadow:var(--shadow);overflow:hidden;animation:in .3s}
[data-theme=light] .sheet{background:#fff}
.sheet .hd{padding:16px 18px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
.crumbs{display:flex;flex-wrap:wrap;gap:2px;font-size:13px;flex:1;min-width:0}
.crumbs button{background:none;border:0;color:var(--mut);font:inherit;cursor:pointer;padding:3px 6px;border-radius:8px}
.crumbs button:hover{background:var(--card2);color:var(--txt)}
.crumbs button:last-child{color:var(--txt);font-weight:700}
.sheet .bd{display:grid;grid-template-columns:150px 1fr;min-height:0;flex:1}
.side{border-right:1px solid var(--line);padding:10px;overflow:auto}
.side button,.dirs button{display:flex;align-items:center;gap:8px;width:100%;text-align:left;background:none;border:0;color:var(--txt);font:inherit;
font-size:14px;padding:8px 10px;border-radius:10px;cursor:pointer}
.side button:hover,.dirs button:hover{background:var(--card2)}
.dirs{padding:10px;overflow:auto}
.dirs button span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dirs svg,.side svg{width:17px;height:17px;flex:none;color:var(--acc2)}
.sheet .ft{padding:14px 18px;border-top:1px solid var(--line);display:flex;gap:10px;align-items:center;justify-content:space-between}
.empty{color:var(--mut);padding:20px;text-align:center}
.toasts{position:fixed;right:18px;bottom:18px;display:grid;gap:8px;z-index:90}
.toast{background:var(--bg);border:1px solid var(--bad);color:var(--txt);border-radius:12px;padding:11px 16px;box-shadow:var(--shadow);max-width:360px;animation:in .3s}
.toast.info{border-color:var(--acc)}
[data-theme=light] .toast{background:#fff}
.confetti{position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:80}
.confetti i{position:absolute;top:-20px;width:9px;height:14px;border-radius:2px;animation:fall linear forwards}
@keyframes fall{to{transform:translateY(110vh) rotate(720deg);opacity:.8}}
@media(max-width:760px){.stepper li span{display:none}.hero{grid-template-columns:1fr;justify-items:center}.sheet .bd{grid-template-columns:1fr}.side{display:none}
.panel{padding:18px}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<div class="blobs"><i></i><i></i><i></i></div>
<div class="wrap">
  <header>
    <div class="logo"><svg viewBox="0 0 26 26" fill="#fff"><rect x="2" y="9" width="3" height="8" rx="1.5"/><rect x="8" y="4" width="3" height="18" rx="1.5"/><rect x="14" y="7" width="3" height="12" rx="1.5"/><rect x="20" y="10" width="3" height="6" rx="1.5"/></svg></div>
    <div><h1>Video to Audio</h1><div class="sub">Fast local converter &middot; your files never leave this computer</div></div>
    <div class="sp"></div>
    <span class="chip" id="ffchip">ffmpeg</span>
    <button class="icon-btn" id="themeBtn" title="Light / dark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
    <button class="icon-btn" id="quitBtn" title="Quit the app"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 3v9M6.3 6.3a8 8 0 1 0 11.4 0"/></svg></button>
  </header>

  <ol class="stepper" id="stepper">
    <li data-s="1"><b>1</b><span>Folder</span></li>
    <li data-s="2"><b>2</b><span>Videos</span></li>
    <li data-s="3"><b>3</b><span>Format</span></li>
    <li data-s="4"><b>4</b><span>Review</span></li>
    <li data-s="5"><b>5</b><span>Convert</span></li>
  </ol>

  <!-- STEP 1 -->
  <section class="panel" id="p1">
    <h2>Where are your videos?</h2>
    <p class="lead">Pick the folder on your computer. Nothing is uploaded; the app reads the files straight from disk.</p>
    <div class="row">
      <input type="text" id="path" class="grow" placeholder="Paste a folder path, or press Browse" spellcheck="false">
      <button class="btn" id="browseBtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>Browse</button>
    </div>
    <div class="row" style="margin-top:16px">
      <label class="switch"><input type="checkbox" id="recursive"><span></span>Include sub-folders</label>
    </div>
    <div class="chips" id="recents"></div>
    <div class="foot"><span></span><button class="btn primary" id="scanBtn">Find videos
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button></div>
  </section>

  <!-- STEP 2 -->
  <section class="panel" id="p2">
    <h2>Choose the videos</h2>
    <p class="lead">Sorted by the number in each file name, so Session 2 comes before Session 10.</p>
    <div class="tiles">
      <div class="tile"><small>Videos found</small><b id="tFound">0</b></div>
      <div class="tile"><small>Total size</small><b id="tSize">0</b></div>
      <div class="tile ok"><small>Selected</small><b id="tSel">0</b></div>
      <div class="tile"><small>Selected size</small><b id="tSelSize">0</b></div>
    </div>
    <div class="toolbar">
      <input type="text" id="q" class="grow" placeholder="Search by name...">
      <div class="rng">No. <input type="number" id="rFrom" placeholder="from" min="0"> to <input type="number" id="rTo" placeholder="to" min="0">
        <button class="btn" id="rApply">Select range</button></div>
    </div>
    <div class="toolbar">
      <button class="btn" id="selAll">All</button><button class="btn" id="selNone">None</button><button class="btn" id="selInv">Invert</button>
      <span class="sub" id="visInfo"></span>
    </div>
    <div class="tablewrap"><table><thead><tr><th style="width:40px"><input type="checkbox" class="cb" id="hdrCb"></th><th>No.</th><th>Video</th><th class="sz">Size</th></tr></thead>
      <tbody id="vbody"></tbody></table></div>
    <div class="foot"><button class="btn" data-go="1">Back</button><button class="btn primary" id="to3">Continue</button></div>
  </section>

  <!-- STEP 3 -->
  <section class="panel" id="p3">
    <h2>Pick an audio format</h2>
    <p class="lead">For transcription with Whisper, the first option is small and works well.</p>
    <div class="cards" id="presets"></div>
    <div class="field"><label class="t">Save audio to</label>
      <div class="row"><input type="text" id="outdir" class="grow" spellcheck="false">
      <button class="btn" id="outBrowse">Browse</button><button class="btn" id="outReset">Default</button></div></div>
    <div class="field"><label class="switch"><input type="checkbox" id="overwrite"><span></span>Replace audio files that already exist</label></div>
    <div class="foot"><button class="btn" data-go="2">Back</button><button class="btn primary" id="to4">Review (dry run)</button></div>
  </section>

  <!-- STEP 4 -->
  <section class="panel" id="p4">
    <h2>Review before converting</h2>
    <p class="lead">This is a dry run. Nothing has been converted yet.</p>
    <div id="planLoad" class="loader"><div class="spinner"></div><div id="planMsg">Reading video info...</div>
      <div class="bar" style="max-width:320px;margin:14px auto 0"><i id="planBar"></i></div></div>
    <div id="planBody" style="display:none">
      <div class="tiles" id="planTiles"></div>
      <div class="warns" id="planWarns"></div>
      <div class="tablewrap"><table><thead><tr><th>No.</th><th>Video</th><th class="sz">Size</th><th class="sz">Length</th><th>Status</th></tr></thead><tbody id="planRows"></tbody></table></div>
    </div>
    <div class="foot"><button class="btn" data-go="3">Back</button><button class="btn primary" id="startBtn" disabled>Start converting
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg></button></div>
  </section>

  <!-- STEP 5 -->
  <section class="panel" id="p5">
    <div class="banner" id="doneBanner"><div class="big" id="doneIcon"></div><div><b id="doneTitle"></b><div class="sub" id="doneSub"></div></div></div>
    <div class="hero">
      <div class="ring" id="ring"><svg viewBox="0 0 120 120"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>
        <circle class="bg" cx="60" cy="60" r="54"/><circle class="fg" id="ringFg" cx="60" cy="60" r="54"/></svg>
        <div class="pc"><b id="pct">0%</b><small id="pcSub">starting</small></div></div>
      <div class="stats">
        <div class="tile"><small>Files done</small><b id="sDone">0/0</b></div>
        <div class="tile"><small>Time left</small><b id="sEta">--</b></div>
        <div class="tile"><small>Elapsed</small><b id="sEl">0:00</b></div>
        <div class="tile"><small>Speed</small><b id="sX">--</b></div>
        <div class="tile"><small>Audio so far</small><b id="sOut">0</b></div>
      </div>
    </div>
    <div class="cur" id="curBox"><div class="top"><b id="curName">Waiting...</b><span id="curInfo"></span></div>
      <div class="bar" id="curBar"><i id="curFill"></i></div></div>
    <div class="queue" id="queue"></div>
    <details><summary>Activity log</summary><pre class="log" id="logBox"></pre></details>
    <div class="foot"><button class="btn danger" id="cancelBtn">Cancel</button>
      <div class="row"><button class="btn" id="openBtn" style="display:none">Open output folder</button>
      <button class="btn primary" id="againBtn" style="display:none">Convert more</button></div></div>
  </section>
</div>

<div class="modal" id="modal"><div class="sheet">
  <div class="hd"><div class="crumbs" id="crumbs"></div><button class="icon-btn" id="mClose" title="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div>
  <div class="bd"><div class="side" id="side"></div><div class="dirs" id="dirs"></div></div>
  <div class="ft"><span class="sub" id="mInfo"></span><button class="btn primary" id="mPick">Use this folder</button></div>
</div></div>
<div class="toasts" id="toasts"></div>
<div class="confetti" id="confetti"></div>

<script>
const TOKEN="__TOKEN__";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const S={step:1,maxStep:1,info:null,root:"",items:[],sel:new Set(),preset:"1",outEdited:false,planId:null,plan:null,running:false,timer:null,queueBuilt:false};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const human=n=>{if(n==null)return"--";const u=["B","KB","MB","GB","TB"];let i=0;while(n>=1024&&i<4){n/=1024;i++}return(i<2?Math.round(n):n.toFixed(1))+" "+u[i]};
const hms=s=>{if(s==null||!isFinite(s))return"--";s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return h?`${h}:${String(m).padStart(2,"0")}:${String(x).padStart(2,"0")}`:`${m}:${String(x).padStart(2,"0")}`};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

async function api(path,body){
  const opt=body===undefined?{headers:{"X-Token":TOKEN}}:{method:"POST",headers:{"X-Token":TOKEN,"Content-Type":"application/json"},body:JSON.stringify(body)};
  let r;try{r=await fetch("/api/"+path,opt)}catch(e){throw new Error("Cannot reach the app. Is it still running?")}
  let j={};try{j=await r.json()}catch(e){}
  if(!r.ok)throw new Error(j.error||("Request failed ("+r.status+")"));return j;
}
function toast(msg,info){const t=document.createElement("div");t.className="toast"+(info?" info":"");t.textContent=msg;$("#toasts").appendChild(t);setTimeout(()=>t.remove(),5200)}

/* ---------- navigation ---------- */
function go(n){
  S.step=n;S.maxStep=Math.max(S.maxStep,n);
  $$(".panel").forEach(p=>p.classList.toggle("active",p.id==="p"+n));
  $$("#stepper li").forEach(li=>{const s=+li.dataset.s;li.classList.toggle("active",s===n);li.classList.toggle("done",s<n);li.classList.toggle("reach",s<=S.maxStep&&!S.running)});
  window.scrollTo({top:0,behavior:"smooth"});
}
$$("#stepper li").forEach(li=>li.onclick=()=>{const s=+li.dataset.s;if(S.running||s>S.maxStep)return;if(s===5&&!S.queueBuilt)return;go(s)});
$$("[data-go]").forEach(b=>b.onclick=()=>go(+b.dataset.go));

/* ---------- theme / quit ---------- */
function setTheme(t){document.documentElement.dataset.theme=t;try{localStorage.setItem("v2a-theme",t)}catch(e){}}
$("#themeBtn").onclick=()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark");
$("#quitBtn").onclick=async()=>{if(S.running&&!confirm("A conversion is running. Quit and cancel it?"))return;
  try{await api("quit",{})}catch(e){}document.body.innerHTML='<div style="display:grid;place-items:center;height:100vh;text-align:center;color:#8e98b0"><div><h2 style="color:#e9edf7">App closed</h2>You can close this tab.</div></div>'};

/* ---------- folder browser ---------- */
let browseCb=null,browsePath="";
const ICON_DIR='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>';
async function openBrowser(start,cb){browseCb=cb;$("#modal").classList.add("show");await loadDir(start||S.info.home)}
async function loadDir(p){
  try{
    const d=await api("fs?path="+encodeURIComponent(p||""));browsePath=d.path;
    const sep=S.info.sep;const parts=[];
    parts.push(`<button data-p="">This PC</button>`);
    if(d.path){let acc="";const segs=d.path.split(sep).filter(Boolean);
      segs.forEach((sg,i)=>{acc=(i===0?(S.info.sep==="\\"?sg+"\\":"/"+sg):acc+(acc.endsWith(sep)?"":sep)+sg);parts.push(`<span style="color:var(--mut)">&rsaquo;</span><button data-p="${esc(acc)}">${esc(sg)}</button>`)})}
    $("#crumbs").innerHTML=parts.join("");
    $("#side").innerHTML=(S.info.places||[]).map(x=>`<button data-p="${esc(x.path)}">${ICON_DIR}<span>${esc(x.name)}</span></button>`).join("");
    let rows=[];if(d.path&&d.parent!==null)rows.push(`<button data-p="${esc(d.parent)}">${ICON_DIR}<span>.. (up)</span></button>`);
    rows=rows.concat(d.dirs.map(x=>`<button data-p="${esc(x.path)}">${ICON_DIR}<span>${esc(x.name)}</span></button>`));
    $("#dirs").innerHTML=rows.length?rows.join(""):'<div class="empty">No sub-folders here</div>';
    $("#mInfo").textContent=d.path?(d.videos?`${d.videos} video file(s) in this folder`:"No videos directly in this folder"):"Choose a drive";
    $("#mPick").disabled=!d.path;
    $$("#crumbs button,#side button,#dirs button").forEach(b=>b.onclick=()=>loadDir(b.dataset.p));
    $("#dirs").scrollTop=0;
  }catch(e){toast(e.message)}
}
$("#mClose").onclick=()=>$("#modal").classList.remove("show");
$("#modal").onclick=e=>{if(e.target.id==="modal")$("#modal").classList.remove("show")};
$("#mPick").onclick=()=>{$("#modal").classList.remove("show");if(browseCb)browseCb(browsePath)};

/* ---------- step 1 ---------- */
function recents(){let r=[];try{r=JSON.parse(localStorage.getItem("v2a-recent")||"[]")}catch(e){}return r}
function saveRecent(p){let r=recents().filter(x=>x!==p);r.unshift(p);try{localStorage.setItem("v2a-recent",JSON.stringify(r.slice(0,5)))}catch(e){}}
function renderRecents(){const r=recents();$("#recents").innerHTML=r.length?r.map(p=>`<button title="${esc(p)}" data-p="${esc(p)}">${esc(p)}</button>`).join(""):"";
  $$("#recents button").forEach(b=>b.onclick=()=>{$("#path").value=b.dataset.p})}
$("#browseBtn").onclick=()=>openBrowser($("#path").value.trim(),p=>{$("#path").value=p});
$("#path").addEventListener("keydown",e=>{if(e.key==="Enter")scan()});
$("#scanBtn").onclick=scan;
async function scan(){
  const p=$("#path").value.trim().replace(/^["']|["']$/g,"");if(!p){toast("Choose a folder first");return}
  const b=$("#scanBtn");b.disabled=true;const old=b.innerHTML;b.textContent="Searching...";
  try{
    const d=await api("scan",{path:p,recursive:$("#recursive").checked});
    if(!d.items.length){toast("No videos found in that folder");return}
    S.root=d.root;S.items=d.items;S.sel=new Set(d.items.map(i=>i.id));saveRecent(d.root);renderRecents();
    $("#path").value=d.root;if(!S.outEdited)$("#outdir").value=d.root+(d.root.endsWith(S.info.sep)?"":S.info.sep)+"audio";
    $("#q").value="";$("#rFrom").value="";$("#rTo").value="";S.maxStep=2;S.queueBuilt=false;
    renderVideos();go(2);
  }catch(e){toast(e.message)}finally{b.disabled=false;b.innerHTML=old}
}

/* ---------- step 2 ---------- */
function visible(){const q=$("#q").value.trim().toLowerCase();return S.items.filter(i=>!q||(i.rel+"/"+i.name).toLowerCase().includes(q))}
function renderVideos(){
  const vis=visible();
  $("#vbody").innerHTML=vis.map(i=>`<tr><td><input type="checkbox" class="cb row-cb" data-id="${i.id}" ${S.sel.has(i.id)?"checked":""}></td>
    <td class="n">${i.num==null?"-":i.num}</td><td class="nm"><div title="${esc(i.name)}">${esc(i.name)}${i.cloud?" \u2601":""}</div>${i.rel?`<small>${esc(i.rel)}</small>`:""}</td>
    <td class="sz">${human(i.size)}</td></tr>`).join("");
  $$(".row-cb").forEach(c=>c.onchange=()=>{const id=+c.dataset.id;c.checked?S.sel.add(id):S.sel.delete(id);counts()});
  $("#visInfo").textContent=vis.length<S.items.length?`Showing ${vis.length} of ${S.items.length}`:"";
  $("#tFound").textContent=S.items.length;$("#tSize").textContent=human(S.items.reduce((a,i)=>a+i.size,0));counts();
}
function counts(){
  const chosen=S.items.filter(i=>S.sel.has(i.id));
  $("#tSel").textContent=chosen.length;$("#tSelSize").textContent=human(chosen.reduce((a,i)=>a+i.size,0));
  const vis=visible();const all=vis.length&&vis.every(i=>S.sel.has(i.id));$("#hdrCb").checked=!!all;
  $("#to3").disabled=!chosen.length;
}
$("#q").oninput=renderVideos;
$("#hdrCb").onchange=e=>{visible().forEach(i=>e.target.checked?S.sel.add(i.id):S.sel.delete(i.id));renderVideos()};
$("#selAll").onclick=()=>{visible().forEach(i=>S.sel.add(i.id));renderVideos()};
$("#selNone").onclick=()=>{visible().forEach(i=>S.sel.delete(i.id));renderVideos()};
$("#selInv").onclick=()=>{visible().forEach(i=>S.sel.has(i.id)?S.sel.delete(i.id):S.sel.add(i.id));renderVideos()};
$("#rApply").onclick=()=>{
  const a=$("#rFrom").value===""?-Infinity:+$("#rFrom").value,b=$("#rTo").value===""?Infinity:+$("#rTo").value;
  if(a===-Infinity&&b===Infinity){toast("Type a start and/or end number");return}
  S.sel=new Set(S.items.filter(i=>i.num!=null&&i.num>=a&&i.num<=b).map(i=>i.id));renderVideos();
  toast(`${S.sel.size} video(s) selected for that range`,true);
};
$("#to3").onclick=()=>{renderPresets();go(3)};

/* ---------- step 3 ---------- */
function renderPresets(){
  $("#presets").innerHTML=S.info.presets.map(p=>{
    const per=p.kbps?`about ${human(p.kbps*1000/8*3600)} per hour of video`:"size varies";
    return `<button class="pcard ${p.id===S.preset?"sel":""}" data-id="${p.id}"><span class="tick"><svg viewBox="0 0 24 24" fill="none" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg></span>
      <h3>${esc(p.title)}${p.tag?`<span class="tag">${esc(p.tag)}</span>`:""}</h3><p>${esc(p.desc)}</p><div class="meta">.${p.ext} &middot; ${per}</div></button>`}).join("");
  $$(".pcard").forEach(c=>c.onclick=()=>{S.preset=c.dataset.id;renderPresets()});
}
$("#outdir").oninput=()=>{S.outEdited=true};
$("#outReset").onclick=()=>{S.outEdited=false;$("#outdir").value=S.root+(S.root.endsWith(S.info.sep)?"":S.info.sep)+"audio"};
$("#outBrowse").onclick=()=>openBrowser(S.root,p=>{S.outEdited=true;$("#outdir").value=p});
$("#to4").onclick=()=>runPlan();

/* ---------- step 4 ---------- */
async function runPlan(){
  const ids=S.items.filter(i=>S.sel.has(i.id)).map(i=>i.id);
  const cloud=S.items.some(i=>S.sel.has(i.id)&&i.cloud);
  let probe=true;
  if(cloud)probe=confirm("Some selected files are online-only (OneDrive). Reading their length downloads them first.\n\nOK = read lengths anyway\nCancel = skip reading lengths");
  go(4);$("#planLoad").style.display="";$("#planBody").style.display="none";$("#startBtn").disabled=true;
  $("#planMsg").textContent=probe?"Reading video info...":"Preparing...";$("#planBar").style.width="0";
  try{
    const r=await api("plan",{ids,preset:S.preset,out_dir:$("#outdir").value.trim(),overwrite:$("#overwrite").checked,probe});
    S.planId=r.plan_id;
    for(;;){
      await sleep(350);const st=await api("plan_status");
      if(st.id!==S.planId)return;
      if(st.state==="running"){$("#planBar").style.width=(st.total?st.done/st.total*100:5)+"%";if(st.total)$("#planMsg").textContent=`Reading video info... ${st.done}/${st.total}`}
      else if(st.state==="error"){throw new Error(st.error)}
      else{renderPlan(st.result);return}
    }
  }catch(e){toast(e.message);go(3)}
}
function renderPlan(r){
  S.plan=r;const s=r.summary;
  $("#planLoad").style.display="none";$("#planBody").style.display="";
  $("#planTiles").innerHTML=`<div class="tile ok"><small>Will convert</small><b>${s.todo}</b></div>
   <div class="tile"><small>Already done</small><b>${s.skipped}</b></div>
   <div class="tile"><small>Video length</small><b>${s.durations_known&&s.duration?hms(s.duration):"--"}</b></div>
   <div class="tile"><small>Audio size (est.)</small><b>${s.est_bytes?"~"+human(s.est_bytes):"--"}</b></div>
   <div class="tile"><small>Free disk space</small><b>${human(s.free_bytes)}</b></div>`;
  $("#planWarns").innerHTML=r.warnings.map(w=>`<div class="warn"><span>&#9888;</span><span>${esc(w)}</span></div>`).join("");
  const lab={convert:"Will convert",skip:"Skip (exists)",noaudio:"No audio track"};
  $("#planRows").innerHTML=r.items.map(i=>`<tr><td class="n">${i.num==null?"-":i.num}</td><td class="nm"><div title="${esc(i.name)}">${esc(i.name)}${i.cloud?" \u2601":""}</div>
    <small>${esc(i.out)}${i.note?" &middot; "+esc(i.note):""}</small></td><td class="sz">${human(i.size)}</td><td class="sz">${i.duration?hms(i.duration):"--"}</td>
    <td><span class="pill ${i.status}">${lab[i.status]}</span></td></tr>`).join("");
  const b=$("#startBtn");b.disabled=s.todo===0;b.firstChild.textContent=s.todo?`Start converting ${s.todo} file${s.todo>1?"s":""} `:"Nothing to convert ";
}
$("#startBtn").onclick=async()=>{
  try{await api("start",{plan_id:S.planId});S.queueBuilt=false;S.running=true;startPolling();go(5)}catch(e){toast(e.message)}
};

/* ---------- step 5 ---------- */
function buildQueue(items){
  $("#queue").innerHTML=items.map(i=>`<div class="qi queued" data-id="${i.id}"><span class="dot"></span><span class="nm" title="${esc(i.name)}">${esc(i.name)}</span><span class="st">waiting</span></div>`).join("");
  S.queueBuilt=true;$("#doneBanner").className="banner";$("#openBtn").style.display="none";$("#againBtn").style.display="none";
  $("#cancelBtn").style.display="";$("#cancelBtn").disabled=false;$("#ring").classList.remove("done");
}
function startPolling(){clearInterval(S.timer);S.timer=setInterval(poll,500);poll()}
let polling=false;
async function poll(){
  if(polling)return;polling=true;
  try{const st=await api("status");render5(st);
    if(st.state!=="running"&&st.state!=="idle"){clearInterval(S.timer);S.running=false;finish(st);go(5)}}
  catch(e){clearInterval(S.timer);toast(e.message)}finally{polling=false}
}
function render5(st){
  if(!st.items.length)return;
  if(!S.queueBuilt)buildQueue(st.items);
  const pct=Math.round(st.fraction*100);
  $("#pct").textContent=pct+"%";$("#ringFg").style.strokeDashoffset=339.3*(1-st.fraction);
  $("#pcSub").textContent=st.state==="running"?"converting":st.state==="cancelled"?"cancelled":"complete";
  const done=st.items.filter(i=>i.status==="done"||i.status==="failed").length;
  $("#sDone").textContent=`${done}/${st.items.length}`;$("#sEta").textContent=st.state==="running"?(st.eta!=null?hms(st.eta):"..."):"0:00";
  $("#sEl").textContent=hms(st.elapsed);$("#sX").textContent=st.x?Math.round(st.x)+"\u00d7":"--";$("#sOut").textContent=human(st.out_size);
  const cur=st.items.find(i=>i.status==="running");
  if(cur){
    $("#curName").textContent=cur.name;$("#curBar").className="bar"+(cur.duration?"":" ind");
    $("#curFill").style.width=(cur.duration?cur.progress*100:35)+"%";
    $("#curInfo").textContent=(cur.speed||"")+"  "+hms(cur.secs)+(cur.duration?" / "+hms(cur.duration):"");
  }else{$("#curName").textContent=st.state==="running"?"Starting next file...":"Idle";$("#curInfo").textContent="";$("#curBar").className="bar idle";$("#curFill").style.width=st.state==="running"?"0":(st.fraction>=1?"100%":"0")}
  st.items.forEach(i=>{
    const el=$(`.qi[data-id="${i.id}"]`);if(!el)return;
    if(el.dataset.s!==i.status){el.dataset.s=i.status;el.className="qi "+i.status;
      if(i.status==="running"){const q=$("#queue");q.scrollTo({top:Math.max(0,el.offsetTop-q.clientHeight/2),behavior:"smooth"})}}
    const t=el.querySelector(".st");
    t.textContent=i.status==="running"?Math.round(i.progress*100)+"%":i.status==="done"?human(i.out_size):i.status==="failed"?"failed":i.status==="cancelled"?"cancelled":"waiting";
    if(i.status==="failed")t.title=i.err||"";
  });
  $("#logBox").textContent=st.log.join("\n");
}
function finish(st){
  $("#cancelBtn").style.display="none";$("#openBtn").style.display="";$("#againBtn").style.display="";
  const b=$("#doneBanner"),ok=st.state==="finished"&&st.failed===0;
  b.className="banner show "+(ok?"ok":"warn2");
  $("#doneIcon").innerHTML=ok?'<svg viewBox="0 0 24 24" fill="none" stroke="#052e1f" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>':"!";
  $("#doneTitle").textContent=st.state==="cancelled"?"Stopped":ok?"All done!":"Finished with some problems";
  $("#doneSub").textContent=`${st.ok} converted${st.failed?`, ${st.failed} failed`:""} in ${hms(st.elapsed)} \u00b7 ${human(st.out_size)} of audio`;
  $("#ring").classList.toggle("done",ok);if(ok)confetti();
}
function confetti(){
  if(matchMedia("(prefers-reduced-motion:reduce)").matches)return;
  const box=$("#confetti"),cols=["#7c5cff","#22d3ee","#ec4899","#34d399","#fbbf24"];
  for(let i=0;i<70;i++){const e=document.createElement("i");e.style.left=Math.random()*100+"%";e.style.background=cols[i%cols.length];
    e.style.animationDuration=(2+Math.random()*2.2)+"s";e.style.animationDelay=Math.random()*.6+"s";box.appendChild(e)}
  setTimeout(()=>box.innerHTML="",5200);
}
$("#cancelBtn").onclick=async()=>{if(!confirm("Cancel the conversion? Finished files are kept."))return;$("#cancelBtn").disabled=true;try{await api("cancel",{})}catch(e){toast(e.message)}};
$("#openBtn").onclick=async()=>{try{const st=await api("status");await api("open",{path:st.out_dir})}catch(e){toast(e.message)}};
$("#againBtn").onclick=()=>{S.queueBuilt=false;S.maxStep=2;go(2);renderVideos()};

/* ---------- init ---------- */
(async function(){
  try{const t=localStorage.getItem("v2a-theme");if(t)setTheme(t)}catch(e){}
  try{
    S.info=await api("info");$("#ffchip").textContent=S.info.ffmpeg.replace(/^ffmpeg version /,"ffmpeg ").slice(0,30);
    renderRecents();const r=recents();$("#path").value=r[0]||S.info.cwd;
    const st=await api("status");
    if(st.items.length&&(st.state==="running"||st.state==="finished"||st.state==="cancelled")){
      S.maxStep=5;S.running=st.state==="running";render5(st);go(5);
      if(S.running)startPolling();else finish(st);
    }else go(1);
  }catch(e){go(1);toast(e.message)}
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- main
def main():
    global SERVER
    ap = argparse.ArgumentParser(description="Local web app: convert videos to audio.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    if not FFMPEG or not FFPROBE:
        print("ffmpeg / ffprobe not found on PATH.")
        print("  Windows: winget install Gyan.FFmpeg   (then reopen the terminal)")
        print("  macOS  : brew install ffmpeg")
        print("  Linux  : sudo apt install ffmpeg")
        sys.exit(1)

    for port in range(args.port, args.port + 20):
        try:
            SERVER = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        print("Could not find a free port.")
        sys.exit(1)
    SERVER.daemon_threads = True

    url = f"http://127.0.0.1:{SERVER.server_address[1]}/"
    print(f"Video to Audio is running at {url}")
    print("Keep this window open while you use the app. Press Ctrl+C here (or use the power button in the app) to quit.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        SERVER.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        api_cancel({})
        SERVER.server_close()
        print("Stopped.")


if __name__ == "__main__":
    main()
