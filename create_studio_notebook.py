#!/usr/bin/env python3
"""
create_studio_notebook.py - Generates Whisper_Video_Studio.ipynb
Combines Video2Audio and faster-whisper into a modern, complete Colab Studio.
"""
import json
from pathlib import Path

import json, zlib, base64
from pathlib import Path

# Read colab_studio.py and compress
studio_bytes = Path("colab_studio.py").read_bytes()
b64_payload = base64.b64encode(zlib.compress(studio_bytes, 9)).decode("ascii")

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# \U0001f680 Whisper & Video Transcriber Studio (Colab Pro Edition)\n",
    "\n",
    "### Complete All-in-One Modern Studio for Video to Audio & faster-whisper\n",
    "\n",
    "**Features:**\n",
    "- \U0001f3ac **FFmpeg Video to Audio Extraction**: Speech-optimized MP3 (32kbps/16kHz mono), WAV, AAC, FLAC, Opus.\n",
    "- \U0001f399\ufe0f **GPU-Accelerated faster-whisper**: `large-v3`, `distil-large-v3` (6x speed), `large-v2`, `medium`.\n",
    "- \u26a1 **One-Click End-to-End Pipeline**: Feed videos \u27a4 extract audio \u27a4 batch transcribe \u27a4 generate `.txt`, `.srt`, and `.md` Study Notes.\n",
    "- \U0001f4c4 **Multi-Format Transcripts**: Timestamped lines (`.txt`), subtitle timings (`.srt`), and readable formatted paragraph study notes (`.md`).\n",
    "- \U0001f3a8 **Modern Glassmorphism UI/UX**: Interactive web UI rendered directly inside your Colab cell or popped out into a dedicated tab.\n",
    "- \U0001f504 **100% Resumable**: Automatically skips finished files; reconnecting never re-transcribes completed sessions.\n",
    "- \U0001f440 **Built-in Transcripts Studio**: Read notes with formatted timestamps, copy to clipboard, or inspect subtitles right in the browser.\n",
    "\n",
    "---\n",
    "### \u26a1 Quick Start\n",
    "1. **Runtime \u2192 Change runtime type \u2192 GPU** (Choose **A100**, **L4**, or **T4**).\n",
    "2. (Optional) Add your Hugging Face token in **Colab Secrets** (\U0001f511 icon on left) named `HF_TOKEN` with Notebook access ON.\n",
    "3. Run **Step 1** (Setup & Mount Drive), then run **Step 2** to launch the interactive Studio!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: One-Click Environment Setup & Drive Mount\n",
    "Installs PyAV 18.1.0, faster-whisper, and CUDA 12 libraries, then mounts Google Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title \U0001f527 Install Dependencies & Mount Drive\n",
    "!pip install -q -U faster-whisper \"av==18.1.0\" nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"\n",
    "\n",
    "import os, sys, glob, ctypes, logging\n",
    "from pathlib import Path\n",
    "\n",
    "# 1. Load CUDA 12 libraries\n",
    "try:\n",
    "    import nvidia.cublas.lib, nvidia.cudnn.lib\n",
    "    libs = []\n",
    "    for pkg in (nvidia.cublas.lib, nvidia.cudnn.lib):\n",
    "        d = list(pkg.__path__)[0]\n",
    "        libs += sorted(glob.glob(os.path.join(d, \"*.so*\")))\n",
    "    for _ in range(2):\n",
    "        for so in libs:\n",
    "            try:\n",
    "                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)\n",
    "            except OSError:\n",
    "                pass\n",
    "    ctypes.CDLL(\"libcublas.so.12\")\n",
    "    print(\"\u2705 CUDA 12 libraries loaded OK\")\n",
    "except Exception as e:\n",
    "    print(f\"\u26a0\ufe0f CUDA 12 load note: {e}\")\n",
    "\n",
    "# 2. Mount Google Drive\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    print(\"\u2705 Google Drive mounted at /content/drive\")\n",
    "except Exception as e:\n",
    "    print(f\"Note: Drive mount skipped ({e})\")\n",
    "\n",
    "# 3. Optional HF Token\n",
    "try:\n",
    "    from google.colab import userdata\n",
    "    hf = userdata.get(\"HF_TOKEN\")\n",
    "    if hf:\n",
    "        os.environ[\"HF_TOKEN\"] = hf\n",
    "        print(\"\u2705 HF_TOKEN loaded from Colab Secrets.\")\n",
    "except Exception:\n",
    "    logging.getLogger(\"huggingface_hub.utils._http\").setLevel(logging.ERROR)\n",
    "\n",
    "# 4. Check GPU\n",
    "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Launch Interactive Studio Web UI\n",
    "Launches the modern all-in-one Web UI directly inside this notebook cell, with a button to pop it into a full browser window."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title \U0001f39b\ufe0f Launch Whisper & Video Studio\n",
    "import os, zlib, base64\n",
    "from pathlib import Path\n",
    "\n",
    "# Self-extract colab_studio.py if not already present\n",
    "if not Path(\"colab_studio.py\").exists():\n",
    "    drive_studio = Path(\"/content/drive/MyDrive/colab_studio.py\")\n",
    "    if drive_studio.exists():\n",
    "        import shutil\n",
    "        shutil.copy(drive_studio, \"colab_studio.py\")\n",
    "    else:\n",
    "        payload = \"" + b64_payload + "\"\n",
    "        Path(\"colab_studio.py\").write_bytes(zlib.decompress(base64.b64decode(payload)))\n",
    "        print(\"\u2705 colab_studio.py unpacked successfully!\")\n",
    "\n",
    "# Launch studio inside notebook cell\n",
    "from colab_studio import launch_in_colab\n",
    "launch_in_colab(port=8765, height=950)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: (Optional) Headless Batch Transcriber\n",
    "If you want to run batch transcription in the background without keeping the interactive web UI open, configure the form below and run this cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title \u2699\ufe0f Headless Batch Configuration\n",
    "from pathlib import Path\n",
    "import os, re, time, shutil\n",
    "from faster_whisper import WhisperModel, BatchedInferencePipeline\n",
    "\n",
    "# ---------------- SETTINGS ----------------\n",
    "AUDIO_DIR     = \"/content/drive/MyDrive/transcribe/audio\" #@param {type:\"string\"}\n",
    "OUT_DIR       = \"/content/drive/MyDrive/transcribe/transcripts_large_v3\" #@param {type:\"string\"}\n",
    "MODEL_NAME    = \"large-v3\" #@param [\"large-v3\", \"distil-large-v3\", \"large-v2\", \"medium\", \"small\"]\n",
    "LANGUAGE      = \"en\" #@param {type:\"string\"}\n",
    "COMPUTE_TYPE  = \"float16\" #@param [\"float16\", \"int8_float16\", \"int8\", \"bfloat16\"]\n",
    "USE_BATCHED   = True #@param {type:\"boolean\"}\n",
    "BATCH_SIZE    = 16 #@param {type:\"integer\"}\n",
    "PARA_SECONDS  = 45 #@param {type:\"integer\"}\n",
    "SESSION_START = 1 #@param {type:\"integer\"}\n",
    "SESSION_END   = 999 #@param {type:\"integer\"}\n",
    "MAX_HOURS     = 10 #@param {type:\"number\"}\n",
    "HOTWORDS      = \"Naresh IT, Veera Babu, Kubernetes, kubectl, Terraform, AWS, EC2, S3, IAM, VPC, Docker, Jenkins, Helm, CI/CD, GitHub Actions, Ansible, Prometheus, Grafana, Azure, GCP, YAML, DevOps, SRE\" #@param {type:\"string\"}\n",
    "# ------------------------------------------\n",
    "\n",
    "audio_path = Path(AUDIO_DIR)\n",
    "out_path = Path(OUT_DIR)\n",
    "out_path.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "KEYWORD_RE = re.compile(r\"\\b(?:session|day|class|lecture|lesson|part|episode|ep|video|module|chapter|week)[\\s._#-]*(\\d+)\", re.I)\n",
    "def file_num(stem):\n",
    "    m = KEYWORD_RE.search(stem)\n",
    "    if m: return int(m.group(1))\n",
    "    m = re.search(r\"\\d+\", stem)\n",
    "    return int(m.group()) if m else None\n",
    "\n",
    "def natural(s):\n",
    "    return [int(t) if t.isdigit() else t.lower() for t in re.split(r\"(\\d+)\", s)]\n",
    "\n",
    "def ts(sec, srt=False):\n",
    "    h, rem = divmod(int(sec), 3600)\n",
    "    m, s = divmod(rem, 60)\n",
    "    if srt:\n",
    "        ms = int((sec - int(sec)) * 1000)\n",
    "        return f\"{h:02d}:{m:02d}:{s:02d},{ms:03d}\"\n",
    "    return f\"{h:02d}:{m:02d}:{s:02d}\"\n",
    "\n",
    "video_exts = {\".mp4\", \".mkv\", \".webm\", \".avi\", \".mov\", \".flv\", \".wmv\", \".m4v\", \".ts\", \".mts\", \".m2ts\", \".mpg\", \".mpeg\", \".3gp\", \".3g2\", \".ogv\", \".vob\", \".rm\", \".rmvb\", \".asf\", \".divx\", \".f4v\", \".m2v\", \".mpv\", \".qt\", \".tod\", \".mod\"}\n",
    "audio_exts = {\".mp3\", \".m4a\", \".wav\", \".aac\", \".flac\", \".ogg\", \".opus\", \".wma\", \".aiff\", \".aif\", \".aifc\", \".alac\", \".mka\", \".ac3\", \".dts\", \".amr\", \".caf\", \".mp2\", \".m4b\", \".m4p\", \".weba\"}\n",
    "exts = video_exts | audio_exts\n",
    "files = [p for p in audio_path.iterdir() if p.suffix.lower() in exts]\n",
    "files.sort(key=lambda p: (file_num(p.stem) is None, file_num(p.stem) or 0, natural(p.name)))\n",
    "\n",
    "# Filter by range\n",
    "files = [p for p in files if (file_num(p.stem) is None) or (SESSION_START <= file_num(p.stem) <= SESSION_END)]\n",
    "todo = [p for p in files if not (out_path / f\"{p.stem}.txt\").exists()]\n",
    "\n",
    "print(f\"\U0001f4ca Total matching files: {len(files)} | Already finished: {len(files)-len(todo)} | Remaining to transcribe: {len(todo)}\")\n",
    "\n",
    "if todo:\n",
    "    print(f\"Loading WhisperModel ({MODEL_NAME} on cuda / {COMPUTE_TYPE})...\")\n",
    "    model = WhisperModel(MODEL_NAME, device=\"cuda\", compute_type=COMPUTE_TYPE)\n",
    "    batched = BatchedInferencePipeline(model=model) if USE_BATCHED else None\n",
    "    T0 = time.time()\n",
    "    ok, failed, total_audio_min = 0, [], 0.0\n",
    "\n",
    "    for n, f in enumerate(todo, 1):\n",
    "        if MAX_HOURS and (time.time() - T0) > MAX_HOURS * 3600:\n",
    "            print(f\"\\n\u23f0 Reached MAX_HOURS ({MAX_HOURS} h). Clean stop.\")\n",
    "            break\n",
    "        \n",
    "        t0 = time.time()\n",
    "        is_aud = f.suffix.lower() in audio_exts\n",
    "        local = Path(\"/content\") / (f.name if is_aud else f\"{f.stem}.mp3\")\n",
    "        try:\n",
    "            if is_aud:\n",
    "                shutil.copy(f, local)  # Direct audio file, no conversion needed\n",
    "            else:\n",
    "                import subprocess\n",
    "                subprocess.run([\"ffmpeg\", \"-y\", \"-hide_banner\", \"-loglevel\", \"error\", \"-i\", str(f), \"-vn\", \"-map\", \"0:a:0\", \"-ac\", \"1\", \"-ar\", \"16000\", \"-b:a\", \"32k\", str(local)], check=True)\n",
    "            \n",
    "            if USE_BATCHED:\n",
    "                segments, info = batched.transcribe(str(local), language=LANGUAGE, batch_size=BATCH_SIZE, beam_size=5, initial_prompt=HOTWORDS)\n",
    "            else:\n",
    "                segments, info = model.transcribe(str(local), language=LANGUAGE, vad_filter=True, beam_size=5, condition_on_previous_text=False, initial_prompt=HOTWORDS)\n",
    "            \n",
    "            txt_tmp = out_path / f\"{f.stem}.txt.part\"\n",
    "            srt_tmp = out_path / f\"{f.stem}.srt.part\"\n",
    "            md_tmp  = out_path / f\"{f.stem}.md.part\"\n",
    "            paras = []\n",
    "\n",
    "            with open(txt_tmp, \"w\", encoding=\"utf-8\") as ft, open(srt_tmp, \"w\", encoding=\"utf-8\") as fs:\n",
    "                for i, s in enumerate(segments, 1):\n",
    "                    text = s.text.strip()\n",
    "                    ft.write(f\"[{ts(s.start)}] {text}\\n\")\n",
    "                    fs.write(f\"{i}\\n{ts(s.start, True)} --> {ts(s.end, True)}\\n{text}\\n\\n\")\n",
    "                    if not paras or s.start - paras[-1][0] >= PARA_SECONDS:\n",
    "                        paras.append([s.start, []])\n",
    "                    paras[-1][1].append(text)\n",
    "\n",
    "            with open(md_tmp, \"w\", encoding=\"utf-8\") as fm:\n",
    "                fm.write(f\"# {f.stem}\\n\\n> **Duration:** {ts(info.duration)} | **Model:** {MODEL_NAME}\\n\\n---\\n\\n\")\n",
    "                for start_sec, texts in paras:\n",
    "                    fm.write(f\"**[{ts(start_sec)}]** \" + \" \".join(texts) + \"\\n\\n\")\n",
    "\n",
    "            srt_tmp.rename(out_path / f\"{f.stem}.srt\")\n",
    "            md_tmp.rename(out_path / f\"{f.stem}.md\")\n",
    "            txt_tmp.rename(out_path / f\"{f.stem}.txt\")\n",
    "\n",
    "            took = time.time() - t0\n",
    "            dur = info.duration\n",
    "            ok += 1\n",
    "            total_audio_min += dur / 60\n",
    "            avg_took = (time.time() - T0) / ok\n",
    "            left = len(todo) - n\n",
    "            print(f\"[{n}/{len(todo)}] \u2705 {f.name} | {dur/60:.1f} min audio | took {took/60:.1f} min ({dur/took:.1f}x) | ETA: {left*avg_took/3600:.1f} h left\")\n",
    "        except Exception as e:\n",
    "            failed.append(f.name)\n",
    "            print(f\"[{n}/{len(todo)}] \u274c FAILED: {f.name} -> {e}\")\n",
    "        finally:\n",
    "            local.unlink(missing_ok=True)\n",
    "\n",
    "    print(\"\\n================ SUMMARY ================\")\n",
    "    print(f\"Finished: {ok} file(s) | {total_audio_min/60:.1f} h audio in {(time.time()-T0)/3600:.2f} h total time\")\n",
    "    print(f\"Failed: {len(failed)} {failed if failed else ''}\")\n",
    "    remaining = len([p for p in files if not (out_path / f\"{p.stem}.txt\").exists()])\n",
    "    print(f\"Remaining: {remaining}\")\n",
    "else:\n",
    "    print(\"\U0001f389 All files in the selected range are already transcribed!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Spot-Check Recent Transcripts\n",
    "Quickly print the top of the most recently generated `.txt` transcript and `.md` study notes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# @title \U0001f50d Preview Latest Transcript\n",
    "from pathlib import Path\n",
    "out_dir = Path(\"/content/drive/MyDrive/transcribe/transcripts_large_v3\")\n",
    "latest = max(out_dir.glob(\"*.txt\"), key=lambda p: p.stat().st_mtime, default=None)\n",
    "if latest:\n",
    "    print(f\"\U0001f4c4 Latest File: {latest.name}\\n\")\n",
    "    print(\"\".join(open(latest, encoding=\"utf-8\").readlines()[:25]))\n",
    "    md_file = out_dir / f\"{latest.stem}.md\"\n",
    "    if md_file.exists():\n",
    "        print(f\"\\n\U0001f4d6 Study Notes Preview ({md_file.name}):\\n\")\n",
    "        print(\"\".join(open(md_file, encoding=\"utf-8\").readlines()[:20]))\n",
    "else:\n",
    "    print(\"No transcripts found yet in\", out_dir)"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

Path("Whisper_Video_Studio.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Whisper_Video_Studio.ipynb created successfully!")
