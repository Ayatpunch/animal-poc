# Week 1 — Animal Presence Inference POC

## Purpose

This document defines **exactly what we are building in Week 1** and the **step-by-step instructions** to do it.

This is a **software engineering proof-of-concept**, not an AI research task.

The goal is to validate the **end-to-end inference pipeline**:

**CCTV / RTSP video → object detection → structured JSON events**

Accuracy, training, and production integrations come later.

---

## What Success Looks Like (Week 1)

By the end of Week 1, we must be able to say:

* Animal presence can be detected from CCTV-style video
* The system works on **daylight and night/IR footage**
* JSON events are emitted reliably
* There are **no blockers** to starting custom training in Week 2

---

## Explicit Non-Goals (Do NOT do these)

* ❌ No custom model training
* ❌ No accuracy tuning
* ❌ No alerting, dashboards, or integrations
* ❌ No real-time guarantees
* ❌ No hardware optimization

YOLO is treated as a **temporary black-box inference engine**.

---

## High-Level Architecture

```
Video Source (MP4 or RTSP)
        ↓
Frame Sampling
        ↓
YOLO Inference (pretrained COCO weights)
        ↓
Detection Filtering (coarse)
        ↓
JSON Event Output
```

Single stream only. Realtime not required.

---

## Detection Scope (Week 1)

### Output Concept

We only care about **animal presence**, not exact species.

Internal boolean:

```json
animal_present = true | false
```

### Temporary Internal Grouping (Optional)

* `canine_like`
* `cervidae_like`

These are **approximate** and **not final labels**.

### Raw Classes

We keep YOLO/COCO class names verbatim in output for later analysis.

---

## JSON Output Schema (POC)

Each processed frame emits **one JSON event** (JSONL format):

```json
{
  "timestamp": "2026-01-20T14:03:21.123Z",
  "source": "camera_id",
  "animal_present": true,
  "raw_classes": ["dog", "cat"],
  "confidence": 0.72
}
```

Notes:

* Multiple events per clip are expected
* Schema **must remain consistent**
* JSONL (one JSON per line) is intentional

---

## Tools & Constraints

* OS: Windows
* Python: **3.12** (not 3.13)
* GPU: **CPU only** (no NVIDIA / CUDA)
* Model: **Ultralytics YOLO11 or YOLO12**

Recommended for CPU:

* `yolo11n.pt` (fastest, stable)

---

## Step-by-Step Implementation Guide

### Step 1 — Project Setup

```powershell
mkdir animal-poc
cd animal-poc
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

Expected:

```
Python 3.12.x
```

---

### Step 2 — Install Dependencies

```powershell
pip install --upgrade pip
pip install ultralytics opencv-python
```

Sanity check:

```powershell
python -c "import cv2; from ultralytics import YOLO; print('ok')"
```

---

### Step 3 — Download YOLO Weights

YOLO11 (recommended):

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

YOLO12 (optional):

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt')"
```

---

### Step 4 — Inference Script (`infer.py`)

The script must:

1. Accept MP4 or RTSP input
2. Sample frames
3. Run YOLO inference
4. Filter detections
5. Emit JSON events

Key behaviors:

* Single stream only
* No real-time requirement
* Works on night/IR footage

(See `infer.py` in repo — this is the primary deliverable.)

---

### Step 5 — Test on MP4 (Day)

```powershell
python infer.py \
  --source .\day_clip.mp4 \
  --camera-id cam_day_01 \
  --out day_events.jsonl \
  --model yolo11n.pt
```

Verify output:

```powershell
Get-Content day_events.jsonl -TotalCount 5
```

---

### Step 6 — Test on MP4 (Night / IR)

```powershell
python infer.py \
  --source .\night_ir.mp4 \
  --camera-id cam_night_01 \
  --out night_events.jsonl \
  --model yolo11n.pt
```

This step is **mandatory**.

---

### Step 7 — Test on RTSP (Single Stream)

```powershell
python infer.py \
  --source "rtsp://user:pass@ip:554/..." \
  --camera-id gate_cam \
  --out rtsp_events.jsonl \
  --model yolo11n.pt \
  --max-frames 2000
```

If RTSP is unstable:

* Record a short RTSP segment to MP4
* Run inference on the MP4

Still counts for Week 1.

---

## False Positive Observation (Required)

Do **not** fix errors.

Record observations only:

* IR noise
* Shadows
* Vegetation movement
* Reflections

Create `notes.md` with bullet points:

* What YOLO detects well
* What it misses
* Common false positives

---

## Deliverables Checklist (End of Week)

* ✅ `infer.py` working on MP4 + RTSP
* ✅ Day JSON events
* ✅ Night / IR JSON events
* ✅ `notes.md`
* ✅ Confirmation: pipeline ready for custom training

---

## What Happens in Week 2 (Context Only)

Week 2 will introduce:

* Labeled CCTV frames
* Custom model training
* Accuracy improvements

Week 1 exists **only** to unblock this.

---

## Summary (One Sentence)

> Week 1 proves we can reliably turn CCTV video into structured animal-presence events using a pretrained model, including night IR, with no training or optimization.
