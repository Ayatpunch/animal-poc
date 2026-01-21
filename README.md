# Animal Detection POC - Week 1

Proof of concept for wildlife detection from CCTV/trail camera footage using YOLO object detection.

## 📋 Overview

This project validates an end-to-end pipeline for detecting animal presence in video footage (MP4 files and RTSP streams) and outputting structured JSON events. Tested with both daylight pet footage and night/IR wildlife footage.

**Status:** Week 1 POC Complete ✅  
**Ready for:** Week 2 custom model training

---

## 🚀 Quick Start

### Watch Detections in Real-Time

```powershell
# 1. Activate virtual environment
.\activate_venv.ps1

# 2. Watch wildlife IR footage with bounding boxes
python infer_visualize.py --source night_ir.mp4 --conf 0.35
```

**A desktop window will appear showing:**
- 🟢 Green boxes around detected animals
- 🔵 Blue boxes around other objects
- Class labels and confidence scores
- Press `Q` to quit, `SPACE` to pause

---

## 🗂️ Project Structure

```
animal-poc/
├── infer_visualize.py       # 🎥 Visual detection with bounding boxes (PRIMARY)
├── infer.py                 # 📊 Headless inference for production/batch
├── notes.md                 # Detailed findings and observations (359 lines)
├── product.md               # Original requirements and specifications
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── activate_venv.bat        # Activate venv (Windows CMD)
├── activate_venv.ps1        # Activate venv (Windows PowerShell)
├── activate_venv.sh         # Activate venv (Linux/Mac)
├── mediamtx.yml            # MediaMTX RTSP server config
├── venv/                    # Python virtual environment (gitignored)
├── yolo11n.pt              # Pre-trained YOLO11n model weights (gitignored)
├── *.mp4                   # Video files (gitignored)
└── *.jsonl                 # Output event files (gitignored)
```

**Primary Tools:**
- **`infer_visualize.py`** - Visual detection with real-time display (recommended for development/testing)
- **`infer.py`** - Headless processing for production deployments

**Note:** Large files (videos, model weights, outputs) are gitignored. See setup instructions for obtaining them.

---

## 🔧 Prerequisites

- **Python 3.12+**
- **FFmpeg** (optional, for RTSP testing)
- **4GB+ RAM** (for inference)
- **No GPU required** (CPU-only is sufficient for POC)

---

## 🚀 Setup Instructions

### 1. Clone/Navigate to Project

```bash
cd animal-poc
```

### 2. Create Virtual Environment (if not exists)

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\activate_venv.ps1
```

**Windows CMD:**
```cmd
activate_venv.bat
```

**Linux/Mac:**
```bash
source activate_venv.sh
```

### 4. Install Dependencies

```bash
pip install ultralytics opencv-python
```

### 5. Download YOLO Model (if not exists)

The model will auto-download on first run, or manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

---

## 📹 Usage

### 🎥 Visual Detection (Recommended for Testing & Development)

**Watch detections in real-time with bounding boxes:**

```bash
# Activate virtual environment
.\activate_venv.ps1

# Watch wildlife IR footage with detections
python infer_visualize.py --source night_ir.mp4 --conf 0.35

# Watch daylight footage
python infer_visualize.py --source day_clip.mp4 --conf 0.35 --sample-every 2

# Watch live stream
python infer_visualize.py --source udp://127.0.0.1:1234 --conf 0.35
```

**Features:**
- 🟢 **Green boxes** = Animals detected
- 🔵 **Blue boxes** = Non-animal objects
- **Press `Q`** to quit
- **Press `SPACE`** to pause/resume
- Desktop window shows live detections with labels and confidence scores

---

### 📊 Headless Processing (Production/Batch Processing)

**Generate JSON events without display:**

```bash
# Night IR Wildlife Footage
python infer.py \
  --source night_ir.mp4 \
  --camera-id night_cam \
  --out night_events.jsonl \
  --model yolo11n.pt \
  --sample-every 10 \
  --conf 0.35

# Daylight Footage
python infer.py \
  --source day_clip.mp4 \
  --camera-id day_cam \
  --out day_events.jsonl \
  --model yolo11n.pt \
  --sample-every 10 \
  --conf 0.35

# RTSP Stream (Live Camera)
python infer.py \
  --source rtsp://username:password@192.168.1.100:554/stream \
  --camera-id trail_cam_01 \
  --out live_events.jsonl \
  --model yolo11n.pt \
  --sample-every 10 \
  --conf 0.35 \
  --max-frames 1000
```

**Note:** Use `--max-frames` to limit processing (0 = infinite)

---

## ⚙️ Command-Line Arguments

### `infer_visualize.py` (Visual Detection - PRIMARY)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--source` | ✅ | - | Video file path, RTSP URL, or UDP stream |
| `--model` | ❌ | `yolo11n.pt` | YOLO model weights file |
| `--conf` | ❌ | `0.35` | Confidence threshold (0.0-1.0) |
| `--sample-every` | ❌ | `1` | Show 1 in every N frames (1 = all frames) |
| `--max-frames` | ❌ | `0` | Maximum frames to process (0 = no limit) |

**Interactive Controls:**
- `Q` - Quit application
- `SPACE` - Pause/Resume video

**Visual Indicators:**
- 🟢 Green boxes - Animals (dog, cat, horse, sheep, cow, bear, zebra, giraffe, elephant)
- 🔵 Blue boxes - Non-animal objects
- Labels show class name and confidence score

---

### `infer.py` (Headless Processing)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--source` | ✅ | - | Video file path, RTSP URL, or UDP stream |
| `--camera-id` | ✅ | - | Camera identifier for JSON output |
| `--out` | ❌ | `events.jsonl` | Output JSONL file path |
| `--model` | ❌ | `yolo11n.pt` | YOLO model weights file |
| `--sample-every` | ❌ | `10` | Process 1 in every N frames |
| `--conf` | ❌ | `0.35` | Confidence threshold (0.0-1.0) |
| `--max-frames` | ❌ | `0` | Maximum frames to process (0 = no limit) |

### Visual Detection Examples

**See misclassifications in wildlife IR:**
```bash
python infer_visualize.py --source night_ir.mp4 --conf 0.35 --max-frames 300
```

**Slow motion frame-by-frame:**
```bash
python infer_visualize.py --source night_ir.mp4 --conf 0.35 --sample-every 1
```

**Watch live stream:**
```bash
python infer_visualize.py --source udp://127.0.0.1:1234 --conf 0.35
```

**Higher confidence (fewer detections):**
```bash
python infer_visualize.py --source day_clip.mp4 --conf 0.6
```

### Headless Processing Examples

**Process every frame:**
```bash
python infer.py --source video.mp4 --camera-id cam_01 --sample-every 1 --out all_frames.jsonl
```

**Higher confidence threshold:**
```bash
python infer.py --source video.mp4 --camera-id cam_01 --conf 0.5 --out high_conf.jsonl
```

**Limit to 500 frames:**
```bash
python infer.py --source rtsp://... --camera-id cam_01 --max-frames 500 --out test.jsonl
```

---

## 📄 JSON Output Format

Each detection event is written as a JSON object (one per line):

```json
{
  "timestamp": "2026-01-20T13:47:44.610825+00:00",
  "source": "cam_night_01",
  "animal_present": true,
  "raw_classes": ["cat", "horse"],
  "confidence": 0.787
}
```

### Fields

- **`timestamp`** - ISO 8601 UTC timestamp when frame was processed
- **`source`** - Camera ID from `--camera-id` parameter
- **`animal_present`** - Boolean, true if any animal detected (based on ANIMAL_CLASSES)
- **`raw_classes`** - Array of all detected class names from YOLO
- **`confidence`** - Highest confidence score (0.0-1.0, rounded to 4 decimals)

### Animal Classes (Week 1)

Currently detects these as animals:
```python
ANIMAL_CLASSES = {
    "dog", "cat", "horse", "sheep", "cow",
    "bear", "zebra", "giraffe", "elephant"
}
```

**Note:** Week 2 will expand to include wildlife species (wolf, fox, owl, deer, etc.)

---

## 📊 Test Results

### Night IR Wildlife (`night_ir.mp4`)
- **Events generated:** 696
- **Frame sampling:** Every 10th frame
- **Findings:** High misclassification rate, IR artifacts detected as objects
- **Accuracy:** <20% (requires custom training)

### Daylight Pets (`day_clip.mp4`)
- **Events generated:** 3,576
- **Frame sampling:** Every 10th frame
- **Findings:** Good detection of people, vehicles, pets
- **Accuracy:** ~70-85% (COCO domain match)

**For detailed analysis, see [`notes.md`](notes.md)**

---

## 🔍 Known Issues & Limitations

### Week 1 POC Limitations

1. **Missing Wildlife Species**
   - Wolf, fox, owl, deer, raccoon not in COCO dataset
   - Misclassified as domestic animals (wolf→cat, deer→horse)

2. **IR/Night Vision Issues**
   - IR glare detected as "banana" (65-89% confidence)
   - Terrain textures misclassified as objects (bed, cake, bowl)
   - High false positive rate

3. **False Negatives**
   - Misses some cats and birds even in good lighting
   - Detection rate ~70-85% on COCO domain objects

4. **COCO Domain Only**
   - Pre-trained on urban/pet scenarios
   - Not trained for wildlife or IR footage

**These are expected Week 1 findings. Week 2 will address with custom training.**

---

## 🎯 Week 1 Success Criteria

- [x] Animal presence detected from CCTV footage
- [x] Visual detection tool with real-time bounding boxes
- [x] JSON events generated reliably (4,298+ total events)
- [x] Network streaming validated (UDP tested, RTSP ready)
- [x] Night IR failure modes understood and documented
- [x] Pipeline ready for custom training
- [x] No blockers for Week 2

---

## 📈 Performance

- **Inference speed:** ~10-30ms per frame (CPU only)
- **Memory usage:** Stable across 4,272 events
- **Reliability:** No crashes, consistent JSON output
- **Scalability:** Handles both short and long videos

---

## 🔄 Network Streaming Support

The pipeline supports live network streams (UDP, RTSP) from cameras:

### RTSP Streams (Trail Cameras)
```bash
python infer.py \
  --source rtsp://user:pass@camera-ip:554/stream \
  --camera-id trail_cam_01 \
  --out live_events.jsonl \
  --max-frames 0
```

### UDP Streams (Tested ✅)
```bash
# Stream from file via FFmpeg (for testing)
ffmpeg -re -stream_loop -1 -i video.mp4 -c:v mpeg2video -f mpegts udp://127.0.0.1:1234

# Process stream with infer.py
python infer.py --source udp://127.0.0.1:1234 --camera-id test_stream --out stream_events.jsonl --max-frames 100
```

**Week 1 Testing:**
- ✅ UDP streaming tested with wildlife IR footage
- ✅ 10 events generated from 100 frames
- ✅ Real-time JSON output validated
- ✅ Detection quality identical to MP4 files

**Production deployment notes:**
- Use `--max-frames 0` for continuous processing
- Monitor output file size (grows continuously)
- Consider log rotation for long-running streams
- Network latency may affect frame timing

---

## 📚 Documentation

- **[notes.md](notes.md)** - Comprehensive Week 1 findings (353 lines)
  - Detailed detection results
  - False positive analysis
  - Misclassification patterns
  - Week 2 requirements
  
- **[product.md](product.md)** - Original requirements and specifications
  - Week 1 scope and deliverables
  - Success criteria
  - Technical approach

---

## 🚧 Next Steps (Week 2)

1. **Collect Training Data**
   - 1,000+ labeled IR/wildlife frames
   - Species: wolf, fox, owl, deer, raccoon, coyote, bobcat
   - Include negative examples (IR artifacts, empty scenes)

2. **Train Custom Model**
   - Fine-tune YOLO11 on wildlife dataset
   - Focus on IR/night vision conditions
   - Minimize false positives (banana, bed, etc.)

3. **Validate Improvements**
   - Measure accuracy on test footage
   - Compare against Week 1 baseline
   - Deploy to production if >85% accuracy

---

## 🛠️ Troubleshooting

### Virtual Environment Issues

**Windows PowerShell execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Manual activation:**
```powershell
.\venv\Scripts\python.exe infer.py --source video.mp4 --camera-id cam_01
```

### Module Not Found Errors

Ensure venv is activated and dependencies installed:
```bash
pip install ultralytics opencv-python
```

### RTSP Connection Issues

- Verify camera IP and credentials
- Check network connectivity
- Try different RTSP transport (TCP vs UDP)
- Some cameras require specific RTSP paths

### Performance Issues

- Increase `--sample-every` (e.g., 30) to process fewer frames
- Use smaller model: `yolo11n.pt` (nano, fastest)
- Close other applications to free RAM

---

## 📝 License

POC project - internal use only.

---

## 👥 Contact

For questions about Week 1 findings or Week 2 planning, refer to `notes.md` for detailed documentation.

**Week 1 Status:** COMPLETE ✅  
**Date:** 2026-01-20
