# Week 3 Setup Documentation

## Summary

Set up the training infrastructure for wildlife detection model v1.0. Dataset prepared with 680 labeled images from Roboflow, code ready for GPU training.

---

## What Was Done

### 1. Folder Structure Created

```
animal-poc/
├── data/
│   ├── train/images/     # 544 training images
│   ├── train/labels/     # 544 YOLO format labels
│   ├── val/images/       # 136 validation images
│   ├── val/labels/       # 136 YOLO format labels
│   ├── test/             # Reserved for holdout testing
│   ├── holdout_a/        # Reserved for Viewtron footage
│   └── holdout_b/        # Reserved for different camera source
├── configs/
│   └── dataset.yaml      # YOLO training configuration
├── runs/                 # Training outputs will go here
├── snapshots/            # Alert images saved here
├── train.py              # Training script
├── infer_gated.py        # Smart inference with event gating
└── requirements.txt      # Dependencies
```

### 2. Dataset Prepared

- **Source:** Roboflow workspace (cctv-camera-pro-poc-v1)
- **Total images:** 680
- **Split:** 80% train (544) / 20% val (136)
- **Classes:**
  - `canine` (class 0) - wolves, coyotes, dogs, foxes
  - `cervidae` (class 1) - deer, elk, moose
  - `negative` (class 2) - bears, non-targets

### 3. Training Script (`train.py`)

Features:
- Fine-tunes YOLO from pretrained weights
- Configurable epochs, image size, batch size
- Saves artifacts: best.pt, last.pt, config, metrics
- Dataset version hashing for reproducibility

Usage:
```bash
python train.py --data configs/dataset.yaml --epochs 50
```

### 4. Gated Inference Script (`infer_gated.py`)

Features:
- Event gating: requires N detections in M frames before alert
- Object tracking (ByteTrack) to prevent duplicate alerts
- Saves snapshot images for every alert
- Full JSON output contract per Week 3 spec

Usage:
```bash
python infer_gated.py --source video.mp4 --camera-id cam_01 --track
```

### 5. Environment Setup

- Python 3.12 virtual environment
- Dependencies: ultralytics, opencv-python, torch 2.2.2
- Tested and verified all imports work

---

## What's Needed Next

### GPU for Training

Training requires a GPU. Options:
- **Lambda Labs** - RTX 4090 (~$1.10/hr)
- **RunPod** - RTX 4090 (~$0.74/hr)
- **Google Colab Pro** - A100 (~$10/month subscription)

Estimated training time: 2-4 hours for 50 epochs on 680 images.

### Holdout Data

Per Week 3 plan, need to set up:
- Holdout A: Viewtron/Mike footage (for external validation)
- Holdout B: Different camera/environment footage

### After Training

1. Run evaluation on holdouts
2. Replay inference on RTSP clips
3. Threshold sweep report (0.50 / 0.65 / 0.75 / 0.85)
4. FPS/latency benchmarking

---

## Files Changed/Created

| File | Status | Description |
|------|--------|-------------|
| `train.py` | Created | YOLO training script |
| `infer_gated.py` | Created | Smart inference with gating |
| `configs/dataset.yaml` | Created | Dataset configuration |
| `data/README.md` | Created | Data organization guide |
| `requirements.txt` | Updated | Dependencies list |
| `.gitignore` | Updated | Added data/runs/snapshots |
| `data/train/*` | Added | 544 images + labels |
| `data/val/*` | Added | 136 images + labels |

---

## Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Train model (GPU)
python train.py --data configs/dataset.yaml --epochs 50 --device 0

# Train model (CPU - slow, for testing only)
python train.py --data configs/dataset.yaml --epochs 2 --device cpu

# Run inference with gating
python infer_gated.py --source video.mp4 --camera-id cam_01 --track

# Run inference with custom model
python infer_gated.py --source video.mp4 --camera-id cam_01 --model runs/detect/run_xxx/weights/best.pt
```

---

**Date:** 2026-01-27
**Status:** Ready for GPU training
