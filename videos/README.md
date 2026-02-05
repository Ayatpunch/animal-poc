# Videos Folder

Place your video files here for replay testing and threshold sweep analysis.

## Supported Formats

- MP4 (recommended)
- AVI
- MOV
- Any format supported by OpenCV

## How to Add Videos

### Option 1: AnyDesk File Transfer
1. Open AnyDesk and connect to your remote machine
2. Click the file transfer icon (folder) or press Ctrl+F
3. Navigate to this `videos/` folder on the local side
4. Drag video files from the remote machine

### Option 2: Direct Copy
```bash
cp /path/to/your/videos/*.mp4 videos/
```

### Option 3: Download from Cloud
If videos are on Google Drive or similar:
1. Download to your local machine
2. Move to this `videos/` folder

## Naming Convention (Recommended)

For best results with the threshold sweep script, use descriptive names:

```
location_01_2024-01-15_night.mp4
location_03_2024-02-20_day.mp4
test_clip_coyote.mp4
```

## Usage

Once videos are added, run:

```bash
# Threshold sweep across all videos
python threshold_sweep.py --videos videos/ --model runs/detect/run_20260203_032801/weights/best.pt

# Single video inference
python infer_gated.py --source videos/your_video.mp4 --camera-id test --model runs/detect/run_20260203_032801/weights/best.pt
```

## Note

Video files are excluded from git (too large). Make sure to back them up separately.
