"""
Threshold Sweep Script for Wildlife Detection

Runs inference on video files at multiple confidence thresholds to find
the optimal operating point for your deployment.

Reports:
- Detections per hour at each threshold
- Alert frequency (with event gating)
- Comparison table across thresholds

Usage:
    python threshold_sweep.py --videos videos/ --model best.pt
    python threshold_sweep.py --videos videos/clip.mp4 --model best.pt --thresholds 0.4 0.5 0.6 0.7 0.8
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO


# Default thresholds to test (per Week 3 plan)
DEFAULT_THRESHOLDS = [0.50, 0.65, 0.75, 0.85]

# Animal classes to count
ANIMAL_CLASSES = {"bear", "canine", "cervidae", "dog", "cat", "deer"}


class ThresholdSweepResult:
    """Store results for a single threshold run."""
    
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.total_frames = 0
        self.frames_with_detections = 0
        self.total_detections = 0
        self.detections_by_class = defaultdict(int)
        self.detection_timestamps = []  # Frame numbers with detections
        self.elapsed_seconds = 0.0
        self.video_duration_seconds = 0.0
    
    @property
    def detections_per_hour(self) -> float:
        if self.video_duration_seconds == 0:
            return 0
        hours = self.video_duration_seconds / 3600
        return self.total_detections / hours if hours > 0 else 0
    
    @property
    def detection_rate(self) -> float:
        """Percentage of frames with at least one detection."""
        if self.total_frames == 0:
            return 0
        return (self.frames_with_detections / self.total_frames) * 100
    
    def to_dict(self) -> Dict:
        return {
            "threshold": self.threshold,
            "total_frames": self.total_frames,
            "frames_with_detections": self.frames_with_detections,
            "total_detections": self.total_detections,
            "detections_by_class": dict(self.detections_by_class),
            "detections_per_hour": round(self.detections_per_hour, 1),
            "detection_rate_percent": round(self.detection_rate, 2),
            "processing_time_seconds": round(self.elapsed_seconds, 2),
            "video_duration_seconds": round(self.video_duration_seconds, 2),
        }


def run_threshold_sweep(
    video_path: str,
    model: YOLO,
    thresholds: List[float],
    sample_every: int = 5,
    max_frames: int = 0,
) -> Dict[float, ThresholdSweepResult]:
    """
    Run inference on a video at multiple thresholds.
    
    Returns dict mapping threshold -> ThresholdSweepResult
    """
    results = {}
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    print(f"\nVideo: {video_path}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}, Duration: {video_duration:.1f}s")
    print(f"  Sampling every {sample_every} frames")
    
    for threshold in thresholds:
        print(f"\n  Testing threshold {threshold}...", end=" ", flush=True)
        
        result = ThresholdSweepResult(threshold)
        result.video_duration_seconds = video_duration
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
            
            if frame_idx % sample_every != 0:
                continue
            
            result.total_frames += 1
            
            # Run inference at this threshold
            detections = model.predict(frame, conf=threshold, verbose=False)
            
            if detections[0].boxes is not None and len(detections[0].boxes) > 0:
                result.frames_with_detections += 1
                result.detection_timestamps.append(frame_idx)
                
                for box in detections[0].boxes:
                    cls_id = int(box.cls.item())
                    cls_name = detections[0].names.get(cls_id, str(cls_id))
                    
                    if cls_name in ANIMAL_CLASSES:
                        result.total_detections += 1
                        result.detections_by_class[cls_name] += 1
        
        cap.release()
        result.elapsed_seconds = time.time() - start_time
        
        print(f"Done. {result.total_detections} detections, {result.detections_per_hour:.0f}/hr")
        
        results[threshold] = result
    
    return results


def estimate_alert_rate(
    result: ThresholdSweepResult,
    min_hits: int = 2,
    hit_window: int = 5,
    cooldown_frames: int = 150,  # ~5 seconds at 30fps
) -> Tuple[int, float]:
    """
    Estimate alert rate after applying event gating logic.
    
    Returns (alert_count, alerts_per_hour)
    """
    if not result.detection_timestamps:
        return 0, 0.0
    
    alerts = 0
    last_alert_frame = -cooldown_frames
    hit_buffer = []
    
    for frame in result.detection_timestamps:
        # Add to buffer
        hit_buffer.append(frame)
        
        # Remove old frames from buffer
        hit_buffer = [f for f in hit_buffer if frame - f <= hit_window * 5]  # Approximate
        
        # Check if we should fire an alert
        if len(hit_buffer) >= min_hits and (frame - last_alert_frame) >= cooldown_frames:
            alerts += 1
            last_alert_frame = frame
    
    hours = result.video_duration_seconds / 3600 if result.video_duration_seconds > 0 else 1
    alerts_per_hour = alerts / hours if hours > 0 else 0
    
    return alerts, alerts_per_hour


def print_comparison_table(all_results: Dict[str, Dict[float, ThresholdSweepResult]]):
    """Print a comparison table across all videos and thresholds."""
    
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP COMPARISON")
    print("=" * 80)
    
    # Get all thresholds
    all_thresholds = set()
    for video_results in all_results.values():
        all_thresholds.update(video_results.keys())
    thresholds = sorted(all_thresholds)
    
    # Print header
    header = f"{'Video':<30} | " + " | ".join(f"{t:>8}" for t in thresholds)
    print(header)
    print("-" * len(header))
    
    # Detections per hour
    print("\n📊 Detections per Hour:")
    print("-" * len(header))
    for video_name, video_results in all_results.items():
        short_name = Path(video_name).stem[:28]
        values = [f"{video_results.get(t, ThresholdSweepResult(t)).detections_per_hour:>8.0f}" 
                  for t in thresholds]
        print(f"{short_name:<30} | " + " | ".join(values))
    
    # Alerts per hour (with gating)
    print("\n🚨 Alerts per Hour (with 2-in-5 gating):")
    print("-" * len(header))
    for video_name, video_results in all_results.items():
        short_name = Path(video_name).stem[:28]
        values = []
        for t in thresholds:
            result = video_results.get(t)
            if result:
                _, alerts_per_hour = estimate_alert_rate(result)
                values.append(f"{alerts_per_hour:>8.1f}")
            else:
                values.append(f"{'N/A':>8}")
        print(f"{short_name:<30} | " + " | ".join(values))
    
    # Detection rate
    print("\n📈 Detection Rate (% frames with detections):")
    print("-" * len(header))
    for video_name, video_results in all_results.items():
        short_name = Path(video_name).stem[:28]
        values = [f"{video_results.get(t, ThresholdSweepResult(t)).detection_rate:>7.1f}%" 
                  for t in thresholds]
        print(f"{short_name:<30} | " + " | ".join(values))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Aggregate across all videos
    for t in thresholds:
        total_detections = sum(
            video_results.get(t, ThresholdSweepResult(t)).total_detections 
            for video_results in all_results.values()
        )
        total_duration = sum(
            video_results.get(t, ThresholdSweepResult(t)).video_duration_seconds 
            for video_results in all_results.values()
        )
        hours = total_duration / 3600 if total_duration > 0 else 1
        det_per_hour = total_detections / hours if hours > 0 else 0
        
        total_alerts = sum(
            estimate_alert_rate(video_results.get(t, ThresholdSweepResult(t)))[0]
            for video_results in all_results.values()
        )
        alerts_per_hour = total_alerts / hours if hours > 0 else 0
        
        print(f"Threshold {t}: {total_detections} detections ({det_per_hour:.0f}/hr), "
              f"{total_alerts} gated alerts ({alerts_per_hour:.1f}/hr)")


def main():
    parser = argparse.ArgumentParser(description="Run threshold sweep on videos")
    
    parser.add_argument("--videos", required=True, 
                        help="Path to video file or folder containing videos")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                        help=f"Confidence thresholds to test (default: {DEFAULT_THRESHOLDS})")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Process 1 in N frames (default: 5)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0=unlimited)")
    parser.add_argument("--output", type=str, default="threshold_sweep_report.json",
                        help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Find videos
    videos_path = Path(args.videos)
    if videos_path.is_file():
        video_files = [videos_path]
    elif videos_path.is_dir():
        # Case-insensitive video file matching
        video_files = list(videos_path.glob("*.mp4")) + \
                      list(videos_path.glob("*.MP4")) + \
                      list(videos_path.glob("*.avi")) + \
                      list(videos_path.glob("*.AVI")) + \
                      list(videos_path.glob("*.mov")) + \
                      list(videos_path.glob("*.MOV"))
    else:
        print(f"Error: {args.videos} is not a valid file or directory")
        return
    
    if not video_files:
        print(f"No video files found in {args.videos}")
        print("\nPlease add video files to the videos/ folder.")
        print("Supported formats: .mp4, .avi, .mov")
        print("\nSee videos/README.md for instructions.")
        return
    
    print(f"\nFound {len(video_files)} video(s)")
    print(f"Testing thresholds: {args.thresholds}")
    print("=" * 60)
    
    # Run sweep on each video
    all_results = {}
    
    for video_file in video_files:
        try:
            results = run_threshold_sweep(
                str(video_file),
                model,
                args.thresholds,
                args.sample_every,
                args.max_frames,
            )
            all_results[str(video_file)] = results
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
    
    # Print comparison
    if all_results:
        print_comparison_table(all_results)
    
    # Save results
    output_data = {
        "model": args.model,
        "thresholds": args.thresholds,
        "sample_every": args.sample_every,
        "timestamp": datetime.now().isoformat(),
        "videos": {
            video: {
                str(t): result.to_dict()
                for t, result in video_results.items()
            }
            for video, video_results in all_results.items()
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
