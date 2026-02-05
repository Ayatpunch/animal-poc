"""
Gated Inference Script with Event Tracking and Snapshots

This is the "smart" version of infer.py that:
1. Requires N detections in M frames before firing an alert (reduces spam)
2. Tracks objects to avoid duplicate alerts for the same animal
3. Saves snapshot images when alerts fire
4. Outputs the full JSON contract required for integration

Usage:
    # Basic usage with event gating
    python infer_gated.py --source video.mp4 --camera-id cam_01
    
    # Adjust gating sensitivity (require 3 hits in 7 frames)
    python infer_gated.py --source video.mp4 --camera-id cam_01 --min-hits 3 --hit-window 7
    
    # With object tracking to dedupe
    python infer_gated.py --source video.mp4 --camera-id cam_01 --track
    
    # Process RTSP stream
    python infer_gated.py --source rtsp://user:pass@ip:554/stream --camera-id outdoor_cam
"""

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO


# Animal classes that should trigger alerts
# These match the classes trained in your custom model (v2)
ANIMAL_CLASSES = {
    "bear",      # bears - ALERT
    "canine",    # wolves, coyotes, dogs, foxes - ALERT
    "cervidae",  # deer, elk, moose - ALERT
    # Also keep COCO classes for backward compatibility with pretrained models
    "dog", "cat", "horse", "sheep", "cow", "zebra", "giraffe", "elephant", "deer"
}

# Classes to ignore (won't trigger alerts)
IGNORE_CLASSES = {
    "negative",  # for future use - non-target detections
    "human", "vehicle", "person", "car", "truck", "bicycle", "motorcycle"
}


def iso_utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class DetectionBuffer:
    """
    Sliding window buffer to implement N-of-M gating.
    
    Only fires an alert when we see at least `min_hits` detections
    within the last `window_size` frames. This prevents single-frame
    false positives from triggering alerts.
    """
    
    def __init__(self, window_size: int = 5, min_hits: int = 2):
        self.window_size = window_size
        self.min_hits = min_hits
        self.buffer = deque(maxlen=window_size)
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # seconds between alerts for same object
    
    def add_frame(self, has_detection: bool, detection_class: Optional[str] = None):
        """Add a frame result to the buffer."""
        self.buffer.append({
            "has_detection": has_detection,
            "class": detection_class,
            "time": time.time()
        })
    
    def should_alert(self) -> bool:
        """Check if we have enough hits to fire an alert."""
        if len(self.buffer) < self.min_hits:
            return False
        
        hits = sum(1 for frame in self.buffer if frame["has_detection"])
        
        if hits >= self.min_hits:
            # Check cooldown
            now = time.time()
            if now - self.last_alert_time >= self.alert_cooldown:
                self.last_alert_time = now
                return True
        
        return False
    
    def get_dominant_class(self) -> Optional[str]:
        """Get the most common class in recent detections."""
        classes = [f["class"] for f in self.buffer if f["class"] is not None]
        if not classes:
            return None
        # Return most common
        return max(set(classes), key=classes.count)


class ObjectTracker:
    """
    Simple object tracker to avoid duplicate alerts for the same animal.
    
    Uses track IDs from YOLO's built-in tracker (ByteTrack).
    Remembers which track IDs we've already alerted on.
    """
    
    def __init__(self, memory_seconds: float = 30.0):
        self.alerted_tracks = {}  # track_id -> last_alert_time
        self.memory_seconds = memory_seconds
    
    def should_alert_track(self, track_id: int) -> bool:
        """Check if we should alert for this track ID."""
        now = time.time()
        
        # Clean old entries
        self.alerted_tracks = {
            tid: t for tid, t in self.alerted_tracks.items()
            if now - t < self.memory_seconds
        }
        
        if track_id in self.alerted_tracks:
            return False
        
        self.alerted_tracks[track_id] = now
        return True


def save_snapshot(frame, output_dir: Path, camera_id: str, timestamp: str) -> str:
    """Save a snapshot image and return the path."""
    # Create filename from timestamp
    safe_timestamp = timestamp.replace(":", "-").replace("+", "_")
    filename = f"{camera_id}_{safe_timestamp}.jpg"
    filepath = output_dir / filename
    
    cv2.imwrite(str(filepath), frame)
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Gated inference with tracking and snapshots")
    
    # Input/Output
    parser.add_argument("--source", required=True, help="Video file, RTSP URL, or camera index")
    parser.add_argument("--camera-id", required=True, help="Camera identifier for output")
    parser.add_argument("--out", default="events.jsonl", help="Output JSONL file")
    parser.add_argument("--snapshot-dir", default="snapshots", help="Directory to save snapshots")
    
    # Model
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    parser.add_argument("--model-version", default="v1.0", help="Model version string for output")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    
    # Sampling
    parser.add_argument("--sample-every", type=int, default=5, help="Process 1 in N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=unlimited)")
    
    # Event gating
    parser.add_argument("--min-hits", type=int, default=2, 
                        help="Minimum detections required to fire alert")
    parser.add_argument("--hit-window", type=int, default=5,
                        help="Window size (frames) for hit counting")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Seconds between alerts for same area")
    
    # Tracking
    parser.add_argument("--track", action="store_true",
                        help="Enable object tracking (ByteTrack) to dedupe alerts")
    parser.add_argument("--track-memory", type=float, default=30.0,
                        help="Seconds to remember alerted track IDs")
    
    args = parser.parse_args()
    
    # Setup
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")
    
    # Get video properties for stats
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    model = YOLO(args.model)
    
    # Initialize gating and tracking
    detection_buffer = DetectionBuffer(
        window_size=args.hit_window,
        min_hits=args.min_hits
    )
    detection_buffer.alert_cooldown = args.cooldown
    
    tracker = ObjectTracker(memory_seconds=args.track_memory) if args.track else None
    
    # Stats
    frame_idx = 0
    events_written = 0
    alerts_fired = 0
    start_time = time.time()
    
    print(f"Starting gated inference on: {args.source}")
    print(f"Event gating: {args.min_hits} hits in {args.hit_window} frames")
    print(f"Tracking: {'enabled' if args.track else 'disabled'}")
    print("-" * 50)
    
    with open(args.out, "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break
            
            if frame_idx % args.sample_every != 0:
                continue
            
            # Run inference (with or without tracking)
            if args.track:
                results = model.track(frame, conf=args.conf, verbose=False, persist=True)
            else:
                results = model.predict(frame, conf=args.conf, verbose=False)
            
            r0 = results[0]
            
            # Extract detections
            detections = []
            has_animal = False
            best_animal_conf = 0.0
            best_animal_class = None
            
            if r0.boxes is not None and len(r0.boxes) > 0:
                for i, box in enumerate(r0.boxes):
                    cls_id = int(box.cls.item())
                    cls_name = r0.names.get(cls_id, str(cls_id))
                    conf = float(box.conf.item())
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    # Get track ID if tracking
                    track_id = None
                    if args.track and hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id.item())
                    
                    # Skip ignored classes
                    if cls_name in IGNORE_CLASSES:
                        continue
                    
                    detection = {
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x, 1) for x in bbox],
                    }
                    if track_id is not None:
                        detection["track_id"] = track_id
                    
                    detections.append(detection)
                    
                    # Check if it's an animal
                    if cls_name in ANIMAL_CLASSES:
                        has_animal = True
                        if conf > best_animal_conf:
                            best_animal_conf = conf
                            best_animal_class = cls_name
            
            # Update detection buffer
            detection_buffer.add_frame(has_animal, best_animal_class)
            
            # Check if we should fire an alert
            should_fire = detection_buffer.should_alert()
            
            # If tracking, also check track-level deduplication
            if should_fire and args.track and detections:
                # Find the best animal detection with a track ID
                animal_detections = [d for d in detections if d["class"] in ANIMAL_CLASSES]
                if animal_detections:
                    best_detection = max(animal_detections, key=lambda d: d["confidence"])
                    if "track_id" in best_detection:
                        if not tracker.should_alert_track(best_detection["track_id"]):
                            should_fire = False  # Already alerted this animal
            
            # Fire alert event
            if should_fire and has_animal:
                timestamp = iso_utc_now()
                
                # Save snapshot
                snapshot_path = save_snapshot(frame, snapshot_dir, args.camera_id, timestamp)
                
                # Build event JSON
                event = {
                    "timestamp": timestamp,
                    "camera_id": args.camera_id,
                    "animal_present": True,
                    "class": detection_buffer.get_dominant_class() or best_animal_class,
                    "confidence": round(best_animal_conf, 4),
                    "bboxes": [d["bbox"] for d in detections if d["class"] in ANIMAL_CLASSES],
                    "snapshot_path": snapshot_path,
                    "model_version": args.model_version,
                    "frame_number": frame_idx,
                    "all_detections": detections,  # Full detail for debugging
                }
                
                f.write(json.dumps(event) + "\n")
                f.flush()
                events_written += 1
                alerts_fired += 1
                
                print(f"[ALERT] Frame {frame_idx}: {best_animal_class} ({best_animal_conf:.2f}) -> {snapshot_path}")
            
            # Progress update
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_idx} | Alerts: {alerts_fired} | FPS: {fps_actual:.1f}")
    
    cap.release()
    
    # Final stats
    elapsed = time.time() - start_time
    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
    
    print("\n" + "=" * 50)
    print("Inference Complete")
    print("=" * 50)
    print(f"Frames processed: {frame_idx}")
    print(f"Alerts fired:     {alerts_fired}")
    print(f"Events written:   {events_written}")
    print(f"Total time:       {elapsed:.1f}s")
    print(f"Average FPS:      {fps_actual:.1f}")
    print(f"Output file:      {args.out}")
    print(f"Snapshots:        {snapshot_dir}/")
    print("=" * 50)
    
    # Write summary stats to separate file
    stats = {
        "camera_id": args.camera_id,
        "source": args.source,
        "model": args.model,
        "model_version": args.model_version,
        "frames_processed": frame_idx,
        "alerts_fired": alerts_fired,
        "total_seconds": round(elapsed, 2),
        "average_fps": round(fps_actual, 2),
        "gating_config": {
            "min_hits": args.min_hits,
            "hit_window": args.hit_window,
            "cooldown": args.cooldown,
            "tracking_enabled": args.track,
        }
    }
    
    stats_file = Path(args.out).with_suffix(".stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to:   {stats_file}")


if __name__ == "__main__":
    main()
