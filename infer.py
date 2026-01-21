import argparse
import json
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO

# Week 1: coarse "animal presence" using COCO-ish classes.
ANIMAL_CLASSES = {
    "dog", "cat", "horse", "sheep", "cow",
    "bear", "zebra", "giraffe", "elephant"
}

def iso_utc_now():
    return datetime.now(timezone.utc).isoformat()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="MP4 path or RTSP URL")
    ap.add_argument("--camera-id", required=True, help="camera identifier in JSON output")
    ap.add_argument("--out", default="events.jsonl", help="output JSONL path")
    ap.add_argument("--model", default="yolo11n.pt", help="e.g. yolo11n.pt or yolo12n.pt")
    ap.add_argument("--sample-every", type=int, default=10, help="process 1 in every N frames")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    model = YOLO(args.model)

    written = 0
    frame_idx = 0

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

            results = model.predict(frame, conf=args.conf, verbose=False)
            r0 = results[0]

            raw_classes = []
            confidences = []

            if r0.boxes is not None and len(r0.boxes) > 0:
                for b in r0.boxes:
                    cls_id = int(b.cls.item())
                    cls_name = r0.names.get(cls_id, str(cls_id))
                    conf = float(b.conf.item())
                    raw_classes.append(cls_name)
                    confidences.append(conf)

            animal_hits = [c for c in raw_classes if c in ANIMAL_CLASSES]
            animal_present = len(animal_hits) > 0

            confidence = 0.0
            if animal_hits:
                confidence = max(
                    conf for (cls, conf) in zip(raw_classes, confidences) if cls in ANIMAL_CLASSES
                )
            elif confidences:
                confidence = max(confidences)

            event = {
                "timestamp": iso_utc_now(),
                "source": args.camera_id,
                "animal_present": animal_present,
                "raw_classes": raw_classes,
                "confidence": round(confidence, 4),
            }

            f.write(json.dumps(event) + "\n")
            f.flush()
            written += 1

            if written % 10 == 0:
                print(f"Wrote {written} events (frame {frame_idx})")

    cap.release()
    print(f"Done. Wrote {written} events to {args.out}")

if __name__ == "__main__":
    main()
