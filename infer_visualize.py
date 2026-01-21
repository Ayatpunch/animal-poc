import argparse
import cv2
from ultralytics import YOLO

# Week 1: coarse "animal presence" using COCO-ish classes.
ANIMAL_CLASSES = {
    "dog", "cat", "horse", "sheep", "cow",
    "bear", "zebra", "giraffe", "elephant"
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="MP4 path, RTSP URL, or UDP stream")
    ap.add_argument("--model", default="yolo11n.pt", help="e.g. yolo11n.pt or yolo12n.pt")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--sample-every", type=int, default=1, help="show 1 in every N frames (1=all frames)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    model = YOLO(args.model)
    
    print(f"Playing: {args.source}")
    print("Press 'q' to quit, 'SPACE' to pause/resume")
    
    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video or stream interrupted")
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # Skip frames if sampling
            if frame_idx % args.sample_every != 0:
                continue

            # Run inference
            results = model.predict(frame, conf=args.conf, verbose=False)
            r0 = results[0]

            # Draw detections
            if r0.boxes is not None and len(r0.boxes) > 0:
                for b in r0.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    
                    # Get class info
                    cls_id = int(b.cls.item())
                    cls_name = r0.names.get(cls_id, str(cls_id))
                    conf = float(b.conf.item())
                    
                    # Determine color based on whether it's an animal
                    is_animal = cls_name in ANIMAL_CLASSES
                    color = (0, 255, 0) if is_animal else (255, 0, 0)  # Green for animals, Blue for others
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{cls_name}: {conf:.2f}"
                    if is_animal:
                        label = f"[ANIMAL] {label}"
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)

            # Add frame counter and info
            info_text = f"Frame: {frame_idx} | Press Q to quit, SPACE to pause"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("Animal Detection - Live View", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit requested")
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")

if __name__ == "__main__":
    main()
