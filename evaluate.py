"""
Holdout Evaluation Script for Wildlife Detection Model

Evaluates model performance on holdout datasets with detailed metrics:
- Precision/recall per class
- Confusion matrix (canine vs cervidae focus)
- Day vs night breakdown (based on filename timestamps)
- Nuisance detection analysis

Usage:
    python evaluate.py --model runs/detect/run_20260203_032801/weights/best.pt --holdout a
    python evaluate.py --model best.pt --holdout b --conf 0.5
    python evaluate.py --model best.pt --holdout a --output reports/holdout_a_report.json
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


def parse_filename_metadata(filename: str) -> Dict:
    """
    Extract metadata from filename.
    
    Expected format: ..._YYYY-MM-DD_location_XX_animal_..._HH_MM_SS_...
    Example: DSCF0006_2018-05-04_location_06_coyote_01_21_2026_13_45_47_861_canine_JPG.rf.xxx.jpg
    """
    metadata = {
        "date": None,
        "location": None,
        "time_of_day": None,  # "day" or "night"
        "hour": None,
    }
    
    # Extract date (YYYY-MM-DD format)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        metadata["date"] = date_match.group(1)
    
    # Extract location
    loc_match = re.search(r'location_(\d+)', filename)
    if loc_match:
        metadata["location"] = f"location_{loc_match.group(1)}"
    
    # Extract hour from timestamp in filename (looking for HH_MM_SS pattern after date)
    # The timestamp appears to be embedded as: 01_21_2026_13_45_47 (MM_DD_YYYY_HH_MM_SS)
    time_match = re.search(r'_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_', filename)
    if time_match:
        hour = int(time_match.group(4))
        metadata["hour"] = hour
        # Day: 6am-6pm, Night: 6pm-6am
        metadata["time_of_day"] = "day" if 6 <= hour < 18 else "night"
    
    return metadata


def compute_confusion_matrix(predictions: List[Dict], ground_truth: List[Dict], 
                            class_names: List[str], iou_threshold: float = 0.5) -> np.ndarray:
    """
    Compute confusion matrix from predictions and ground truth.
    
    Returns NxN matrix where:
    - Rows are ground truth classes
    - Columns are predicted classes
    - Extra column for "missed" (false negatives)
    - Extra row for "false positives" (no matching GT)
    """
    n_classes = len(class_names)
    # Matrix: rows=GT, cols=Pred, +1 for background/missed
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    
    # For each image, match predictions to ground truth
    for pred, gt in zip(predictions, ground_truth):
        pred_boxes = pred.get("boxes", [])
        gt_boxes = gt.get("boxes", [])
        
        matched_gt = set()
        
        for p_box in pred_boxes:
            p_cls = p_box["class_id"]
            best_iou = 0
            best_gt_idx = -1
            best_gt_cls = -1
            
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box["bbox"], g_box["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    best_gt_cls = g_box["class_id"]
            
            if best_iou >= iou_threshold:
                # True positive (matched)
                confusion[best_gt_cls, p_cls] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False positive (no matching GT)
                confusion[n_classes, p_cls] += 1
        
        # Count missed GT boxes (false negatives)
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                confusion[g_box["class_id"], n_classes] += 1
    
    return confusion


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def format_confusion_matrix(confusion: np.ndarray, class_names: List[str]) -> str:
    """Format confusion matrix as a readable string."""
    labels = class_names + ["FP/BG"]
    header = ["GT \\ Pred"] + labels
    
    lines = []
    lines.append(" | ".join(f"{h:>10}" for h in header))
    lines.append("-" * len(lines[0]))
    
    row_labels = class_names + ["Missed"]
    for i, row_label in enumerate(row_labels):
        row = [f"{row_label:>10}"] + [f"{confusion[i, j]:>10}" for j in range(len(labels))]
        lines.append(" | ".join(row))
    
    return "\n".join(lines)


def run_evaluation(model_path: str, data_yaml: str, conf: float = 0.25) -> Dict:
    """
    Run YOLO validation and extract detailed metrics.
    """
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        split="test",
        conf=conf,
        iou=0.5,
        verbose=False,
        plots=True,  # Generate confusion matrix plot
    )
    
    # Extract metrics
    metrics = {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }
    
    # Per-class metrics
    class_names = list(results.names.values())
    per_class = {}
    
    for i, cls_name in enumerate(class_names):
        if i < len(results.box.ap50):
            per_class[cls_name] = {
                "ap50": float(results.box.ap50[i]),
                "precision": float(results.box.p[i]) if i < len(results.box.p) else 0,
                "recall": float(results.box.r[i]) if i < len(results.box.r) else 0,
            }
    
    metrics["per_class"] = per_class
    metrics["class_names"] = class_names
    
    # Get confusion matrix if available
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        metrics["confusion_matrix"] = cm.tolist()
    
    return metrics, results


def evaluate_by_time_of_day(model_path: str, holdout_path: str, conf: float = 0.25) -> Dict:
    """
    Run inference on individual images and group results by day/night.
    """
    model = YOLO(model_path)
    
    images_dir = Path(holdout_path) / "images"
    labels_dir = Path(holdout_path) / "labels"
    
    if not images_dir.exists():
        return {"error": f"Images directory not found: {images_dir}"}
    
    day_results = {"tp": 0, "fp": 0, "fn": 0, "images": 0}
    night_results = {"tp": 0, "fp": 0, "fn": 0, "images": 0}
    
    # Get class names from model
    class_names = model.names
    
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        
        # Parse metadata from filename
        metadata = parse_filename_metadata(img_path.name)
        time_of_day = metadata.get("time_of_day", "unknown")
        
        # Run inference
        results = model.predict(str(img_path), conf=conf, verbose=False)
        
        # Count predictions
        n_preds = len(results[0].boxes) if results[0].boxes is not None else 0
        
        # Load ground truth
        label_path = labels_dir / (img_path.stem + ".txt")
        n_gt = 0
        if label_path.exists():
            with open(label_path) as f:
                n_gt = len([l for l in f.readlines() if l.strip()])
        
        # Simple TP/FP/FN counting (approximate)
        tp = min(n_preds, n_gt)
        fp = max(0, n_preds - n_gt)
        fn = max(0, n_gt - n_preds)
        
        # Add to appropriate bucket
        if time_of_day == "day":
            day_results["tp"] += tp
            day_results["fp"] += fp
            day_results["fn"] += fn
            day_results["images"] += 1
        elif time_of_day == "night":
            night_results["tp"] += tp
            night_results["fp"] += fp
            night_results["fn"] += fn
            night_results["images"] += 1
    
    # Calculate precision/recall
    def calc_metrics(r):
        precision = r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) > 0 else 0
        recall = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else 0
        return {
            "images": r["images"],
            "true_positives": r["tp"],
            "false_positives": r["fp"],
            "false_negatives": r["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    
    return {
        "day": calc_metrics(day_results),
        "night": calc_metrics(night_results),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on holdout datasets")
    
    parser.add_argument("--model", required=True, help="Path to model weights (best.pt)")
    parser.add_argument("--holdout", required=True, choices=["a", "b", "both"],
                        help="Which holdout to evaluate: 'a', 'b', or 'both'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Determine which holdouts to evaluate
    holdouts = []
    if args.holdout in ["a", "both"]:
        holdouts.append(("holdout_a", "configs/holdout_a.yaml", "data/holdout_a"))
    if args.holdout in ["b", "both"]:
        holdouts.append(("holdout_b", "configs/holdout_b.yaml", "data/holdout_b"))
    
    all_results = {}
    
    for holdout_name, config_path, data_path in holdouts:
        print(f"\n{'='*60}")
        print(f"Evaluating {holdout_name.upper()}")
        print(f"{'='*60}")
        
        # Check paths exist
        if not Path(config_path).exists():
            print(f"Warning: Config not found: {config_path}")
            continue
        
        if not Path(data_path).exists():
            print(f"Warning: Data path not found: {data_path}")
            continue
        
        # Run main evaluation
        print(f"\nRunning YOLO validation on {holdout_name}...")
        metrics, yolo_results = run_evaluation(args.model, config_path, args.conf)
        
        # Print summary
        print(f"\n--- Overall Metrics ---")
        print(f"mAP50:      {metrics['map50']*100:.1f}%")
        print(f"mAP50-95:   {metrics['map50_95']*100:.1f}%")
        print(f"Precision:  {metrics['precision']*100:.1f}%")
        print(f"Recall:     {metrics['recall']*100:.1f}%")
        
        print(f"\n--- Per-Class Performance ---")
        print(f"{'Class':<12} {'AP50':>8} {'Precision':>10} {'Recall':>8}")
        print("-" * 40)
        for cls_name, cls_metrics in metrics.get("per_class", {}).items():
            print(f"{cls_name:<12} {cls_metrics['ap50']*100:>7.1f}% {cls_metrics['precision']*100:>9.1f}% {cls_metrics['recall']*100:>7.1f}%")
        
        # Confusion matrix
        if "confusion_matrix" in metrics:
            print(f"\n--- Confusion Matrix ---")
            cm = np.array(metrics["confusion_matrix"])
            class_names = metrics.get("class_names", ["bear", "canine", "cervidae"])
            print(format_confusion_matrix(cm, class_names))
            
            # Highlight canine <-> cervidae confusion
            if len(class_names) >= 3:
                canine_idx = class_names.index("canine") if "canine" in class_names else 1
                cervidae_idx = class_names.index("cervidae") if "cervidae" in class_names else 2
                canine_as_cervidae = cm[canine_idx, cervidae_idx] if cm.shape[0] > canine_idx and cm.shape[1] > cervidae_idx else 0
                cervidae_as_canine = cm[cervidae_idx, canine_idx] if cm.shape[0] > cervidae_idx and cm.shape[1] > canine_idx else 0
                print(f"\n⚠️  Canine↔Cervidae Confusion:")
                print(f"   Canine misclassified as Cervidae: {canine_as_cervidae}")
                print(f"   Cervidae misclassified as Canine: {cervidae_as_canine}")
        
        # Day vs Night breakdown
        print(f"\n--- Day vs Night Breakdown ---")
        time_metrics = evaluate_by_time_of_day(args.model, data_path, args.conf)
        
        for period in ["day", "night"]:
            pm = time_metrics.get(period, {})
            if pm.get("images", 0) > 0:
                print(f"\n{period.upper()} ({pm['images']} images):")
                print(f"  Precision: {pm['precision']*100:.1f}%")
                print(f"  Recall:    {pm['recall']*100:.1f}%")
                print(f"  TP: {pm['true_positives']}, FP: {pm['false_positives']}, FN: {pm['false_negatives']}")
            else:
                print(f"\n{period.upper()}: No images classified as {period}")
        
        metrics["time_of_day"] = time_metrics
        all_results[holdout_name] = metrics
    
    # Save results to JSON
    output_path = args.output or f"evaluation_report_{args.holdout}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "confidence_threshold": args.conf,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Report saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
