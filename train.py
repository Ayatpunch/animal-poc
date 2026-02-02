"""
YOLO Training Script for Wildlife Detection

This script fine-tunes a YOLO model on your labeled wildlife dataset.
Run this on a machine with a GPU (cloud or local).

Usage:
    python train.py --data configs/dataset.yaml --epochs 50
    python train.py --data configs/dataset.yaml --epochs 50 --imgsz 960 --model yolo11s.pt

Requirements:
    - GPU with 8GB+ VRAM (RTX 3070 or better recommended)
    - Labeled dataset in data/train and data/val folders
    - ultralytics package installed
"""

import argparse
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def get_dataset_hash(data_path: Path) -> str:
    """Create a hash of the dataset for versioning."""
    hash_md5 = hashlib.md5()
    
    # Hash the dataset.yaml content
    if data_path.exists():
        hash_md5.update(data_path.read_bytes())
    
    # Count files in train/val for a simple version fingerprint
    train_images = list(Path("data/train/images").glob("*")) if Path("data/train/images").exists() else []
    val_images = list(Path("data/val/images").glob("*")) if Path("data/val/images").exists() else []
    
    hash_md5.update(f"train:{len(train_images)},val:{len(val_images)}".encode())
    
    return hash_md5.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on wildlife dataset")
    
    # Required
    parser.add_argument("--data", type=str, default="configs/dataset.yaml",
                        help="Path to dataset.yaml config file")
    
    # Training parameters
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Pretrained model to start from (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (30-50 recommended)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (640 or 960 for night IR)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 for auto based on GPU memory)")
    
    # Run naming
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (e.g., 'v1.0_baseline'). Auto-generated if not provided.")
    
    # Advanced options
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous run (path to last.pt)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to train on ('0' for GPU 0, 'cpu' for CPU)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Validate dataset exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    
    train_path = Path("data/train/images")
    if not train_path.exists() or not any(train_path.iterdir()):
        raise FileNotFoundError(
            "No training images found in data/train/images/\n"
            "Please add labeled images before training.\n"
            "See README for instructions on using Roboflow to label and export data."
        )
    
    # Generate run name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"run_{timestamp}"
    
    # Create dataset version hash
    dataset_hash = get_dataset_hash(data_path)
    
    print("=" * 60)
    print("YOLO Wildlife Training")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.data}")
    print(f"Dataset Hash: {dataset_hash}")
    print(f"Epochs:       {args.epochs}")
    print(f"Image Size:   {args.imgsz}")
    print(f"Batch Size:   {'auto' if args.batch == -1 else args.batch}")
    print(f"Device:       {args.device}")
    print(f"Run Name:     {args.name}")
    print("=" * 60)
    
    # Load model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading pretrained: {args.model}")
        model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        
        # Save settings
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Augmentation (good defaults for wildlife/IR)
        augment=True,
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation  
        hsv_v=0.4,    # Value/brightness augmentation (helps with IR)
        degrees=10,   # Rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        
        # Performance
        amp=True,     # Mixed precision (faster on modern GPUs)
        cache=True,   # Cache images in RAM (faster if you have enough memory)
        
        # Logging
        verbose=True,
        plots=True,
    )
    
    # After training, save metadata
    run_dir = Path(f"runs/detect/{args.name}")
    
    # Save training metadata
    metadata = {
        "run_name": args.name,
        "timestamp": datetime.now().isoformat(),
        "dataset_config": args.data,
        "dataset_hash": dataset_hash,
        "base_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "results": {
            "final_epoch": args.epochs,
            # Results metrics will be in results.csv
        }
    }
    
    metadata_path = run_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {run_dir}")
    print(f"Best weights:     {run_dir}/weights/best.pt")
    print(f"Last weights:     {run_dir}/weights/last.pt")
    print(f"Metadata:         {metadata_path}")
    print("\nNext steps:")
    print("1. Copy best.pt to use for inference:")
    print(f"   cp {run_dir}/weights/best.pt ./wildlife_v1.pt")
    print("2. Run evaluation on holdouts:")
    print(f"   python evaluate.py --model {run_dir}/weights/best.pt --data data/holdout_a")
    print("=" * 60)


if __name__ == "__main__":
    main()
