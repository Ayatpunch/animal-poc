# Data Folder Structure

This folder holds your training/evaluation data organized for YOLO training.

## Folder Layout

```
data/
├── train/              # ~70% of your data - model learns from this
│   ├── images/         # Put .jpg/.png files here
│   └── labels/         # YOLO format .txt files (auto-created by Roboflow)
│
├── val/                # ~20% of your data - checked during training
│   ├── images/
│   └── labels/
│
├── test/               # ~10% of your data - final accuracy check
│   ├── images/
│   └── labels/
│
├── holdout_a/          # Viewtron/Mike footage - NEVER train on this
│   ├── images/
│   └── labels/
│
└── holdout_b/          # Different camera/environment - NEVER train on this
    ├── images/
    └── labels/
```

## Important Rules

### 1. Split by Camera + Time, NOT Random

**WRONG:** Randomly shuffle all frames into train/val/test
- This causes "data leakage" - similar frames end up in both train and test
- Your accuracy looks great but fails on new cameras

**RIGHT:** Keep all frames from one camera session together
- Camera A, Monday morning → all goes to train
- Camera A, Monday afternoon → all goes to val  
- Camera B, Tuesday → all goes to test

### 2. Never Train on Holdouts

The `holdout_a` and `holdout_b` folders are for **evaluation only**:
- Put Mike's Viewtron footage in `holdout_a`
- Put footage from a completely different source in `holdout_b`
- These prove your model works on cameras it's never seen

### 3. YOLO Label Format

Each image needs a matching `.txt` file with the same name:
```
image001.jpg  →  image001.txt
image002.jpg  →  image002.txt
```

Label file format (one line per object):
```
class_id center_x center_y width height
```

Example (`image001.txt`):
```
0 0.5 0.5 0.2 0.3
1 0.7 0.6 0.15 0.25
```

**You don't need to create these manually** - Roboflow exports them for you.

## Using Roboflow

1. Go to the Roboflow workspace (use the invite link)
2. Upload your images
3. Draw bounding boxes around animals
4. Label each box with the class (canine, cervidae, bear, etc.)
5. Export in "YOLO v8" format
6. Unzip and copy files to the appropriate folders here

## Class IDs

From `configs/dataset.yaml`:
```
0: canine       # wolves, coyotes, dogs, foxes
1: cervidae     # deer, elk, moose
2: bear
3: feline       # mountain lions, bobcats, cats
4: small_mammal # raccoons, opossums, rabbits
5: bird
6: human
7: vehicle
```

## Minimum Recommended Data

For decent results:
- **50+ images per class** minimum
- **200+ images per class** recommended
- Include variety: day/night, close/far, different angles
- Include "hard negatives": IR glare, empty scenes, vegetation
