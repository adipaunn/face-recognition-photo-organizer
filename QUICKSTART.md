# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/adipaunn/face-recognition-photo-organizer.git
cd face-recognition-photo-organizer
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Basic Usage

### Organize your photos:
```bash
python main.py --input ./photos --output ./organized_photos
```

This will:
- Detect all faces in your photos
- Group similar faces together
- Create folders named `person_001`, `person_002`, etc.
- Copy each photo to the appropriate person folder

### Example with verbose output:
```bash
python main.py --input ./photos --output ./organized_photos --verbose
```

## Advanced Usage

### Using CNN model (more accurate but slower):
```bash
python main.py --input ./photos --output ./organized_photos --model cnn
```

### With custom tolerance (lower = stricter):
```bash
python main.py --input ./photos --output ./organized_photos --tolerance 0.5
```

### Train recognizer on known faces:

First, organize your known faces in a directory structure:
```
known_faces/
├── alice/
│   ├── photo1.jpg
│   └── photo2.jpg
└── bob/
    ├── photo1.jpg
    └── photo2.jpg
```

Then run:
```bash
python main.py --input ./photos --output ./organized_photos --train --train-data ./known_faces
```

## Tips & Tricks

### 1. Memory Issues?
- Use HOG model (default): faster, uses less RAM
- Reduce image resolution before processing
- Process photos in batches

### 2. Not grouping similar faces?
- Lower the `--tolerance` value (default: 0.6)
- Try: `--tolerance 0.5` or even `--tolerance 0.4`

### 3. Want to use GPU?
- Install GPU-accelerated versions:
```bash
pip install torch torchvision  # For CUDA support
```
- Change `DETECTION_MODEL` in `config.py` to `'cnn'`

### 4. Save disk space?
- Use symbolic links instead of copying (set `USE_SYMLINKS = True` in config.py)
- Before: `python main.py ...` copies all images
- With symlinks: only original files are kept, folders contain links

## Understanding the Output

After running the script, you'll have:

```
organized_photos/
├── person_001/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── person_002/
│   ├── photo3.jpg
│   └── ...
└── ...
```

Each `person_XXX` folder contains photos with the same detected face.

## Troubleshooting

**No faces detected?**
- Check image quality
- Try upsampling: increase `UPSAMPLE_NUM_TIMES` in config.py
- Switch to CNN model

**Too slow?**
- Use HOG model instead of CNN
- Reduce image sizes
- Check if CPU is being maxed out

**Wrong groupings?**
- Adjust tolerance in config.py
- Different lighting conditions can affect accuracy

## Next Steps

1. Rename `person_XXX` folders to actual names
2. Review the results and manually correct if needed
3. Delete duplicate or unwanted groupings
4. Back up your original photos!

## Need help?

- Check `config.py` for all available settings
- Enable verbose logging with `--verbose` flag
- Review the modules in `face_organizer/` folder