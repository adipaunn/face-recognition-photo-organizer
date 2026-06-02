# Face Recognition Photo Organizer

A Python application that automatically organizes photos by identifying and grouping faces. Perfect for organizing large photo collections by person.

## Features

- 🔍 **Face Detection & Identification** - Automatically detect and identify faces in photos
- 📁 **Automatic Organization** - Organize photos into folders by person
- 🎯 **High Accuracy** - Uses state-of-the-art face recognition models
- ⚡ **Batch Processing** - Process multiple photos efficiently
- 🏷️ **Custom Labels** - Add custom names to identified faces
- 📊 **Statistics** - View statistics about detected faces

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- GPU support optional (CUDA for faster processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adipaunn/face-recognition-photo-organizer.git
cd face-recognition-photo-organizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from face_organizer import PhotoOrganizer

# Initialize the organizer
organizer = PhotoOrganizer(input_dir='./photos', output_dir='./organized_photos')

# Process all photos
organizer.process_photos()

# View results
organizer.print_statistics()
```

## Usage

### Basic Usage
```bash
python main.py --input ./photos --output ./organized_photos
```

### With Custom Models
```bash
python main.py --input ./photos --output ./organized_photos --model hog
```

### Training Mode (Identify specific people)
```bash
python main.py --train --data_dir ./known_faces
```

## Project Structure

```
face-recognition-photo-organizer/
├── README.md
├── requirements.txt
├── main.py
├── face_organizer/
│   ├── __init__.py
│   ├── detector.py          # Face detection
│   ├── recognizer.py        # Face identification
│   ├── organizer.py         # Photo organization logic
│   └── utils.py             # Utility functions
├── config.py                # Configuration settings
├── known_faces/             # Training data (optional)
├── photos/                  # Input photos
└── organized_photos/        # Output organized photos
```

## Configuration

Edit `config.py` to customize:
- Model type (CNN or HOG)
- Tolerance levels for face matching
- Output folder structure
- Batch size for processing

## Models

The app supports multiple face recognition models:
- **HOG (Histogram of Oriented Gradients)** - Faster, CPU friendly
- **CNN (Convolutional Neural Networks)** - More accurate, GPU recommended

## Performance Tips

- Use GPU for faster processing (CUDA/cuDNN)
- Start with HOG model for testing
- Increase tolerance for looser grouping
- Process in batches for large collections

## License

This project is open source and available under the MIT License - completely free to use!

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.