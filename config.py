"""
Configuration settings for Face Recognition Photo Organizer
"""

# Face Detection Model
# Options: 'hog' (CPU friendly) or 'cnn' (more accurate, requires GPU)
DETECTION_MODEL = 'hog'

# Face Recognition Model
# Options: 'hog' or 'cnn'
RECOGNITION_MODEL = 'hog'

# Face matching tolerance (lower = more strict, higher = more lenient)
# Recommended range: 0.4 - 0.7
FACE_TOLERANCE = 0.6

# Number of times to upsample image for face detection (higher = more accurate but slower)
UPSAMPLE_NUM_TIMES = 1

# Batch size for processing photos
BATCH_SIZE = 32

# Output folder structure
# Options: 'by_person' (groups by person) or 'by_date' (groups by date)
OUTPUT_STRUCTURE = 'by_person'

# Supported image extensions
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

# Logging level
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'

# Create symbolic links instead of copying files (saves disk space)
USE_SYMLINKS = False

# Minimum face size in pixels (skip faces smaller than this)
MIN_FACE_SIZE = 20

# Maximum number of faces per image before skipping
MAX_FACES_PER_IMAGE = 50