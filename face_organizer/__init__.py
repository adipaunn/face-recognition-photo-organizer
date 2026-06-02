"""
Face Recognition Photo Organizer
Automatically organize photos by identifying and grouping faces
"""

from .organizer import PhotoOrganizer
from .detector import FaceDetector
from .recognizer import FaceRecognizer

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = ['PhotoOrganizer', 'FaceDetector', 'FaceRecognizer']