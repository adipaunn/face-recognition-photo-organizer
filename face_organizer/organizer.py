"""
Main photo organizer module that coordinates detection, recognition, and organization
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .utils import (
    setup_directories, get_image_files, create_person_folder,
    create_unknown_folder, sanitize_filename, Statistics
)
import config

logger = logging.getLogger(__name__)


class PhotoOrganizer:
    """Main class for organizing photos by detected faces"""
    
    def __init__(self, input_dir: str, output_dir: str, model: str = None):
        """
        Initialize photo organizer
        
        Args:
            input_dir: Path to directory containing photos to organize
            output_dir: Path to directory where organized photos will be saved
            model: Detection model to use ('hog' or 'cnn'). If None, uses config.DETECTION_MODEL
        """
        self.input_dir, self.output_dir = setup_directories(input_dir, output_dir)
        self.model = model or config.DETECTION_MODEL
        
        self.detector = FaceDetector(model=self.model, upsample=config.UPSAMPLE_NUM_TIMES)
        self.recognizer = FaceRecognizer(tolerance=config.FACE_TOLERANCE)
        self.stats = Statistics()
        
        logger.info(f"Initialized PhotoOrganizer with model: {self.model}")
    
    def process_photos(self, recursive: bool = False) -> Statistics:
        """
        Process all photos and organize them by detected faces
        
        Args:
            recursive: If True, search subdirectories recursively
            
        Returns:
            Statistics object with processing results
        """
        logger.info("Starting photo organization...")
        
        # Get all image files
        image_files = get_image_files(self.input_dir)
        self.stats.total_images = len(image_files)
        
        if not image_files:
            logger.warning("No image files found!")
            return self.stats
        
        # First pass: detect and extract faces
        all_encodings = []
        face_to_image = {}  # Maps face index to (image_path, face_location)
        
        for image_path in tqdm(image_files, desc="Detecting faces"):
            try:
                face_locations = self.detector.detect_faces(str(image_path))
                
                if face_locations:
                    encodings = self.detector.get_face_encodings(str(image_path), face_locations)
                    
                    for i, (encoding, location) in enumerate(zip(encodings, face_locations)):
                        face_idx = len(all_encodings)
                        all_encodings.append(encoding)
                        face_to_image[face_idx] = (image_path, location)
                    
                    self.stats.total_faces += len(face_locations)
                    self.stats.processed_images += 1
                else:
                    self.stats.processed_images += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                self.stats.failed_images += 1
        
        if not all_encodings:
            logger.warning("No faces detected in any images!")
            self.stats.print_summary()
            return self.stats
        
        # Second pass: cluster faces
        logger.info("Clustering faces...")
        clusters = self.recognizer.cluster_faces(all_encodings)
        
        # Third pass: organize photos
        logger.info("Organizing photos...")
        self.stats.unique_persons = len(clusters)
        
        for person_id, face_indices in clusters.items():
            person_folder = create_person_folder(self.output_dir, person_id)
            
            # Track which images have been copied to avoid duplicates
            copied_images = set()
            
            for face_idx in face_indices:
                image_path, _ = face_to_image[face_idx]
                
                if image_path not in copied_images:
                    self._copy_photo(image_path, person_folder)
                    copied_images.add(image_path)
        
        self.stats.print_summary()
        return self.stats
    
    def _copy_photo(self, source_path: Path, dest_folder: Path):
        """
        Copy or symlink a photo to destination folder
        
        Args:
            source_path: Source image path
            dest_folder: Destination folder
        """
        try:
            dest_path = dest_folder / sanitize_filename(source_path.name)
            
            if config.USE_SYMLINKS:
                if dest_path.exists():
                    dest_path.unlink()
                dest_path.symlink_to(source_path.resolve())
            else:
                shutil.copy2(source_path, dest_path)
            
            logger.debug(f"Copied/linked {source_path.name} to {dest_folder.name}")
            
        except Exception as e:
            logger.error(f"Error copying {source_path}: {str(e)}")
    
    def train_recognizer(self, known_faces_dir: str):
        """
        Train the recognizer on a directory of known faces
        
        Args:
            known_faces_dir: Directory with subdirectories named after people,
                           containing images of those people
        """
        known_faces_path = Path(known_faces_dir)
        
        if not known_faces_path.exists():
            logger.error(f"Known faces directory not found: {known_faces_dir}")
            return
        
        labeled_faces = {}
        
        for person_dir in known_faces_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            encodings = []
            
            for image_file in get_image_files(person_dir):
                try:
                    face_encodings = self.detector.get_face_encodings(str(image_file))
                    if face_encodings:
                        encodings.extend(face_encodings)
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {str(e)}")
            
            if encodings:
                labeled_faces[person_name] = encodings
                logger.info(f"Loaded {len(encodings)} faces for {person_name}")
        
        if labeled_faces:
            self.recognizer.train_on_faces(labeled_faces)
        else:
            logger.warning("No faces found in known faces directory")
    
    def print_statistics(self):
        """Print processing statistics"""
        self.stats.print_summary()