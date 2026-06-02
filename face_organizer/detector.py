"""
Face detection module using face_recognition library
"""

import logging
from typing import List, Tuple
import cv2
import face_recognition
import numpy as np
import config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces in images"""
    
    def __init__(self, model: str = 'hog', upsample: int = 1):
        """
        Initialize face detector
        
        Args:
            model: Detection model to use ('hog' or 'cnn')
            upsample: Number of times to upsample image
        """
        self.model = model
        self.upsample = upsample
        logger.info(f"Initialized FaceDetector with model: {model}")
    
    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                image,
                number_of_times_to_upsample=self.upsample,
                model=self.model
            )
            
            # Filter faces by minimum size
            filtered_faces = self._filter_faces_by_size(face_locations, image.shape)
            
            logger.debug(f"Detected {len(filtered_faces)} faces in {image_path}")
            return filtered_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return []
    
    def get_face_encodings(self, image_path: str, face_locations: List = None) -> List[np.ndarray]:
        """
        Get face encodings for detected faces
        
        Args:
            image_path: Path to image file
            face_locations: Optional list of face locations. If None, will detect faces
            
        Returns:
            List of face encodings (128D arrays)
        """
        try:
            image = face_recognition.load_image_file(image_path)
            
            if face_locations is None:
                face_locations = self.detect_faces(image_path)
            
            if not face_locations:
                return []
            
            encodings = face_recognition.face_encodings(image, face_locations)
            logger.debug(f"Generated {len(encodings)} face encodings")
            return encodings
            
        except Exception as e:
            logger.error(f"Error getting face encodings: {str(e)}")
            return []
    
    def _filter_faces_by_size(self, face_locations: List, image_shape: Tuple) -> List:
        """
        Filter faces by minimum size
        
        Args:
            face_locations: List of face locations
            image_shape: Shape of the image
            
        Returns:
            Filtered list of face locations
        """
        filtered = []
        for top, right, bottom, left in face_locations:
            face_height = bottom - top
            face_width = right - left
            
            if face_height >= config.MIN_FACE_SIZE and face_width >= config.MIN_FACE_SIZE:
                filtered.append((top, right, bottom, left))
        
        return filtered
    
    def draw_faces(self, image_path: str, face_locations: List, output_path: str = None):
        """
        Draw rectangles around detected faces (for visualization)
        
        Args:
            image_path: Path to image file
            face_locations: List of face locations
            output_path: Optional path to save annotated image
        """
        try:
            image = cv2.imread(image_path)
            
            for top, right, bottom, left in face_locations:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if output_path:
                cv2.imwrite(output_path, image)
                logger.info(f"Saved annotated image to {output_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error drawing faces: {str(e)}")
            return None