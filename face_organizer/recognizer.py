"""
Face recognition module for identifying and matching faces
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import face_recognition
import config

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Recognizes and identifies faces"""
    
    def __init__(self, tolerance: float = None):
        """
        Initialize face recognizer
        
        Args:
            tolerance: Face tolerance for matching (0.0-1.0)
        """
        self.tolerance = tolerance or config.FACE_TOLERANCE
        self.known_encodings: List[np.ndarray] = []
        self.known_labels: List[str] = []
        logger.info(f"Initialized FaceRecognizer with tolerance: {self.tolerance}")
    
    def train_on_faces(self, labeled_faces: Dict[str, List[np.ndarray]]):
        """
        Train recognizer on known faces
        
        Args:
            labeled_faces: Dictionary mapping labels to lists of face encodings
        """
        self.known_encodings = []
        self.known_labels = []
        
        for label, encodings in labeled_faces.items():
            for encoding in encodings:
                self.known_encodings.append(encoding)
                self.known_labels.append(label)
        
        logger.info(f"Trained on {len(self.known_encodings)} faces for {len(set(self.known_labels))} unique persons")
    
    def recognize_faces(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Recognize faces and return identity with confidence
        
        Args:
            face_encodings: List of face encodings to recognize
            
        Returns:
            List of (identity, distance) tuples where distance is 0.0-1.0
        """
        results = []
        
        for encoding in face_encodings:
            # Compare with known faces
            distances = face_recognition.face_distance(
                self.known_encodings, encoding
            )
            
            if len(distances) == 0:
                results.append(("unknown", 1.0))
                continue
            
            # Find closest match
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            if min_distance <= self.tolerance:
                identity = self.known_labels[min_distance_idx]
                results.append((identity, min_distance))
            else:
                results.append(("unknown", min_distance))
        
        return results
    
    def cluster_faces(self, face_encodings: List[np.ndarray]) -> Dict[int, List[int]]:
        """
        Cluster similar faces together
        
        Args:
            face_encodings: List of face encodings
            
        Returns:
            Dictionary mapping cluster IDs to lists of face indices
        """
        if not face_encodings:
            return {}
        
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for i, encoding in enumerate(face_encodings):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            # Find similar faces
            for j in range(i + 1, len(face_encodings)):
                if j not in assigned:
                    distance = face_recognition.face_distance(
                        [encoding], face_encodings[j]
                    )[0]
                    
                    if distance <= self.tolerance:
                        cluster.append(j)
                        assigned.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        logger.info(f"Clustered {len(face_encodings)} faces into {len(clusters)} groups")
        return clusters
    
    def add_known_face(self, label: str, encoding: np.ndarray):
        """
        Add a known face for future recognition
        
        Args:
            label: Label/name for the face
            encoding: Face encoding
        """
        self.known_encodings.append(encoding)
        self.known_labels.append(label)
        logger.debug(f"Added known face for: {label}")
    
    def get_confidence_score(self, distance: float) -> float:
        """
        Convert distance to confidence score (0.0-1.0)
        
        Args:
            distance: Face distance from face_recognition
            
        Returns:
            Confidence score where 1.0 is perfect match
        """
        # Convert distance to confidence (inverse relationship)
        return max(0.0, 1.0 - distance)
    
    def reset(self):
        """Reset the recognizer"""
        self.known_encodings = []
        self.known_labels = []
        logger.info("Face recognizer reset")