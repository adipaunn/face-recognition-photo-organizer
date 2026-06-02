"""
Utility functions for face recognition photo organizer
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple
import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def setup_directories(input_dir: str, output_dir: str) -> Tuple[Path, Path]:
    """
    Setup and validate input/output directories
    
    Args:
        input_dir: Path to input photos directory
        output_dir: Path to output organized photos directory
        
    Returns:
        Tuple of (input_path, output_path) as Path objects
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    
    return input_path, output_path


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from a directory
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of image file paths
    """
    image_files = []
    for ext in config.SUPPORTED_FORMATS:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    logger.info(f"Found {len(image_files)} image files")
    return sorted(image_files)


def create_person_folder(output_dir: Path, person_id: int) -> Path:
    """
    Create a folder for a specific person
    
    Args:
        output_dir: Output directory path
        person_id: ID of the person
        
    Returns:
        Path to the created folder
    """
    person_folder = output_dir / f"person_{person_id:03d}"
    person_folder.mkdir(parents=True, exist_ok=True)
    return person_folder


def create_unknown_folder(output_dir: Path) -> Path:
    """
    Create a folder for unknown faces
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Path to the created folder
    """
    unknown_folder = output_dir / "unknown_faces"
    unknown_folder.mkdir(parents=True, exist_ok=True)
    return unknown_folder


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem compatibility
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def format_file_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


class Statistics:
    """Class to track processing statistics"""
    
    def __init__(self):
        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.total_faces = 0
        self.unique_persons = 0
        self.unknown_faces = 0
        
    def print_summary(self):
        """Print statistics summary"""
        logger.info("\n" + "="*50)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total images processed: {self.processed_images}/{self.total_images}")
        logger.info(f"Failed images: {self.failed_images}")
        logger.info(f"Total faces detected: {self.total_faces}")
        logger.info(f"Unique persons identified: {self.unique_persons}")
        logger.info(f"Unknown faces: {self.unknown_faces}")
        logger.info("="*50 + "\n")