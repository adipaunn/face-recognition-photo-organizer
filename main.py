#!/usr/bin/env python3
"""
Main entry point for Face Recognition Photo Organizer
"""

import argparse
import logging
from pathlib import Path
from face_organizer import PhotoOrganizer
import config

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Organize photos by automatically detecting and identifying faces'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to directory containing photos to organize'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output directory for organized photos'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['hog', 'cnn'],
        default=config.DETECTION_MODEL,
        help='Face detection model to use (default: hog)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train recognizer before organizing'
    )
    
    parser.add_argument(
        '--train-data',
        help='Path to directory with known faces for training'
    )
    
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=config.FACE_TOLERANCE,
        help='Face tolerance for matching (default: 0.6)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Update logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Face Recognition Photo Organizer")
    logger.info("=" * 50)
    
    # Create organizer
    organizer = PhotoOrganizer(
        input_dir=args.input,
        output_dir=args.output,
        model=args.model
    )
    
    # Train if requested
    if args.train and args.train_data:
        logger.info(f"Training recognizer on {args.train_data}")
        organizer.train_recognizer(args.train_data)
    
    # Process photos
    organizer.process_photos()
    
    logger.info("Done! Check the output directory for organized photos.")


if __name__ == '__main__':
    main()