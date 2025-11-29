#!/usr/bin/env python3
"""
Script to create a video from a series of images in the inf_log folder.
Each subfolder (step_XXXXXX) contains one rgb.png image.
"""

import os
import re
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def natural_sort_key(s):
    """Sort strings containing numbers in a natural way."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def create_video_from_images(input_dir, output_path, fps=30, image_name='rgb.png'):
    """
    Create a video from a series of images in subdirectories.
    
    Args:
        input_dir: Path to the directory containing step_XXXXXX subdirectories
        output_path: Path to the output video file
        fps: Frames per second for the output video
        image_name: Name of the image file in each subdirectory (default: 'rgb.png')
    """
    input_path = Path(input_dir)
    
    # Get all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    # Sort subdirectories naturally (step_000000, step_000001, ...)
    subdirs = sorted(subdirs, key=lambda x: natural_sort_key(x.name))
    
    print(f"Found {len(subdirs)} subdirectories")
    
    if len(subdirs) == 0:
        print("No subdirectories found!")
        return
    
    # Collect all image paths
    image_paths = []
    for subdir in subdirs:
        img_path = subdir / image_name
        if img_path.exists():
            image_paths.append(img_path)
        else:
            print(f"Warning: {img_path} not found, skipping...")
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        print(f"Error reading first image: {image_paths[0]}")
        return
    
    height, width, channels = first_frame.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi or 'mp4v' for .mp4
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer")
        return
    
    print(f"Creating video at {fps} FPS...")
    
    # Write images to video
    for img_path in tqdm(image_paths, desc="Processing frames"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue
        
        # Ensure frame has the same dimensions as the first frame
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
    
    # Release the video writer
    out.release()
    
    print(f"Video successfully created: {output_path}")
    print(f"Total frames: {len(image_paths)}")
    print(f"Duration: {len(image_paths) / fps:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Create a video from a series of images in subdirectories.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='inf_log',
        help='Path to the directory containing step_XXXXXX subdirectories (default: inf_log)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_video.mp4',
        help='Output video file path (default: output_video.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for the output video (default: 30)'
    )
    parser.add_argument(
        '--image-name',
        type=str,
        default='rgb.png',
        help='Name of the image file in each subdirectory (default: rgb.png)'
    )
    
    args = parser.parse_args()
    
    create_video_from_images(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
        image_name=args.image_name
    )


if __name__ == '__main__':
    main()


