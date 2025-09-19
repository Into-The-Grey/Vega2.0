"""
Video input handling and processing utilities for Vega 2.0 multi-modal learning.

This module provides comprehensive video processing capabilities including:
- Support for common video formats (MP4, AVI, MOV, WebM)
- Video validation and metadata extraction
- Frame extraction and sampling
- Video thumbnail generation
- Basic video processing (OpenCV-based)
"""

import os
import cv2
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    '.mp4': 'MP4',
    '.avi': 'AVI', 
    '.mov': 'MOV',
    '.webm': 'WebM',
    '.mkv': 'MKV',
    '.flv': 'FLV',
    '.wmv': 'WMV',
    '.m4v': 'M4V'
}

class VideoProcessor:
    """Main video processing class with comprehensive video handling capabilities."""
    
    def __init__(self):
        """Initialize the video processor."""
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def is_supported_video(self, video_path: str) -> bool:
        """
        Check if the video format is supported.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if the video format is supported, False otherwise
        """
        try:
            file_ext = Path(video_path).suffix.lower()
            return file_ext in SUPPORTED_VIDEO_FORMATS
        except Exception as e:
            logger.error(f"Error checking video format: {e}")
            return False
    
    def validate_video(self, video_path: str) -> Dict[str, Any]:
        """
        Validate video file and extract basic information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with validation results and basic video info
        """
        result = {
            'is_valid': False,
            'error': None,
            'format': None,
            'duration': 0,
            'frame_count': 0,
            'fps': 0,
            'resolution': (0, 0),
            'file_size': 0
        }
        
        try:
            if not os.path.exists(video_path):
                result['error'] = "Video file does not exist"
                return result
            
            if not self.is_supported_video(video_path):
                result['error'] = f"Unsupported video format: {Path(video_path).suffix}"
                return result
            
            # Get file size
            result['file_size'] = os.path.getsize(video_path)
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['error'] = "Cannot open video file with OpenCV"
                return result
            
            # Extract basic properties
            result['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result['fps'] = cap.get(cv2.CAP_PROP_FPS)
            result['resolution'] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            if result['fps'] > 0:
                result['duration'] = result['frame_count'] / result['fps']
            
            result['format'] = SUPPORTED_VIDEO_FORMATS.get(Path(video_path).suffix.lower(), 'Unknown')
            result['is_valid'] = True
            
            cap.release()
            
        except Exception as e:
            result['error'] = f"Error validating video: {str(e)}"
            logger.error(f"Video validation error: {e}")
        
        return result
    
    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with detailed video metadata
        """
        metadata = {
            'basic_info': {},
            'technical_details': {},
            'timestamps': {},
            'error': None
        }
        
        try:
            # Basic validation first
            validation = self.validate_video(video_path)
            if not validation['is_valid']:
                metadata['error'] = validation['error']
                return metadata
            
            metadata['basic_info'] = validation
            
            # Additional technical details using OpenCV
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                metadata['technical_details'].update({
                    'codec': cap.get(cv2.CAP_PROP_FOURCC),
                    'backend': cap.getBackendName()
                })
                cap.release()
            
            # File system metadata
            stat = os.stat(video_path)
            metadata['timestamps'] = {
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'accessed': stat.st_atime
            }
            
        except Exception as e:
            metadata['error'] = f"Error extracting metadata: {str(e)}"
            logger.error(f"Metadata extraction error: {e}")
        
        return metadata
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None, 
                      frame_interval: int = 30, max_frames: int = 100) -> List[str]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames (optional)
            frame_interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of paths to extracted frame images
        """
        frame_paths = []
        
        try:
            if not self.validate_video(video_path)['is_valid']:
                return frame_paths
            
            if output_dir is None:
                output_dir = os.path.join(self.temp_dir, 'frames')
            
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Convert BGR to RGB for saving
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image.save(frame_path, quality=95)
                    
                    frame_paths.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frame_paths
    
    def sample_frames(self, video_path: str, num_samples: int = 10, 
                     method: str = 'uniform') -> List[np.ndarray]:
        """
        Sample frames from video using different strategies.
        
        Args:
            video_path: Path to the video file
            num_samples: Number of frames to sample
            method: Sampling method ('uniform', 'random', 'keyframes')
            
        Returns:
            List of frame arrays (as numpy arrays)
        """
        frames = []
        
        try:
            validation = self.validate_video(video_path)
            if not validation['is_valid']:
                return frames
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if method == 'uniform':
                # Sample frames uniformly across the video
                frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            elif method == 'random':
                # Sample random frames
                frame_indices = np.random.choice(total_frames, num_samples, replace=False)
                frame_indices.sort()
            else:  # keyframes (simplified - just use uniform for now)
                frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
        
        return frames
    
    def generate_thumbnail(self, video_path: str, output_path: Optional[str] = None, 
                          timestamp: Optional[float] = None, size: Tuple[int, int] = (320, 240)) -> str:
        """
        Generate a thumbnail image from the video.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the thumbnail (optional)
            timestamp: Time in seconds to capture thumbnail (optional, defaults to middle)
            size: Thumbnail size as (width, height)
            
        Returns:
            Path to the generated thumbnail
        """
        try:
            validation = self.validate_video(video_path)
            if not validation['is_valid']:
                return ""
            
            if output_path is None:
                video_name = Path(video_path).stem
                output_path = os.path.join(self.temp_dir, f"{video_name}_thumb.jpg")
            
            cap = cv2.VideoCapture(video_path)
            
            # Set timestamp (default to middle of video)
            if timestamp is None:
                timestamp = validation['duration'] / 2
            
            frame_number = int(timestamp * validation['fps'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to thumbnail size
                image = Image.fromarray(frame_rgb)
                image.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                image.save(output_path, quality=85)
                logger.info(f"Generated thumbnail: {output_path}")
            
            cap.release()
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return ""
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get comprehensive video information for analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with comprehensive video information
        """
        try:
            validation = self.validate_video(video_path)
            metadata = self.extract_metadata(video_path)
            
            return {
                'validation': validation,
                'metadata': metadata,
                'analysis_ready': validation['is_valid'] and not metadata['error']
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {'validation': {'is_valid': False}, 'metadata': {'error': str(e)}, 'analysis_ready': False}

class VideoFormatConverter:
    """Utility class for basic video format operations using OpenCV."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def __del__(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def copy_video_frames(self, input_path: str, output_path: Optional[str] = None, 
                         codec: str = 'mp4v') -> str:
        """
        Copy video frames to a new file (basic conversion).
        
        Args:
            input_path: Path to input video
            output_path: Output file path (optional)
            codec: Video codec to use
            
        Returns:
            Path to output video
        """
        try:
            processor = VideoProcessor()
            validation = processor.validate_video(input_path)
            if not validation['is_valid']:
                return ""
            
            if output_path is None:
                input_name = Path(input_path).stem
                output_path = os.path.join(self.temp_dir, f"{input_name}_converted.mp4")
            
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            
            logger.info(f"Video copied to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error copying video: {e}")
            return ""

# Convenience functions for easy access
def validate_video_file(video_path: str) -> Dict[str, Any]:
    """Validate a video file and return basic information."""
    processor = VideoProcessor()
    return processor.validate_video(video_path)

def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from a video file."""
    processor = VideoProcessor()
    return processor.extract_metadata(video_path)

def extract_video_frames(video_path: str, output_dir: Optional[str] = None, 
                        frame_interval: int = 30, max_frames: int = 100) -> List[str]:
    """Extract frames from a video file."""
    processor = VideoProcessor()
    return processor.extract_frames(video_path, output_dir, frame_interval, max_frames)

def sample_video_frames(video_path: str, num_samples: int = 10, 
                       method: str = 'uniform') -> List[np.ndarray]:
    """Sample frames from a video using different strategies."""
    processor = VideoProcessor()
    return processor.sample_frames(video_path, num_samples, method)

def generate_video_thumbnail(video_path: str, output_path: Optional[str] = None, 
                           timestamp: Optional[float] = None, size: Tuple[int, int] = (320, 240)) -> str:
    """Generate a thumbnail from a video."""
    processor = VideoProcessor()
    return processor.generate_thumbnail(video_path, output_path, timestamp, size)

def get_video_information(video_path: str) -> Dict[str, Any]:
    """Get comprehensive video information."""
    processor = VideoProcessor()
    return processor.get_video_info(video_path)

if __name__ == "__main__":
    # Example usage
    processor = VideoProcessor()
    
    # Example video validation
    test_video = "sample_video.mp4"
    if os.path.exists(test_video):
        validation = processor.validate_video(test_video)
        print(f"Video validation: {validation}")
        
        if validation['is_valid']:
            # Extract metadata
            metadata = processor.extract_metadata(test_video)
            print(f"Metadata: {json.dumps(metadata, indent=2)}")
            
            # Generate thumbnail
            thumbnail = processor.generate_thumbnail(test_video)
            print(f"Thumbnail generated: {thumbnail}")
            
            # Extract some frames
            frames = processor.extract_frames(test_video, max_frames=5)
            print(f"Extracted {len(frames)} frames")