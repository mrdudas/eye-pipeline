#!/usr/bin/env python3
"""
Standalone Pipeline Processor - GUI-independent video/frame processing

This script processes eye tracking videos with configurable pipeline settings.
Can be used for:
- Single frame processing
- Test video generation (N frames)
- Full video processing
- Batch processing multiple videos

Usage:
    # Process 50 frames starting from frame 0
    ./pipeline_processor.py --config pipeline_settings.yaml --video eye1.mp4 --frames 50 --output test.mp4
    
    # Process full video
    ./pipeline_processor.py --config pipeline_settings.yaml --video eye1.mp4 --output processed.mp4
    
    # Generate single annotated frame
    ./pipeline_processor.py --config pipeline_settings.yaml --video eye1.mp4 --frame-num 10 --output frame.png
"""

import argparse
import time
import yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import csv

# Import our detection modules
from ellseg_integration import EllSegDetector
from iris_unwrapping import IrisUnwrapper


class PipelineProcessor:
    """Standalone pipeline processor without GUI dependencies"""
    
    def __init__(self, config_path: str, video_path: str):
        """
        Initialize pipeline processor
        
        Args:
            config_path: Path to YAML config file
            video_path: Path to input video
        """
        self.config_path = Path(config_path)
        self.video_path = Path(video_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Open video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {video_path}")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   Frames: {self.frame_count}")
        print(f"   FPS: {self.fps:.2f}")
        
        # Initialize detection models
        self._init_models()
        
        # Load camera calibration if available
        self._load_camera_calibration()
    
    def _load_config(self) -> Dict:
        """Load pipeline configuration from YAML"""
        if not self.config_path.exists():
            print(f"⚠️  Config not found: {self.config_path}")
            print("Using default settings")
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded: {self.config_path}")
        return config
    
    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'undistort': {'enabled': False},
            'glint': {'enabled': False},
            'noise': {'enabled': False, 'method': 'bilateral', 'strength': 5},
            'clahe': {'enabled': False, 'clip_limit': 2.0, 'tile_size': 8},
            'ellseg': {'enabled': True, 'show_segmentation': True}
        }
    
    def _init_models(self):
        """Initialize detection models"""
        print("\n🔧 Initializing models...")
        
        # Detect compute device
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"   🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("   🚀 Using Apple Metal Performance Shaders (MPS)")
        else:
            device = 'cpu'
            print("   ⚠️  Using CPU (no GPU detected)")
        
        # Initialize EllSeg
        try:
            self.ellseg_detector = EllSegDetector(device=device)
            self.ellseg_available = self.ellseg_detector.model is not None
            if self.ellseg_available:
                print("   ✅ EllSeg loaded successfully")
            else:
                print("   ⚠️  EllSeg model not available")
        except Exception as e:
            print(f"   ⚠️  EllSeg load failed: {e}")
            self.ellseg_detector = None
            self.ellseg_available = False
        
        # Initialize iris unwrapper
        self.iris_unwrapper = IrisUnwrapper()
        print("   ✅ Iris unwrapper initialized")
    
    def _load_camera_calibration(self):
        """Load camera calibration if available"""
        calib_path = Path("camera_calibration.yaml")
        
        if not calib_path.exists():
            print("\n⚠️  No camera calibration found")
            self.camera_matrix = None
            self.dist_coeffs = None
            self.calibration_loaded = False
            return
        
        try:
            with open(calib_path, 'r') as f:
                calib_data = yaml.safe_load(f)
            
            # Load camera matrix
            cm_data = calib_data['camera_matrix']
            if isinstance(cm_data, dict) and 'data' in cm_data:
                # Format: {rows: 3, cols: 3, data: [...]}
                rows = cm_data['rows']
                cols = cm_data['cols']
                self.camera_matrix = np.array(cm_data['data']).reshape(rows, cols)
            else:
                self.camera_matrix = np.array(cm_data)
            
            # Load distortion coefficients
            dc_data = calib_data['distortion_coefficients']
            if isinstance(dc_data, dict) and 'data' in dc_data:
                # Format: {rows: 1, cols: 5, data: [...]}
                self.dist_coeffs = np.array(dc_data['data'])
            else:
                self.dist_coeffs = np.array(dc_data)
            
            self.calibration_loaded = True
            
            print(f"\n✅ Camera calibration loaded")
            print(f"   fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
        except Exception as e:
            print(f"\n⚠️  Failed to load calibration: {e}")
            self.camera_matrix = None
            self.dist_coeffs = None
            self.calibration_loaded = False
    
    def print_settings(self):
        """Print current pipeline settings"""
        print(f"\n{'='*60}")
        print("📋 PIPELINE SETTINGS")
        print(f"{'='*60}")
        
        # Support both config formats: undistort.enabled and camera.undistort_enabled
        undistort_enabled = (self.config.get('undistort', {}).get('enabled', False) or
                            self.config.get('camera', {}).get('undistort_enabled', False))
        print(f"   Undistort: {undistort_enabled}")
        
        print(f"   Glint removal: {self.config.get('glint', {}).get('enabled', False)}")
        
        noise_cfg = self.config.get('noise', {})
        print(f"   Noise reduction: {noise_cfg.get('enabled', False)} "
              f"(method: {noise_cfg.get('method', 'bilateral')}, "
              f"strength: {noise_cfg.get('strength', 5)})")
        
        clahe_cfg = self.config.get('clahe', {})
        print(f"   CLAHE: {clahe_cfg.get('enabled', False)} "
              f"(clip: {clahe_cfg.get('clip_limit', 2.0)}, "
              f"tile: {clahe_cfg.get('tile_size', 8)})")
        
        print(f"   EllSeg: {self.config.get('ellseg', {}).get('enabled', True)}")
        print(f"{'='*60}\n")
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess frame according to config
        
        Args:
            frame: Input BGR frame
            
        Returns:
            (processed_frame, timing_dict)
        """
        timing = {}
        processed = frame.copy()
        
        # Frame copy timing
        t0 = time.perf_counter()
        processed = frame.copy()
        timing['frame_copy'] = time.perf_counter() - t0
        
        # 0. Camera undistortion (support both config formats)
        undistort_enabled = (self.config.get('undistort', {}).get('enabled', False) or
                            self.config.get('camera', {}).get('undistort_enabled', False))
        
        if undistort_enabled and self.calibration_loaded:
            t0 = time.perf_counter()
            processed = cv2.undistort(processed, self.camera_matrix, self.dist_coeffs)
            timing['undistort'] = time.perf_counter() - t0
        else:
            timing['undistort'] = 0.0
        
        # 1. Glint removal
        if self.config.get('glint', {}).get('enabled', False):
            t0 = time.perf_counter()
            processed = self._remove_glints(processed)
            timing['glint_removal'] = time.perf_counter() - t0
        else:
            timing['glint_removal'] = 0.0
        
        # 2. Noise reduction
        if self.config.get('noise', {}).get('enabled', False):
            t0 = time.perf_counter()
            processed = self._reduce_noise(processed)
            timing['noise_reduction'] = time.perf_counter() - t0
        else:
            timing['noise_reduction'] = 0.0
        
        # 3. CLAHE
        if self.config.get('clahe', {}).get('enabled', False):
            t0 = time.perf_counter()
            processed = self._apply_clahe(processed)
            timing['clahe'] = time.perf_counter() - t0
        else:
            timing['clahe'] = 0.0
        
        return processed, timing
    
    def _remove_glints(self, frame: np.ndarray) -> np.ndarray:
        """Remove glints from frame"""
        cfg = self.config.get('glint', {})
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        threshold = cfg.get('threshold', 240)
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        glint_mask = np.zeros_like(bright_mask)
        min_area = cfg.get('min_area', 5)
        max_area = cfg.get('max_area', 200)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                cv2.drawContours(glint_mask, [contour], -1, 255, -1)
        
        iterations = cfg.get('iterations', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glint_mask = cv2.dilate(glint_mask, kernel, iterations=iterations)
        
        result = cv2.inpaint(frame, glint_mask, 3, cv2.INPAINT_TELEA)
        return result
    
    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        cfg = self.config.get('noise', {})
        method = cfg.get('method', 'bilateral')
        strength = cfg.get('strength', 5)
        
        if method == 'bilateral':
            return cv2.bilateralFilter(frame, strength*2+1, strength*10, strength*10)
        elif method == 'gaussian':
            ksize = strength*2+1
            return cv2.GaussianBlur(frame, (ksize, ksize), 0)
        elif method == 'median':
            ksize = strength*2+1
            return cv2.medianBlur(frame, ksize)
        
        return frame
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE"""
        cfg = self.config.get('clahe', {})
        clip_limit = cfg.get('clip_limit', 2.0)
        tile_size = cfg.get('tile_size', 8)
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return result
    
    def detect_ellseg(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict], np.ndarray]:
        """
        Detect pupil and iris using EllSeg
        
        Args:
            frame: Preprocessed BGR frame
            
        Returns:
            (annotated_frame, pupil_data, iris_data, gray_frame)
        """
        if not self.config.get('ellseg', {}).get('enabled', True):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            return frame, None, None, gray
        
        if not self.ellseg_available:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            return frame, None, None, gray
        
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        
        try:
            results = self.ellseg_detector.detect(gray)
            
            pupil_ellipse = results['pupil_ellipse']
            iris_ellipse = results['iris_ellipse']
            seg_map = results['seg_map']
            confidence = results['confidence']
            
            # Show segmentation overlay
            if self.config.get('ellseg', {}).get('show_segmentation', True):
                overlay = annotated.copy()
                overlay[seg_map == 1] = [0, 255, 0]    # Iris = green
                overlay[seg_map == 2] = [255, 255, 0]  # Pupil = yellow
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
            
            # Draw pupil ellipse
            if not np.all(pupil_ellipse == -1):
                cx, cy, a, b, angle = pupil_ellipse
                cv2.ellipse(annotated, 
                           (int(cx), int(cy)), 
                           (int(a), int(b)), 
                           np.rad2deg(angle), 
                           0, 360, (0, 0, 255), 2)
                cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            
            # Draw iris ellipse
            if not np.all(iris_ellipse == -1):
                cx, cy, a, b, angle = iris_ellipse
                cv2.ellipse(annotated, 
                           (int(cx), int(cy)), 
                           (int(a), int(b)), 
                           np.rad2deg(angle), 
                           0, 360, (0, 255, 0), 2)
            
            # Display confidence
            cv2.putText(annotated, f"Conf: {confidence:.2f}", 
                       (10, self.height - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Format data
            pupil_data = None
            iris_data = None
            
            if not np.all(pupil_ellipse == -1):
                cx, cy, a, b, angle = pupil_ellipse
                pupil_data = {
                    'center': (int(cx), int(cy)),
                    'axes': (2*a, 2*b),
                    'angle': np.rad2deg(angle),
                    'area': np.pi * a * b,
                    'ellipse': pupil_ellipse  # Store raw ellipse for iris unwrapping
                }
            
            if not np.all(iris_ellipse == -1):
                cx, cy, a, b, angle = iris_ellipse
                iris_data = {
                    'center': (int(cx), int(cy)),
                    'axes': (2*a, 2*b),
                    'angle': np.rad2deg(angle),
                    'area': np.pi * a * b,
                    'ellipse': iris_ellipse  # Store raw ellipse for iris unwrapping
                }
            
            return annotated, pupil_data, iris_data, gray
            
        except Exception as e:
            print(f"⚠️  EllSeg detection failed: {e}")
            return frame, None, None, gray
    
    def process_frames(self, start_frame: int, num_frames: int, output_path: Path):
        """
        Process N frames and generate video
        
        Args:
            start_frame: Starting frame number
            num_frames: Number of frames to process
            output_path: Output video path
        """
        overall_start = time.perf_counter()
        
        print(f"\n{'='*60}")
        print(f"🎬 PROCESSING {num_frames} FRAMES")
        print(f"{'='*60}")
        self.print_settings()
        
        # Timing accumulators
        timing_stats = {
            'read': 0.0,
            'preprocess': 0.0,
            'preprocess_undistort': 0.0,
            'preprocess_glint': 0.0,
            'preprocess_noise': 0.0,
            'preprocess_clahe': 0.0,
            'detect': 0.0,
            'annotate': 0.0,
            'write': 0.0
        }
        
        # Setup output video - three views side-by-side: original + detection + frontal
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frontal_width = int(400 * self.height / 400)  # Scale to match height
        combined_width = self.width * 2 + frontal_width
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (combined_width, self.height))
        
        # Create CSV file path (same name as video, but .csv extension)
        csv_path = output_path.with_suffix('.csv')
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        
        # Write CSV header
        csv_writer.writerow([
            'frame_id',
            'pupil_x', 'pupil_y', 'pupil_diameter_px',
            'iris_x', 'iris_y',
            'pupil_ellipse_cx', 'pupil_ellipse_cy', 'pupil_ellipse_a', 'pupil_ellipse_b', 'pupil_ellipse_angle',
            'iris_ellipse_cx', 'iris_ellipse_cy', 'iris_ellipse_a', 'iris_ellipse_b', 'iris_ellipse_angle',
            'detected'
        ])
        
        print(f"📊 CSV output: {csv_path}")
        
        # Seek to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        detected_count = 0
        
        try:
            for idx in range(num_frames):
                frame_num = start_frame + idx
                
                # Read frame
                t0 = time.perf_counter()
                ret, frame = self.cap.read()
                timing_stats['read'] += time.perf_counter() - t0
                
                if not ret:
                    print(f"⚠️  Failed to read frame {frame_num}")
                    break
                
                # Preprocess
                t0 = time.perf_counter()
                preprocessed, preproc_timing = self.preprocess_frame(frame)
                timing_stats['preprocess'] += time.perf_counter() - t0
                timing_stats['preprocess_undistort'] += preproc_timing.get('undistort', 0)
                timing_stats['preprocess_glint'] += preproc_timing.get('glint_removal', 0)
                timing_stats['preprocess_noise'] += preproc_timing.get('noise_reduction', 0)
                timing_stats['preprocess_clahe'] += preproc_timing.get('clahe', 0)
                
                # Detect
                t0 = time.perf_counter()
                result_frame, pupil_data, iris_data, gray = self.detect_ellseg(preprocessed)
                timing_stats['detect'] += time.perf_counter() - t0
                
                # Initialize CSV row data
                csv_row = {
                    'frame_id': frame_num,
                    'pupil_x': '', 'pupil_y': '', 'pupil_diameter_px': '',
                    'iris_x': '', 'iris_y': '',
                    'pupil_ellipse_cx': '', 'pupil_ellipse_cy': '', 
                    'pupil_ellipse_a': '', 'pupil_ellipse_b': '', 'pupil_ellipse_angle': '',
                    'iris_ellipse_cx': '', 'iris_ellipse_cy': '', 
                    'iris_ellipse_a': '', 'iris_ellipse_b': '', 'iris_ellipse_angle': '',
                    'detected': 0
                }
                
                if pupil_data is not None:
                    detected_count += 1
                    csv_row['detected'] = 1
                    
                    # Store pupil center
                    if 'center' in pupil_data:
                        csv_row['pupil_x'] = f"{pupil_data['center'][0]:.2f}"
                        csv_row['pupil_y'] = f"{pupil_data['center'][1]:.2f}"
                    
                    # Store pupil ellipse
                    if 'ellipse' in pupil_data:
                        ellipse = pupil_data['ellipse']  # [cx, cy, a, b, angle]
                        csv_row['pupil_ellipse_cx'] = f"{ellipse[0]:.2f}"
                        csv_row['pupil_ellipse_cy'] = f"{ellipse[1]:.2f}"
                        csv_row['pupil_ellipse_a'] = f"{ellipse[2]:.2f}"
                        csv_row['pupil_ellipse_b'] = f"{ellipse[3]:.2f}"
                        csv_row['pupil_ellipse_angle'] = f"{ellipse[4]:.4f}"
                
                # Store iris data
                if iris_data is not None:
                    if 'center' in iris_data:
                        csv_row['iris_x'] = f"{iris_data['center'][0]:.2f}"
                        csv_row['iris_y'] = f"{iris_data['center'][1]:.2f}"
                    
                    if 'ellipse' in iris_data:
                        ellipse = iris_data['ellipse']  # [cx, cy, a, b, angle]
                        csv_row['iris_ellipse_cx'] = f"{ellipse[0]:.2f}"
                        csv_row['iris_ellipse_cy'] = f"{ellipse[1]:.2f}"
                        csv_row['iris_ellipse_a'] = f"{ellipse[2]:.2f}"
                        csv_row['iris_ellipse_b'] = f"{ellipse[3]:.2f}"
                        csv_row['iris_ellipse_angle'] = f"{ellipse[4]:.4f}"
                
                # Generate frontal view if iris detected
                unwrapped_frontal = None
                pupil_measurements = None
                iris_ellipse_frontal = None
                pupil_ellipse_frontal = None
                if iris_data is not None and 'ellipse' in iris_data:
                    try:
                        # Get iris and pupil ellipses (already in EllSeg format: [cx, cy, a, b, angle_rad])
                        iris_ellipse = iris_data['ellipse']
                        pupil_ellipse = pupil_data['ellipse'] if pupil_data and 'ellipse' in pupil_data else iris_ellipse
                        
                        # Use the same gray image that was used for detection (matches GUI behavior)
                        results = self.iris_unwrapper.process_iris(gray, iris_ellipse, pupil_ellipse)
                        unwrapped_frontal = results['frontal_view']
                        
                        # Extract transformed ellipses for drawing on frontal view
                        iris_ellipse_frontal = results.get('iris_ellipse_frontal')
                        pupil_ellipse_frontal = results.get('pupil_ellipse_frontal')
                        
                        # Extract pupil measurements from frontal view
                        if results.get('pupil_from_frontal'):
                            pupil_measurements = results['pupil_from_frontal']
                            
                            # Store pupil diameter from frontal view (perspective-corrected)
                            if 'pupil_diameter_from_area_px' in pupil_measurements:
                                csv_row['pupil_diameter_px'] = f"{pupil_measurements['pupil_diameter_from_area_px']:.2f}"
                        
                    except Exception as e:
                        print(f"⚠️  Iris unwrapping failed for frame {frame_num}: {e}")
                        import traceback
                        traceback.print_exc()
                        unwrapped_frontal = None
                        pupil_measurements = None
                        iris_ellipse_frontal = None
                        pupil_ellipse_frontal = None
                
                # Annotate original frame
                t0 = time.perf_counter()
                info_frame = frame.copy()
                cv2.putText(info_frame, "Original", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(info_frame, f"Frame: {frame_num}", (10, self.height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Annotate detection result
                status_text = "DETECTED" if pupil_data else "NOT DETECTED"
                status_color = (0, 255, 0) if pupil_data else (0, 0, 255)
                cv2.putText(result_frame, status_text, (10, self.height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                detection_rate = (detected_count / (idx + 1)) * 100
                cv2.putText(result_frame, f"Rate: {detection_rate:.1f}%", (10, self.height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Prepare frontal view (resize to match height)
                if unwrapped_frontal is not None:
                    # Convert grayscale to BGR if needed
                    if len(unwrapped_frontal.shape) == 2:
                        frontal_bgr = cv2.cvtColor(unwrapped_frontal, cv2.COLOR_GRAY2BGR)
                    else:
                        frontal_bgr = unwrapped_frontal.copy()
                    
                    # Draw transformed ellipses on frontal view (matches GUI behavior)
                    # Draw iris ellipse (green)
                    if iris_ellipse_frontal is not None:
                        if isinstance(iris_ellipse_frontal, tuple) and len(iris_ellipse_frontal) == 3:
                            center, axes, angle = iris_ellipse_frontal
                            cv2.ellipse(frontal_bgr, 
                                       (int(center[0]), int(center[1])), 
                                       (int(axes[0]/2), int(axes[1]/2)),
                                       angle, 0, 360, (0, 255, 0), 2)
                        elif len(iris_ellipse_frontal) >= 5:
                            cx, cy, a, b, angle = iris_ellipse_frontal
                            cv2.ellipse(frontal_bgr, 
                                       (int(cx), int(cy)), 
                                       (int(a), int(b)),
                                       np.rad2deg(angle), 0, 360, (0, 255, 0), 2)
                    
                    # Draw pupil ellipse (blue)
                    if pupil_ellipse_frontal is not None:
                        if isinstance(pupil_ellipse_frontal, tuple) and len(pupil_ellipse_frontal) == 3:
                            center, axes, angle = pupil_ellipse_frontal
                            cv2.ellipse(frontal_bgr, 
                                       (int(center[0]), int(center[1])), 
                                       (int(axes[0]/2), int(axes[1]/2)),
                                       angle, 0, 360, (255, 0, 0), 2)
                        elif len(pupil_ellipse_frontal) >= 5:
                            cx, cy, a, b, angle = pupil_ellipse_frontal
                            cv2.ellipse(frontal_bgr, 
                                       (int(cx), int(cy)), 
                                       (int(a), int(b)),
                                       np.rad2deg(angle), 0, 360, (255, 0, 0), 2)
                    
                    frontal_resized = cv2.resize(frontal_bgr, (frontal_width, self.height))
                    
                    # Add label
                    cv2.putText(frontal_resized, "Frontal View", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Add pupil measurements if available
                    if pupil_measurements:
                        # Main measurement: diameter from area (pixels)
                        pupil_d = pupil_measurements.get('pupil_diameter_from_area_px', 0)
                        text = f"Pupil D: {pupil_d:.1f}px"
                        cv2.putText(frontal_resized, text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Additional info: area
                        if 'pupil_area_px' in pupil_measurements:
                            area = pupil_measurements['pupil_area_px']
                            text_area = f"Area: {area:.0f}px^2"
                            cv2.putText(frontal_resized, text_area, (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            
                            # Axes if available
                            if 'pupil_major_axis_px' in pupil_measurements:
                                major = pupil_measurements['pupil_major_axis_px']
                                minor = pupil_measurements['pupil_minor_axis_px']
                                text_axes = f"Axes: {major:.1f} x {minor:.1f}"
                                cv2.putText(frontal_resized, text_axes, (10, 115),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    # Create black placeholder if no frontal view
                    frontal_resized = np.zeros((self.height, frontal_width, 3), dtype=np.uint8)
                    cv2.putText(frontal_resized, "No frontal view", (10, self.height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                timing_stats['annotate'] += time.perf_counter() - t0
                
                # Write CSV row
                csv_writer.writerow([
                    csv_row['frame_id'],
                    csv_row['pupil_x'], csv_row['pupil_y'], csv_row['pupil_diameter_px'],
                    csv_row['iris_x'], csv_row['iris_y'],
                    csv_row['pupil_ellipse_cx'], csv_row['pupil_ellipse_cy'], 
                    csv_row['pupil_ellipse_a'], csv_row['pupil_ellipse_b'], csv_row['pupil_ellipse_angle'],
                    csv_row['iris_ellipse_cx'], csv_row['iris_ellipse_cy'], 
                    csv_row['iris_ellipse_a'], csv_row['iris_ellipse_b'], csv_row['iris_ellipse_angle'],
                    csv_row['detected']
                ])
                
                # Compose three views side-by-side
                combined = np.hstack([info_frame, result_frame, frontal_resized])
                
                # Write frame
                t0 = time.perf_counter()
                out.write(combined)
                timing_stats['write'] += time.perf_counter() - t0
                
                # Progress report
                if (idx + 1) % 10 == 0:
                    elapsed = time.perf_counter() - overall_start
                    fps_current = (idx + 1) / elapsed
                    eta = (num_frames - idx - 1) / fps_current if fps_current > 0 else 0
                    detection_rate = (detected_count / (idx + 1)) * 100
                    print(f"📊 Frame {idx+1}/{num_frames} | "
                          f"FPS: {fps_current:.2f} | "
                          f"Detected: {detected_count}/{idx+1} ({detection_rate:.1f}%) | "
                          f"ETA: {eta:.1f}s")
        
        finally:
            out.release()
            csv_file.close()
        
        # Print final report
        overall_time = time.perf_counter() - overall_start
        detection_rate = (detected_count / num_frames) * 100 if num_frames > 0 else 0
        
        print(f"\n📊 CSV data saved to: {csv_path}")
        print(f"   {num_frames} rows written")
        
        self._print_timing_report(timing_stats, num_frames, overall_time, 
                                  detection_rate, detected_count, output_path)
    
    def _print_timing_report(self, timing_stats: Dict, num_frames: int, 
                            overall_time: float, detection_rate: float, 
                            detected_count: int, output_path: Path):
        """Print detailed timing report"""
        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Output: {output_path}")
        print(f"🎞️  Frames processed: {num_frames}")
        print(f"👁️  Detection rate: {detection_rate:.1f}% ({detected_count}/{num_frames})")
        print(f"⏱️  Total time: {overall_time:.2f}s")
        print(f"⚡ Average FPS: {num_frames/overall_time:.2f}")
        
        print(f"\n{'─'*60}")
        print(f"TIMING BREAKDOWN (per frame average):")
        print(f"{'─'*60}")
        
        if num_frames > 0:
            print(f"  📖 Read frame:       {timing_stats['read']/num_frames*1000:7.2f} ms  ({timing_stats['read']/overall_time*100:5.1f}%)")
            print(f"  🔧 Preprocessing:    {timing_stats['preprocess']/num_frames*1000:7.2f} ms  ({timing_stats['preprocess']/overall_time*100:5.1f}%)")
            print(f"     ├─ Undistort:     {timing_stats['preprocess_undistort']/num_frames*1000:7.2f} ms  ({timing_stats['preprocess_undistort']/overall_time*100:5.1f}%)")
            print(f"     ├─ Glint removal: {timing_stats['preprocess_glint']/num_frames*1000:7.2f} ms  ({timing_stats['preprocess_glint']/overall_time*100:5.1f}%)")
            print(f"     ├─ Noise reduc.:  {timing_stats['preprocess_noise']/num_frames*1000:7.2f} ms  ({timing_stats['preprocess_noise']/overall_time*100:5.1f}%)")
            print(f"     └─ CLAHE:         {timing_stats['preprocess_clahe']/num_frames*1000:7.2f} ms  ({timing_stats['preprocess_clahe']/overall_time*100:5.1f}%)")
            print(f"  🎯 Detection:        {timing_stats['detect']/num_frames*1000:7.2f} ms  ({timing_stats['detect']/overall_time*100:5.1f}%)")
            print(f"  ✏️  Annotation:       {timing_stats['annotate']/num_frames*1000:7.2f} ms  ({timing_stats['annotate']/overall_time*100:5.1f}%)")
            print(f"  💾 Write frame:      {timing_stats['write']/num_frames*1000:7.2f} ms  ({timing_stats['write']/overall_time*100:5.1f}%)")
            print(f"{'─'*60}")
            print(f"  🔢 Total per frame:  {overall_time/num_frames*1000:7.2f} ms")
        
        print(f"{'='*60}\n")
    
    def close(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Standalone Eye Tracking Pipeline Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 50 frames
  %(prog)s --config pipeline_settings.yaml --video eye1.mp4 --frames 50 --output test.mp4
  
  # Process full video
  %(prog)s --config pipeline_settings.yaml --video eye1.mp4 --output processed.mp4
        """
    )
    
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output video/image')
    parser.add_argument('--frames', type=int, help='Number of frames to process (default: all)')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame number (default: 0)')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = PipelineProcessor(args.config, args.video)
        
        # Determine number of frames
        if args.frames is None:
            num_frames = processor.frame_count - args.start_frame
        else:
            num_frames = min(args.frames, processor.frame_count - args.start_frame)
        
        # Process
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processor.process_frames(args.start_frame, num_frames, output_path)
        
        # Cleanup
        processor.close()
        
        print("✅ Done!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
