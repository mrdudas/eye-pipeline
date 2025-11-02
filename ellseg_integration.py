#!/usr/bin/env python3
"""
EllSeg Integration for Eye Pipeline
Based on https://github.com/RSKothari/EllSeg

This module provides pupil and iris ellipse detection using the pre-trained
EllSeg DenseElNet model. Works even when eyelids occlude the pupil/iris.
"""

import cv2
import copy
import torch
import numpy as np
from pathlib import Path
import sys

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import EllSeg model
try:
    from models.RITnet_v3 import DenseNet2D
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import DenseNet2D: {e}")
    print("Run setup_ellseg.sh first!")
    MODEL_AVAILABLE = False

# Try to download weights if not present
WEIGHTS_URL = "https://github.com/RSKothari/EllSeg/raw/master/weights/all.git_ok"
WEIGHTS_PATH = Path("./weights/ellseg_all.pt")

class EllSegDetector:
    """
    EllSeg-based pupil and iris detector
    
    Features:
    - Robust to eyelid occlusion
    - Detects both pupil AND iris ellipses
    - Uses pre-trained DenseElNet (RITnet_v3) architecture
    """
    
    def __init__(self, weights_path=None, device='cuda', use_fp16=False):
        """
        Initialize EllSeg detector
        
        Parameters:
        -----------
        weights_path : str or Path
            Path to pre-trained weights file (.pt or .git_ok)
        device : str
            'cuda', 'mps', or 'cpu' (auto-detects if CUDA/MPS not available)
        use_fp16 : bool
            Use half precision (FP16) for faster inference (requires GPU)
        """
        # Auto-detect best available device
        if device == 'cuda' and not torch.cuda.is_available():
            # Fallback to MPS (Apple Silicon) or CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        elif device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            device = 'cpu'
        
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device != 'cpu'  # FP16 only on GPU
        
        # Load model architecture (we'll implement simplified version)
        self.model = self._build_model()
        
        # Store last detection for visualization
        self.last_detection = None
        
        # Load weights
        if weights_path is None:
            weights_path = WEIGHTS_PATH
        
        if Path(weights_path).exists():
            self._load_weights(weights_path)
        else:
            print(f"⚠️  Weights not found at {weights_path}")
            print(f"Please download from: {WEIGHTS_URL}")
            print(f"Or train your own model using EllSeg repository")
    
    def _build_model(self):
        """
        Build EllSeg DenseNet2D model (RITnet_v3 architecture)
        """
        if not MODEL_AVAILABLE:
            return None
        
        try:
            # Create model with default parameters
            model = DenseNet2D(
                chz=32,
                growth=1.2,
                actfunc=torch.nn.functional.leaky_relu,
                norm=torch.nn.InstanceNorm2d,
                selfCorr=False,
                disentangle=False
            )
            print("✅ DenseNet2D model created")
            return model
        except Exception as e:
            print(f"❌ Error building model: {e}")
            return None
    
    def _load_weights(self, weights_path):
        """Load pre-trained weights"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if self.model is not None:
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load weights (strict=False allows partial loading)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                
                if len(missing) > 0:
                    print(f"⚠️  Missing keys: {len(missing)}")
                if len(unexpected) > 0:
                    print(f"⚠️  Unexpected keys: {len(unexpected)}")
                
                self.model.to(self.device)
                self.model.eval()
                
                # Convert to FP16 if requested
                if self.use_fp16:
                    self.model = self.model.half()
                    print(f"✅ Loaded EllSeg weights from {weights_path} (FP16)")
                else:
                    print(f"✅ Loaded EllSeg weights from {weights_path}")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            import traceback
            traceback.print_exc()
    
    def preprocess_frame(self, frame, target_size=(320, 240)):
        """
        Preprocess frame for EllSeg input with aspect ratio preservation
        
        Parameters:
        -----------
        frame : np.ndarray
            Input grayscale image (H, W)
        target_size : tuple
            Target size (width, height) for model input
        
        Returns:
        --------
        tensor : torch.Tensor
            Preprocessed frame [1, 1, H, W]
        transform_info : dict
            Transformation info for rescaling output back to original coords
        """
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        orig_h, orig_w = frame.shape
        target_w, target_h = target_size
        
        # Calculate scale to fit image in target size (maintain aspect ratio)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize with aspect ratio preserved
        frame_resized = cv2.resize(frame, (new_w, new_h), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate padding needed
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Add padding to reach target size
        frame_padded = np.pad(frame_resized, 
                             ((pad_top, pad_bottom), (pad_left, pad_right)), 
                             mode='constant', 
                             constant_values=0)
        
        # Normalize
        frame_norm = (frame_padded - frame_padded.mean()) / (frame_padded.std() + 1e-7)
        
        # Convert to tensor [1, 1, H, W]
        tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        
        # Convert to FP16 if model is in FP16
        if self.use_fp16:
            tensor = tensor.half()
        
        # Store transformation info for inverse transform
        transform_info = {
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
            'orig_shape': (orig_h, orig_w),
            'resized_shape': (new_h, new_w)
        }
        
        return tensor.to(self.device), transform_info
    
    def detect(self, frame):
        """
        Detect pupil and iris ellipses
        
        Parameters:
        -----------
        frame : np.ndarray
            Input grayscale image (H, W) or (H, W, 3)
        
        Returns:
        --------
        results : dict
            {
                'pupil_ellipse': (cx, cy, a, b, angle),  # -1s if not detected
                'iris_ellipse': (cx, cy, a, b, angle),   # -1s if not detected
                'seg_map': np.ndarray,  # Segmentation map (0=bg, 1=iris, 2=pupil)
                'confidence': float     # Detection confidence
            }
        """
        if self.model is None:
            return self._fallback_detection(frame)
        
        # Preprocess
        input_tensor, transform_info = self.preprocess_frame(frame)
        
        # Forward pass
        with torch.no_grad():
            # Encoder
            x4, x3, x2, x1, x = self.model.enc(input_tensor)
            
            # Ellipse regression
            latent = torch.mean(x.flatten(start_dim=2), -1)
            elOut = self.model.elReg(x, 0)
            
            # Decoder (segmentation)
            seg_out = self.model.dec(x4, x3, x2, x1, x)
        
        # Post-process
        seg_map = self._get_segmentation_map(seg_out)
        pupil_ellipse, iris_ellipse = self._extract_ellipses(seg_out, elOut, seg_map)
        
        # Rescale to original size using inverse transform
        pupil_ellipse = self._rescale_ellipse(pupil_ellipse, transform_info)
        iris_ellipse = self._rescale_ellipse(iris_ellipse, transform_info)
        seg_map = self._rescale_segmap(seg_map, transform_info)
        
        results = {
            'pupil_ellipse': pupil_ellipse,
            'iris_ellipse': iris_ellipse,
            'seg_map': seg_map,
            'confidence': self._compute_confidence(seg_map)
        }
        
        # Store last detection for visualization
        self.last_detection = results
        
        return results
    
    def _fallback_detection(self, frame):
        """Fallback when model not loaded"""
        h, w = frame.shape[:2] if len(frame.shape) == 2 else frame.shape[:2]
        return {
            'pupil_ellipse': np.array([-1, -1, -1, -1, -1]),
            'iris_ellipse': np.array([-1, -1, -1, -1, -1]),
            'seg_map': np.zeros((h, w), dtype=np.uint8),
            'confidence': 0.0
        }
    
    def _get_segmentation_map(self, seg_out):
        """Convert network output to segmentation map"""
        seg_probs = torch.softmax(seg_out, dim=1)
        seg_map = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
        return seg_map.astype(np.uint8)
    
    def _extract_ellipses(self, seg_out, elOut, seg_map):
        """Extract ellipse parameters from network output"""
        # Method 1: Direct from network regression
        # elOut contains [pupil_params, iris_params]
        # Each: [cx_norm, cy_norm, a_norm, b_norm, angle]
        
        # Method 2: Fit ellipse to segmentation mask
        # More robust, especially with occlusions
        
        pupil_ellipse = self._fit_ellipse_to_mask(seg_map, class_id=2)  # Pupil
        iris_ellipse = self._fit_ellipse_to_mask(seg_map, class_id=1)   # Iris
        
        return pupil_ellipse, iris_ellipse
    
    def _fit_ellipse_to_mask(self, seg_map, class_id):
        """Fit ellipse to segmentation mask"""
        mask = (seg_map == class_id).astype(np.uint8)
        
        if np.sum(mask) < 50:  # Not enough pixels
            return np.array([-1, -1, -1, -1, -1])
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.array([-1, -1, -1, -1, -1])
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        if len(largest) < 5:
            return np.array([-1, -1, -1, -1, -1])
        
        # Fit ellipse
        try:
            ellipse = cv2.fitEllipse(largest)
            # Format: ((cx, cy), (major, minor), angle)
            cx, cy = ellipse[0]
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2  # Semi-axes
            angle = np.deg2rad(ellipse[2])
            
            return np.array([cx, cy, a, b, angle])
        except:
            return np.array([-1, -1, -1, -1, -1])
    
    def _rescale_ellipse(self, ellipse, transform_info):
        """
        Rescale ellipse from model coords to original image coords
        
        Inverse transformation:
        1. Remove padding offset
        2. Scale back to original size
        """
        if np.all(ellipse == -1):
            return ellipse
        
        scale = transform_info['scale']
        pad_left = transform_info['pad_left']
        pad_top = transform_info['pad_top']
        
        cx, cy, a, b, angle = ellipse
        
        # Step 1: Remove padding offset (coords are in padded image space)
        cx_unpadded = cx - pad_left
        cy_unpadded = cy - pad_top
        
        # Step 2: Scale back to original image coordinates
        cx_orig = cx_unpadded / scale
        cy_orig = cy_unpadded / scale
        a_orig = a / scale
        b_orig = b / scale
        # Angle remains the same (rotation invariant)
        
        return np.array([cx_orig, cy_orig, a_orig, b_orig, angle])
    
    def _rescale_segmap(self, seg_map, transform_info):
        """
        Rescale segmentation map to original size
        
        Steps:
        1. Remove padding
        2. Resize to original dimensions
        """
        pad_left = transform_info['pad_left']
        pad_right = transform_info['pad_right']
        pad_top = transform_info['pad_top']
        pad_bottom = transform_info['pad_bottom']
        orig_h, orig_w = transform_info['orig_shape']
        resized_h, resized_w = transform_info['resized_shape']
        
        # Step 1: Remove padding (extract the resized image area)
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            seg_map_unpadded = seg_map[pad_top:pad_top+resized_h, 
                                       pad_left:pad_left+resized_w]
        else:
            seg_map_unpadded = seg_map
        
        # Step 2: Resize back to original dimensions
        seg_map_orig = cv2.resize(seg_map_unpadded, (orig_w, orig_h), 
                                 interpolation=cv2.INTER_NEAREST)
        
        return seg_map_orig
    
    def _compute_confidence(self, seg_map):
        """Compute detection confidence based on segmentation quality"""
        pupil_pixels = np.sum(seg_map == 2)
        iris_pixels = np.sum(seg_map == 1)
        total_pixels = seg_map.size
        
        if pupil_pixels < 50:
            return 0.0
        
        # Simple confidence: ratio of detected pixels
        confidence = (pupil_pixels + iris_pixels) / total_pixels
        return min(confidence * 10, 1.0)  # Scale up and cap at 1.0
    
    def visualize(self, frame, results):
        """
        Visualize detection results
        
        Parameters:
        -----------
        frame : np.ndarray
            Original frame
        results : dict
            Output from detect()
        
        Returns:
        --------
        vis_frame : np.ndarray
            Frame with overlaid detections
        """
        vis_frame = frame.copy()
        if len(vis_frame.shape) == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
        
        seg_map = results['seg_map']
        
        # Overlay segmentation (semi-transparent)
        overlay = vis_frame.copy()
        overlay[seg_map == 1] = [0, 255, 0]    # Iris = green
        overlay[seg_map == 2] = [255, 255, 0]  # Pupil = yellow
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # Draw pupil ellipse (red)
        pupil_ellipse = results['pupil_ellipse']
        if not np.all(pupil_ellipse == -1):
            cx, cy, a, b, angle = pupil_ellipse
            cv2.ellipse(vis_frame, 
                       (int(cx), int(cy)), 
                       (int(a), int(b)), 
                       np.rad2deg(angle), 
                       0, 360, (0, 0, 255), 2)
            cv2.circle(vis_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(vis_frame, "Pupil", (int(cx)+10, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw iris ellipse (green)
        iris_ellipse = results['iris_ellipse']
        if not np.all(iris_ellipse == -1):
            cx, cy, a, b, angle = iris_ellipse
            cv2.ellipse(vis_frame, 
                       (int(cx), int(cy)), 
                       (int(a), int(b)), 
                       np.rad2deg(angle), 
                       0, 360, (0, 255, 0), 2)
            cv2.putText(vis_frame, "Iris", (int(cx)+10, int(cy)+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw confidence
        conf = results['confidence']
        cv2.putText(vis_frame, f"Conf: {conf:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame


def download_weights():
    """Download pre-trained EllSeg weights"""
    import urllib.request
    
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    
    print(f"Downloading EllSeg weights from {WEIGHTS_URL}...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print(f"✅ Downloaded to {WEIGHTS_PATH}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Please download manually from:")
        print(WEIGHTS_URL)


if __name__ == "__main__":
    """
    Test EllSeg detector
    """
    import sys
    
    # Check if weights exist
    if not WEIGHTS_PATH.exists():
        print("Weights not found. Attempting download...")
        download_weights()
    
    # Create detector
    detector = EllSegDetector(device='cpu')  # Use CPU for testing
    
    print("\n" + "="*60)
    print("EllSeg Integration Status:")
    print("="*60)
    print(f"Model loaded: {detector.model is not None}")
    print(f"Weights path: {WEIGHTS_PATH}")
    print(f"Weights exist: {WEIGHTS_PATH.exists()}")
    print("\nTo use EllSeg:")
    print("1. Clone: git clone https://github.com/RSKothari/EllSeg")
    print("2. Copy models/* and utils.py to eye_pipeline/")
    print("3. Download weights/all.git_ok to weights/ellseg_all.pt")
    print("4. Test with: python ellseg_integration.py")
    print("="*60)
