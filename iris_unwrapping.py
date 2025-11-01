#!/usr/bin/env python3
"""
Iris Unwrapping Module
Provides multiple methods for transforming elliptical iris to frontal/unwrapped views
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Optional


class IrisUnwrapper:
    """
    Iris unwrapping utilities for converting elliptical iris projections
    to frontal circular views and measuring pupil size
    """
    
    def __init__(self):
        """Initialize unwrapper with standard iris diameter"""
        # Standard iris diameter in mm (average human iris)
        self.IRIS_DIAMETER_MM = 11.8
    
    def unwrap_to_frontal_view_v2(
        self,
        frame: np.ndarray,
        iris_ellipse: np.ndarray,
        pupil_ellipse: np.ndarray,
        iris_radius_px: int = 150
    ) -> tuple:
        """
        Create frontal view using step-by-step transformation (Version 2):
        1. Create 400x400 empty bitmap
        2. Copy original image centered on pupil
        3. Rotate so iris major axis is vertical
        4. Scale vertically to make iris 150px tall
        5. Scale horizontally to make iris 150px wide  
        6. Rotate back by same angle
        
        Parameters:
        -----------
        frame : np.ndarray
            Input grayscale frame
        iris_ellipse : np.ndarray
            Iris ellipse parameters [cx, cy, a, b, angle]
        pupil_ellipse : np.ndarray
            Pupil ellipse parameters [cx, cy, a, b, angle]
        iris_radius_px : int
            Target iris radius in pixels (default 150)
            
        Returns:
        --------
        frontal_view : np.ndarray
            400x400 frontal view image
        info : dict
            Transformation information
        """
        output_size = 400
        center = output_size // 2  # 200, 200
        
        # Parse ellipses
        iris_cx, iris_cy, iris_a, iris_b, iris_angle = iris_ellipse
        pupil_cx, pupil_cy = pupil_ellipse[0], pupil_ellipse[1]
        
        # Determine major/minor axes
        if iris_a > iris_b:
            iris_major = iris_a
            iris_minor = iris_b
            major_angle = iris_angle
        else:
            iris_major = iris_b
            iris_minor = iris_a
            major_angle = iris_angle + np.pi/2
        
        # Step 1: Create 400x400 empty bitmap
        canvas = np.zeros((output_size, output_size), dtype=frame.dtype)
        
        # Step 2: Copy original image centered on pupil
        # Translation: pupil center -> canvas center
        tx = center - pupil_cx
        ty = center - pupil_cy
        
        M_translate = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
        
        canvas = cv2.warpAffine(frame, M_translate, (output_size, output_size))
        
        # Update iris center position after translation
        iris_cx_translated = iris_cx + tx
        iris_cy_translated = iris_cy + ty
        
        # Step 3: Rotate so iris major axis is vertical
        # Major axis angle relative to horizontal, rotate to make it vertical (90 degrees)
        rotation_angle_deg = np.rad2deg(major_angle) - 90
        
        M_rotate = cv2.getRotationMatrix2D((center, center), rotation_angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, M_rotate, (output_size, output_size))
        
        # Update iris center after rotation
        iris_center_rotated = M_rotate @ np.array([iris_cx_translated, iris_cy_translated, 1])
        iris_cx_rotated = iris_center_rotated[0]
        iris_cy_rotated = iris_center_rotated[1]
        
        # After rotation, major axis is vertical, minor is horizontal
        # Step 4: Scale vertically to make iris 150px tall (major axis)
        scale_y = iris_radius_px / iris_major
        
        M_scale_y = np.float32([
            [1, 0, 0],
            [0, scale_y, center * (1 - scale_y)]
        ])
        canvas = cv2.warpAffine(canvas, M_scale_y, (output_size, output_size))
        
        # Update iris center after vertical scaling
        iris_cy_scaled = iris_cy_rotated * scale_y + center * (1 - scale_y)
        
        # Step 5: Scale horizontally to make iris 150px wide (minor axis)
        scale_x = iris_radius_px / iris_minor
        
        M_scale_x = np.float32([
            [scale_x, 0, center * (1 - scale_x)],
            [0, 1, 0]
        ])
        canvas = cv2.warpAffine(canvas, M_scale_x, (output_size, output_size))
        
        # Update iris center after horizontal scaling
        iris_cx_scaled = iris_cx_rotated * scale_x + center * (1 - scale_x)
        
        # Step 6: Rotate back by the same angle (opposite direction)
        M_rotate_back = cv2.getRotationMatrix2D((center, center), -rotation_angle_deg, 1.0)
        frontal_view = cv2.warpAffine(canvas, M_rotate_back, (output_size, output_size))
        
        # Calculate viewing angle from aspect ratio
        aspect_ratio = iris_major / iris_minor
        viewing_angle_rad = np.arccos(1.0 / aspect_ratio) if aspect_ratio > 1 else 0
        viewing_angle_deg = np.rad2deg(viewing_angle_rad)
        
        info = {
            'iris_major_axis': iris_major,
            'iris_minor_axis': iris_minor,
            'aspect_ratio': aspect_ratio,
            'viewing_angle_rad': viewing_angle_rad,
            'viewing_angle_deg': viewing_angle_deg,
            'rotation_angle_deg': rotation_angle_deg,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'iris_center_final': (center, center),  # Should be at center after all transforms
            'scale_to_major': iris_radius_px / iris_major,
            'scale_to_minor': iris_radius_px / iris_minor
        }
        
        return frontal_view, info
    
    def visualize_transformation_steps(
        self,
        frame: np.ndarray,
        iris_ellipse: np.ndarray,
        pupil_ellipse: np.ndarray,
        iris_radius_px: int = 150
    ) -> list:
        """
        Visualize all 6 transformation steps with ellipses drawn
        
        Returns:
        --------
        list of dicts, each containing:
            - 'image': Step result image
            - 'iris_ellipse': Iris ellipse at this step ((cx, cy), (w, h), angle)
            - 'pupil_ellipse': Pupil ellipse at this step
            - 'title': Step description
        """
        output_size = 400
        center = output_size // 2
        
        # Parse ellipses
        iris_cx, iris_cy, iris_a, iris_b, iris_angle = iris_ellipse
        pupil_cx, pupil_cy, pupil_a, pupil_b, pupil_angle = pupil_ellipse
        
        # Determine major/minor axes
        if iris_a > iris_b:
            iris_major = iris_a
            iris_minor = iris_b
            major_angle = iris_angle
        else:
            iris_major = iris_b
            iris_minor = iris_a
            major_angle = iris_angle + np.pi/2
        
        rotation_angle_deg = np.rad2deg(major_angle) - 90
        scale_y = iris_radius_px / iris_major
        scale_x = iris_radius_px / iris_minor
        
        # Convert frame to grayscale if needed (for warpAffine operations)
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()
        
        steps = []
        
        # STEP 0: Original image with ellipses
        # Convert to BGR if grayscale
        if len(frame.shape) == 2:
            step0 = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        else:
            step0 = frame.copy()
        
        cv2.ellipse(step0, (int(iris_cx), int(iris_cy)), 
                   (int(iris_a), int(iris_b)), np.rad2deg(iris_angle), 
                   0, 360, (0, 255, 0), 2)
        cv2.ellipse(step0, (int(pupil_cx), int(pupil_cy)), 
                   (int(pupil_a), int(pupil_b)), np.rad2deg(pupil_angle), 
                   0, 360, (255, 0, 0), 2)
        steps.append({
            'image': step0,
            'iris_ellipse': ((iris_cx, iris_cy), (2*iris_a, 2*iris_b), np.rad2deg(iris_angle)),
            'pupil_ellipse': ((pupil_cx, pupil_cy), (2*pupil_a, 2*pupil_b), np.rad2deg(pupil_angle)),
            'title': '0. Original Image'
        })
        
        # STEP 1: Create 400x400 empty bitmap
        canvas = np.zeros((output_size, output_size), dtype=frame.dtype)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        steps.append({
            'image': canvas_bgr.copy(),
            'iris_ellipse': None,
            'pupil_ellipse': None,
            'title': '1. Empty 400x400 Canvas'
        })
        
        # STEP 2: Copy original centered on pupil
        tx = center - pupil_cx
        ty = center - pupil_cy
        
        M_translate = np.float32([
            [1, 0, tx],
            [0, 1, ty]
        ])
        
        canvas = cv2.warpAffine(frame_gray, M_translate, (output_size, output_size))
        
        # Transform ellipse centers
        iris_cx_t = iris_cx + tx
        iris_cy_t = iris_cy + ty
        pupil_cx_t = pupil_cx + tx
        pupil_cy_t = pupil_cy + ty
        
        canvas_bgr = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
        cv2.ellipse(canvas_bgr, (int(iris_cx_t), int(iris_cy_t)), 
                   (int(iris_a), int(iris_b)), np.rad2deg(iris_angle), 
                   0, 360, (0, 255, 0), 2)
        cv2.ellipse(canvas_bgr, (int(pupil_cx_t), int(pupil_cy_t)), 
                   (int(pupil_a), int(pupil_b)), np.rad2deg(pupil_angle), 
                   0, 360, (255, 0, 0), 2)
        steps.append({
            'image': canvas_bgr,
            'iris_ellipse': ((iris_cx_t, iris_cy_t), (2*iris_a, 2*iris_b), np.rad2deg(iris_angle)),
            'pupil_ellipse': ((pupil_cx_t, pupil_cy_t), (2*pupil_a, 2*pupil_b), np.rad2deg(pupil_angle)),
            'title': '2. Centered on Pupil'
        })
        
        # STEP 3: Rotate to make major axis vertical
        # OpenCV rotation: positive angle = counter-clockwise for image
        # OpenCV ellipse: positive angle = clockwise for ellipse
        # So we need to SUBTRACT the rotation angle from ellipse to keep them aligned
        M_rotate = cv2.getRotationMatrix2D((center, center), rotation_angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, M_rotate, (output_size, output_size))
        
        # Transform ellipse centers
        iris_center_rot = M_rotate @ np.array([iris_cx_t, iris_cy_t, 1])
        pupil_center_rot = M_rotate @ np.array([pupil_cx_t, pupil_cy_t, 1])
        iris_cx_r = iris_center_rot[0]
        iris_cy_r = iris_center_rot[1]
        pupil_cx_r = pupil_center_rot[0]
        pupil_cy_r = pupil_center_rot[1]
        
        # Rotate ellipse angles opposite to image rotation (OpenCV convention)
        iris_angle_r = iris_angle - np.deg2rad(rotation_angle_deg)
        pupil_angle_r = pupil_angle - np.deg2rad(rotation_angle_deg)
        
        canvas_bgr = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
        cv2.ellipse(canvas_bgr, (int(iris_cx_r), int(iris_cy_r)), 
                   (int(iris_a), int(iris_b)), np.rad2deg(iris_angle_r), 
                   0, 360, (0, 255, 0), 2)
        cv2.ellipse(canvas_bgr, (int(pupil_cx_r), int(pupil_cy_r)), 
                   (int(pupil_a), int(pupil_b)), np.rad2deg(pupil_angle_r), 
                   0, 360, (255, 0, 0), 2)
        steps.append({
            'image': canvas_bgr,
            'iris_ellipse': ((iris_cx_r, iris_cy_r), (2*iris_a, 2*iris_b), np.rad2deg(iris_angle_r)),
            'pupil_ellipse': ((pupil_cx_r, pupil_cy_r), (2*pupil_a, 2*pupil_b), np.rad2deg(pupil_angle_r)),
            'title': f'3. Rotate {rotation_angle_deg:.1f}° (Major→Vertical)'
        })
        
        # STEP 4: Scale Y (height -> 150px)
        M_scale_y = np.float32([
            [1, 0, 0],
            [0, scale_y, center * (1 - scale_y)]
        ])
        canvas = cv2.warpAffine(canvas, M_scale_y, (output_size, output_size))
        
        iris_cy_sy = iris_cy_r * scale_y + center * (1 - scale_y)
        pupil_cy_sy = pupil_cy_r * scale_y + center * (1 - scale_y)
        iris_b_sy = iris_b * scale_y
        pupil_b_sy = pupil_b * scale_y
        
        canvas_bgr = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
        cv2.ellipse(canvas_bgr, (int(iris_cx_r), int(iris_cy_sy)), 
                   (int(iris_a), int(iris_b_sy)), np.rad2deg(iris_angle_r), 
                   0, 360, (0, 255, 0), 2)
        cv2.ellipse(canvas_bgr, (int(pupil_cx_r), int(pupil_cy_sy)), 
                   (int(pupil_a), int(pupil_b_sy)), np.rad2deg(pupil_angle_r), 
                   0, 360, (255, 0, 0), 2)
        steps.append({
            'image': canvas_bgr,
            'iris_ellipse': ((iris_cx_r, iris_cy_sy), (2*iris_a, 2*iris_b_sy), np.rad2deg(iris_angle_r)),
            'pupil_ellipse': ((pupil_cx_r, pupil_cy_sy), (2*pupil_a, 2*pupil_b_sy), np.rad2deg(pupil_angle_r)),
            'title': f'4. Scale Y {scale_y:.2f}x (Height→150px)'
        })
        
        # STEP 5: Scale X (width -> 150px)
        M_scale_x = np.float32([
            [scale_x, 0, center * (1 - scale_x)],
            [0, 1, 0]
        ])
        canvas = cv2.warpAffine(canvas, M_scale_x, (output_size, output_size))
        
        iris_cx_sx = iris_cx_r * scale_x + center * (1 - scale_x)
        pupil_cx_sx = pupil_cx_r * scale_x + center * (1 - scale_x)
        iris_a_sx = iris_a * scale_x
        pupil_a_sx = pupil_a * scale_x
        
        canvas_bgr = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
        cv2.ellipse(canvas_bgr, (int(iris_cx_sx), int(iris_cy_sy)), 
                   (int(iris_a_sx), int(iris_b_sy)), np.rad2deg(iris_angle_r), 
                   0, 360, (0, 255, 0), 2)
        cv2.ellipse(canvas_bgr, (int(pupil_cx_sx), int(pupil_cy_sy)), 
                   (int(pupil_a_sx), int(pupil_b_sy)), np.rad2deg(pupil_angle_r), 
                   0, 360, (255, 0, 0), 2)
        steps.append({
            'image': canvas_bgr,
            'iris_ellipse': ((iris_cx_sx, iris_cy_sy), (2*iris_a_sx, 2*iris_b_sy), np.rad2deg(iris_angle_r)),
            'pupil_ellipse': ((pupil_cx_sx, pupil_cy_sy), (2*pupil_a_sx, 2*pupil_b_sy), np.rad2deg(pupil_angle_r)),
            'title': f'5. Scale X {scale_x:.2f}x (Width→150px)'
        })
        
        # STEP 6: Rotate back
        # We rotated by +rotation_angle_deg counter-clockwise, so rotate back the same way
        # but use negative angle to go the opposite direction
        M_rotate_back = cv2.getRotationMatrix2D((center, center), -rotation_angle_deg, 1.0)
        frontal_view = cv2.warpAffine(canvas, M_rotate_back, (output_size, output_size))
        
        # Transform ellipse centers back
        iris_center_final = M_rotate_back @ np.array([iris_cx_sx, iris_cy_sy, 1])
        pupil_center_final = M_rotate_back @ np.array([pupil_cx_sx, pupil_cy_sy, 1])
        
        # Add back the rotation (opposite of step 3 subtraction)
        iris_angle_final = iris_angle_r + np.deg2rad(rotation_angle_deg)
        pupil_angle_final = pupil_angle_r + np.deg2rad(rotation_angle_deg)
        
        frontal_bgr = cv2.cvtColor(frontal_view.copy(), cv2.COLOR_GRAY2BGR)
        cv2.ellipse(frontal_bgr, (int(iris_center_final[0]), int(iris_center_final[1])), 
                   (int(iris_a_sx), int(iris_b_sy)), np.rad2deg(iris_angle_final), 
                   0, 360, (0, 255, 0), 2)
        cv2.ellipse(frontal_bgr, (int(pupil_center_final[0]), int(pupil_center_final[1])), 
                   (int(pupil_a_sx), int(pupil_b_sy)), np.rad2deg(pupil_angle_final), 
                   0, 360, (255, 0, 0), 2)
        steps.append({
            'image': frontal_bgr,
            'iris_ellipse': ((iris_center_final[0], iris_center_final[1]), 
                           (2*iris_a_sx, 2*iris_b_sy), np.rad2deg(iris_angle_final)),
            'pupil_ellipse': ((pupil_center_final[0], pupil_center_final[1]), 
                            (2*pupil_a_sx, 2*pupil_b_sy), np.rad2deg(pupil_angle_final)),
            'title': f'6. Rotate Back -{rotation_angle_deg:.1f}° (Final)'
        })
        
        return steps
    
    def unwrap_to_frontal_view(
        self, 
        frame: np.ndarray, 
        iris_ellipse: np.ndarray,
        pupil_ellipse: Optional[np.ndarray] = None,
        iris_radius_px: int = 150
    ) -> Tuple[np.ndarray, dict]:
        """
        Transform elliptical iris to circular frontal view
        Centered on pupil if available
        
        NEW: Iris is always a circle with radius = iris_radius_px (default 300)
        
        Concept:
        - Real iris is circular in 3D with fixed size
        - Appears elliptical due to viewing angle
        - Undo perspective distortion → frontal circular view
        - Iris radius is ALWAYS iris_radius_px pixels in output
        - Center on pupil center (not iris center)
        
        Parameters:
        -----------
        frame : np.ndarray
            Input grayscale frame
        iris_ellipse : np.ndarray
            Iris ellipse parameters [cx, cy, a, b, angle] (semi-axes)
        pupil_ellipse : np.ndarray, optional
            Pupil ellipse parameters [cx, cy, a, b, angle]
        iris_radius_px : int
            Target iris radius in pixels (default 300 = 600px diameter)
            
        Returns:
        --------
        frontal_view : np.ndarray
            Unwrapped iris in frontal view (circular, iris radius = iris_radius_px)
        info : dict
            Transformation info (scale factors, etc.)
        """
        if iris_ellipse is None or len(iris_ellipse) < 5:
            return None, {}
        
        iris_cx, iris_cy, iris_a, iris_b, iris_angle = iris_ellipse
        
        # Ensure we have valid dimensions
        if iris_a <= 0 or iris_b <= 0:
            return None, {}
        
        # Use pupil center if available, otherwise iris center
        if pupil_ellipse is not None and len(pupil_ellipse) >= 5:
            pupil_cx, pupil_cy = pupil_ellipse[0:2]
            center_x, center_y = pupil_cx, pupil_cy
        else:
            center_x, center_y = iris_cx, iris_cy
        
        # Output size is 2 * iris_radius_px (full diameter) plus margin
        output_size = iris_radius_px * 2 + 100  # +100 for margin
        
        # 1. Extract larger ROI around CENTER (pupil or iris)
        # ROI should be large enough to contain the iris after transformation
        roi_size = int(max(iris_a, iris_b) * 3)  # Larger to accommodate transformation
        roi_size = max(roi_size, 100)
        
        try:
            # Get sub-pixel accurate ROI centered on pupil/iris center
            roi = cv2.getRectSubPix(frame, (roi_size, roi_size), (center_x, center_y))
        except cv2.error:
            # Handle edge cases where ROI goes outside frame
            x1 = max(0, int(center_x - roi_size/2))
            y1 = max(0, int(center_y - roi_size/2))
            x2 = min(frame.shape[1], int(center_x + roi_size/2))
            y2 = min(frame.shape[0], int(center_y + roi_size/2))
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None, {}
        
        # 2. Calculate aspect ratio correction (based on iris ellipse)
        # Major axis / Minor axis = viewing angle distortion
        aspect_ratio = max(iris_a, iris_b) / min(iris_a, iris_b)
        
        # 3. Calculate target scale to make iris a perfect circle
        # BOTH axes must become iris_radius_px
        iris_major_axis = max(iris_a, iris_b)
        iris_minor_axis = min(iris_a, iris_b)
        
        # Scale to make BOTH axes equal to iris_radius_px
        # Major axis → iris_radius_px: scale_major = iris_radius_px / major
        # Minor axis → iris_radius_px: scale_minor = iris_radius_px / minor
        scale_to_major = iris_radius_px / iris_major_axis
        scale_to_minor = iris_radius_px / iris_minor_axis
        
        # 4. Apply scaling transformation
        # Build transformation matrix that scales along ellipse axes
        # The center should remain at the ROI center (which is the pupil/iris center)
        roi_center_x = roi.shape[1] / 2
        roi_center_y = roi.shape[0] / 2
        
        # Rotation matrix to align with ellipse axes
        cos_a = np.cos(iris_angle)
        sin_a = np.sin(iris_angle)
        
        # Determine scale factors based on which axis is major
        if iris_a > iris_b:
            # Major axis (a) along angle direction → scale to iris_radius_px
            # Minor axis (b) perpendicular → scale to iris_radius_px
            scale_major = scale_to_major  # Scale major axis to target
            scale_minor = scale_to_minor  # Scale minor axis to target (compensates viewing angle)
        else:
            # Minor axis (b) along angle direction
            scale_major = scale_to_minor
            scale_minor = scale_to_major
        
        # Calculate output ROI size after transformation
        # Make it large enough to hold the scaled iris
        output_roi_size = int(max(roi_size * scale_major, roi_size * scale_minor) * 1.2)
        output_roi_size = max(output_roi_size, output_size)
        
        # Center of output image
        output_center_x = output_roi_size / 2
        output_center_y = output_roi_size / 2
        
        # Combined transformation: scale along ellipse axes around ROI center
        # Then translate to output center
        # M = T(output_center) * R(-θ) * S * R(θ) * T(-roi_center)
        
        # Build the transformation matrix step by step
        # 1. Translate to origin (from roi_center)
        # 2. Rotate to align with ellipse axes
        # 3. Scale
        # 4. Rotate back
        # 5. Translate to output center
        
        # Combined matrix for steps 2-4 (rotation + scale + rotation back)
        M_scale = np.array([
            [cos_a**2 * scale_major + sin_a**2 * scale_minor, 
             cos_a * sin_a * (scale_major - scale_minor)],
            [cos_a * sin_a * (scale_major - scale_minor),
             sin_a**2 * scale_major + cos_a**2 * scale_minor]
        ], dtype=np.float32)
        
        # Translation to move ROI center to output center with scaling
        tx = output_center_x - (M_scale[0,0] * roi_center_x + M_scale[0,1] * roi_center_y)
        ty = output_center_y - (M_scale[1,0] * roi_center_x + M_scale[1,1] * roi_center_y)
        
        # Full affine transformation matrix
        M = np.array([
            [M_scale[0,0], M_scale[0,1], tx],
            [M_scale[1,0], M_scale[1,1], ty]
        ], dtype=np.float32)
        
        corrected = cv2.warpAffine(roi, M, (output_roi_size, output_roi_size))
        
        # 5. Extract centered region with iris at radius iris_radius_px
        # The iris center is now at (output_center_x, output_center_y)
        output_center = output_roi_size // 2
        half_size = output_size // 2
        
        # Extract square region of size output_size centered on iris
        x1 = max(0, output_center - half_size)
        y1 = max(0, output_center - half_size)
        x2 = min(output_roi_size, output_center + half_size)
        y2 = min(output_roi_size, output_center + half_size)
        
        frontal = corrected[y1:y2, x1:x2]
        
        # Ensure output is exactly output_size x output_size
        if frontal.shape[0] != output_size or frontal.shape[1] != output_size:
            frontal = cv2.resize(frontal, (output_size, output_size))
        
        # 6. Create circular mask (iris radius = iris_radius_px)
        mask = np.zeros((output_size, output_size), dtype=np.uint8)
        mask_center = output_size // 2
        cv2.circle(mask, (mask_center, mask_center), iris_radius_px, 255, -1)
        frontal_view = cv2.bitwise_and(frontal, frontal, mask=mask)
        
        # Store transformation info (use iris ellipse for viewing angle)
        info = {
            'aspect_ratio': aspect_ratio,
            'scale_major': scale_major,
            'scale_minor': scale_minor,
            'scale_to_major': scale_to_major,
            'scale_to_minor': scale_to_minor,
            'iris_angle_rad': iris_angle,
            'iris_radius_px': iris_radius_px,
            'output_size': output_size,
            'viewing_angle_deg': np.rad2deg(np.arccos(min(iris_a, iris_b) / max(iris_a, iris_b))),
            'pupil_centered': pupil_ellipse is not None,
            'roi_size': roi_size,
            'center_point': (center_x, center_y)
        }
        
        return frontal_view, info
    
    def measure_pupil_from_ellipse(
        self,
        pupil_ellipse: np.ndarray,
        iris_ellipse: np.ndarray
    ) -> dict:
        """
        Measure pupil diameter from ellipse, compensating for viewing angle
        
        Concept:
        - Pupil is circular in 3D (like iris)
        - Appears elliptical due to same viewing angle as iris
        - Use iris ellipse to determine viewing angle
        - Compensate pupil measurement accordingly
        
        Parameters:
        -----------
        pupil_ellipse : np.ndarray
            Pupil ellipse [cx, cy, a, b, angle]
        iris_ellipse : np.ndarray
            Iris ellipse [cx, cy, a, b, angle] (for viewing angle)
            
        Returns:
        --------
        dict with:
            - pupil_diameter_px: Compensated pupil diameter in pixels
            - pupil_diameter_mm: Pupil diameter in mm (using iris as reference)
            - viewing_angle_deg: Viewing angle from iris
            - pupil_ellipse_major_px: Major axis of pupil ellipse
            - pupil_ellipse_minor_px: Minor axis of pupil ellipse
        """
        if pupil_ellipse is None or len(pupil_ellipse) < 5:
            return None
        if iris_ellipse is None or len(iris_ellipse) < 5:
            return None
            
        pupil_a, pupil_b = pupil_ellipse[2:4]
        iris_a, iris_b = iris_ellipse[2:4]
        
        if pupil_a <= 0 or pupil_b <= 0 or iris_a <= 0 or iris_b <= 0:
            return None
        
        # Calculate viewing angle from iris ellipse
        # cos(θ) = minor_axis / major_axis
        viewing_angle_rad = np.arccos(min(iris_b, iris_a) / max(iris_a, iris_b))
        viewing_angle_deg = np.rad2deg(viewing_angle_rad)
        
        # Pupil major axis (the one we see, not compressed)
        pupil_major_px = max(pupil_a, pupil_b) * 2  # diameter
        
        # Compensate for viewing angle to get true circular diameter
        # If viewed at angle θ, the minor axis is compressed by cos(θ)
        # So true_diameter = major_axis (the uncompressed dimension)
        pupil_diameter_px = pupil_major_px
        
        # Convert to mm using iris as reference
        # Iris diameter in pixels (frontal, compensated)
        iris_major_px = max(iris_a, iris_b) * 2
        iris_diameter_px = iris_major_px  # True circular diameter
        
        # Scale: iris_diameter_px corresponds to IRIS_DIAMETER_MM
        px_to_mm = self.IRIS_DIAMETER_MM / iris_diameter_px
        pupil_diameter_mm = pupil_diameter_px * px_to_mm
        
        return {
            'pupil_diameter_px': pupil_diameter_px,
            'pupil_diameter_mm': pupil_diameter_mm,
            'viewing_angle_deg': viewing_angle_deg,
            'pupil_ellipse_major_px': pupil_major_px,
            'pupil_ellipse_minor_px': min(pupil_a, pupil_b) * 2,
            'iris_reference_px': iris_diameter_px,
            'px_to_mm_ratio': px_to_mm
        }
    
    def measure_pupil_from_frontal(
        self,
        frontal_view: np.ndarray,
        pupil_ellipse_frontal: tuple,
        iris_ellipse: np.ndarray
    ) -> dict:
        """
        Measure pupil diameter from transformed pupil ellipse in frontal view
        
        Uses the transformed pupil ellipse (after perspective correction)
        to calculate pupil area and equivalent circular diameter.
        
        Parameters:
        -----------
        frontal_view : np.ndarray
            Perspective-corrected frontal image (for visualization)
        pupil_ellipse_frontal : tuple
            Transformed pupil ellipse ((cx, cy), (width, height), angle_deg)
        iris_ellipse : np.ndarray
            Original iris ellipse (for mm conversion if needed)
            
        Returns:
        --------
        dict with:
            - pupil_area_px: Area of pupil ellipse in frontal view
            - pupil_diameter_from_area_px: Equivalent circular diameter
            - pupil_ellipse_axes: (major, minor) axes in pixels
        """
        if pupil_ellipse_frontal is None:
            return None
        
        # Parse transformed ellipse
        # Format: ((cx, cy), (width, height), angle_deg)
        if isinstance(pupil_ellipse_frontal, tuple) and len(pupil_ellipse_frontal) == 3:
            center, axes, angle = pupil_ellipse_frontal
            width, height = axes
            # axes are full width/height, convert to semi-axes
            a = width / 2.0
            b = height / 2.0
        elif len(pupil_ellipse_frontal) >= 5:
            # Format: [cx, cy, a, b, angle]
            cx, cy, a, b, angle = pupil_ellipse_frontal[:5]
        else:
            return None
        
        # Calculate ellipse area: Area = π * a * b
        pupil_area_px = np.pi * a * b
        
        # Calculate equivalent circle diameter from area
        # Area = π * r^2  =>  r = sqrt(Area / π)  =>  diameter = 2 * sqrt(Area / π)
        pupil_diameter_from_area_px = 2 * np.sqrt(pupil_area_px / np.pi)
        
        # Major and minor axes (diameters)
        major_axis_px = max(a, b) * 2
        minor_axis_px = min(a, b) * 2
        
        return {
            'pupil_area_px': pupil_area_px,
            'pupil_diameter_from_area_px': pupil_diameter_from_area_px,
            'pupil_major_axis_px': major_axis_px,
            'pupil_minor_axis_px': minor_axis_px,
            'detection_method': 'transformed_ellipse'
        }
    
    def transform_ellipse_to_frontal_v2(
        self,
        ellipse: np.ndarray,
        iris_ellipse: np.ndarray,
        pupil_ellipse: np.ndarray,
        frontal_info: dict
    ) -> tuple:
        """
        Transform ellipse to frontal view coordinates (V2 - step-by-step)
        Follows EXACTLY same transformation steps as visualize_transformation_steps
        
        Parameters:
        -----------
        ellipse : np.ndarray
            Ellipse to transform [cx, cy, a, b, angle]
        iris_ellipse : np.ndarray
            Iris ellipse (for reference)
        pupil_ellipse : np.ndarray
            Pupil ellipse (center point)
        frontal_info : dict
            Transformation info from unwrap_to_frontal_view_v2
            
        Returns:
        --------
        tuple : Ellipse in frontal view coordinates ((cx, cy), (2*a, 2*b), angle_deg)
        """
        if ellipse is None or len(ellipse) < 5:
            return None
        
        output_size = 400
        center = output_size // 2
        
        # Parse ellipses
        cx, cy, a, b, angle = ellipse
        iris_cx, iris_cy, iris_a, iris_b, iris_angle = iris_ellipse
        pupil_cx, pupil_cy = pupil_ellipse[0], pupil_ellipse[1]
        
        # Get transformation parameters
        rotation_angle_deg = frontal_info['rotation_angle_deg']
        scale_x = frontal_info['scale_x']
        scale_y = frontal_info['scale_y']
        
        # STEP 2: Translate (pupil center -> canvas center)
        tx = center - pupil_cx
        ty = center - pupil_cy
        cx_t = cx + tx
        cy_t = cy + ty
        # angle stays same
        
        # STEP 3: Rotate to make major axis vertical
        M_rotate = cv2.getRotationMatrix2D((center, center), rotation_angle_deg, 1.0)
        center_rot = M_rotate @ np.array([cx_t, cy_t, 1])
        cx_r = center_rot[0]
        cy_r = center_rot[1]
        angle_r = angle - np.deg2rad(rotation_angle_deg)
        
        # STEP 4: Scale Y (height -> 150px)
        cy_sy = cy_r * scale_y + center * (1 - scale_y)
        b_sy = b * scale_y
        # cx_r, a, angle_r stay same
        
        # STEP 5: Scale X (width -> 150px)
        cx_sx = cx_r * scale_x + center * (1 - scale_x)
        a_sx = a * scale_x
        # cy_sy, b_sy, angle_r stay same
        
        # STEP 6: Rotate back
        M_rotate_back = cv2.getRotationMatrix2D((center, center), -rotation_angle_deg, 1.0)
        center_final = M_rotate_back @ np.array([cx_sx, cy_sy, 1])
        cx_final = center_final[0]
        cy_final = center_final[1]
        angle_final = angle_r + np.deg2rad(rotation_angle_deg)
        
        # Return in OpenCV format: ((cx, cy), (width, height), angle_deg)
        return ((cx_final, cy_final), (2*a_sx, 2*b_sy), np.rad2deg(angle_final))
    
    def transform_ellipse_to_frontal(
        self,
        ellipse: np.ndarray,
        iris_ellipse: np.ndarray,
        pupil_ellipse: Optional[np.ndarray],
        frontal_info: dict
    ) -> np.ndarray:
        """
        Transform an ellipse from original image to frontal view coordinates
        
        Parameters:
        -----------
        ellipse : np.ndarray
            Ellipse to transform [cx, cy, a, b, angle]
        iris_ellipse : np.ndarray
            Original iris ellipse (for reference)
        pupil_ellipse : np.ndarray, optional
            Original pupil ellipse (for centering reference)
        frontal_info : dict
            Transformation info from unwrap_to_frontal_view
            
        Returns:
        --------
        transformed_ellipse : np.ndarray
            Ellipse in frontal view coordinates [cx, cy, a, b, angle]
        """
        if ellipse is None or len(ellipse) < 5:
            return None
        if frontal_info is None:
            return None
        
        # Get output_size from frontal_info
        output_size = frontal_info.get('output_size', 700)
        iris_radius_px = frontal_info.get('iris_radius_px', 300)
            
        cx, cy, a, b, angle = ellipse
        iris_cx, iris_cy, iris_a, iris_b, iris_angle = iris_ellipse
        
        # Determine center point used for frontal view
        if pupil_ellipse is not None and len(pupil_ellipse) >= 5:
            center_x, center_y = pupil_ellipse[0:2]
        else:
            center_x, center_y = iris_cx, iris_cy
        
        # Get transformation parameters
        aspect_ratio = frontal_info.get('aspect_ratio', 1.0)
        scale_major = frontal_info.get('scale_major', 1.0)
        scale_minor = frontal_info.get('scale_minor', 1.0)
        target_scale = frontal_info.get('target_scale', 1.0)
        iris_angle_rad = frontal_info.get('iris_angle_rad', 0.0)
        
        # 1. Translate relative to center
        dx = cx - center_x
        dy = cy - center_y
        
        # 2. Apply transformation (same as frontal view)
        cos_a = np.cos(iris_angle_rad)
        sin_a = np.sin(iris_angle_rad)
        
        # Rotate into iris coordinate system
        dx_rot = dx * cos_a + dy * sin_a
        dy_rot = -dx * sin_a + dy * cos_a
        
        # Apply scaling (same as frontal view transformation)
        if iris_a > iris_b:
            dx_scaled = dx_rot * scale_major
            dy_scaled = dy_rot * scale_minor
        else:
            dx_scaled = dx_rot * scale_minor
            dy_scaled = dy_rot * scale_major
        
        # Rotate back
        dx_final = dx_scaled * cos_a - dy_scaled * sin_a
        dy_final = dx_scaled * sin_a + dy_scaled * cos_a
        
        # Position in frontal view (centered at output_size // 2)
        new_cx = output_size // 2 + dx_final
        new_cy = output_size // 2 + dy_final
        
        # 3. Transform ellipse axes
        # Ellipse axes are scaled by the transformation
        rel_angle = angle - iris_angle_rad
        
        if iris_a > iris_b:
            new_a = a * scale_major
            new_b = b * scale_minor
        else:
            new_a = a * scale_minor
            new_b = b * scale_major
        
        # Angle remains relative to iris
        new_angle = rel_angle
        
        return np.array([new_cx, new_cy, new_a, new_b, new_angle])
    
    def process_iris(
        self,
        frame: np.ndarray,
        iris_ellipse: np.ndarray,
        pupil_ellipse: np.ndarray
    ) -> dict:
        """
        Complete iris processing: frontal view + pupil measurements
        
        Parameters:
        -----------
        frame : np.ndarray
            Input grayscale frame
        iris_ellipse : np.ndarray
            Iris ellipse [cx, cy, a, b, angle]
        pupil_ellipse : np.ndarray
            Pupil ellipse [cx, cy, a, b, angle]
            
        Returns:
        --------
        dict with:
            - frontal_view: Perspective-corrected iris image
            - frontal_info: Transformation info
            - pupil_from_ellipse: Pupil measurements from ellipse
            - pupil_from_frontal: Pupil measurements from frontal view
        """
        # Get frontal view (centered on pupil) - using V2 step-by-step approach
        frontal_view, frontal_info = self.unwrap_to_frontal_view_v2(
            frame, iris_ellipse, pupil_ellipse
        )
        
        # Measure pupil from original ellipse (compensated for viewing angle)
        pupil_from_ellipse = self.measure_pupil_from_ellipse(
            pupil_ellipse, iris_ellipse
        )
        
        # Transform ellipses to frontal view coordinates (using V2)
        iris_ellipse_frontal = self.transform_ellipse_to_frontal_v2(
            iris_ellipse, iris_ellipse, pupil_ellipse, frontal_info
        )
        
        pupil_ellipse_frontal = None
        if pupil_ellipse is not None and not np.all(pupil_ellipse == -1):
            pupil_ellipse_frontal = self.transform_ellipse_to_frontal_v2(
                pupil_ellipse, iris_ellipse, pupil_ellipse, frontal_info
            )
        
        # Measure pupil from transformed ellipse in frontal view
        pupil_from_frontal = self.measure_pupil_from_frontal(
            frontal_view, pupil_ellipse_frontal, iris_ellipse
        )
        
        return {
            'frontal_view': frontal_view,
            'frontal_info': frontal_info,
            'pupil_from_ellipse': pupil_from_ellipse,
            'pupil_from_frontal': pupil_from_frontal,
            'iris_ellipse_frontal': iris_ellipse_frontal,
            'pupil_ellipse_frontal': pupil_ellipse_frontal
        }
    
    def visualize_results(
        self,
        original: np.ndarray,
        frontal: np.ndarray,
        pupil_from_ellipse: dict,
        pupil_from_frontal: dict,
        iris_ellipse: np.ndarray,
        pupil_ellipse: Optional[np.ndarray] = None,
        iris_ellipse_frontal: Optional[np.ndarray] = None,
        pupil_ellipse_frontal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create visualization showing original + frontal view + measurements
        
        Returns:
        --------
        visualization : np.ndarray
            Combined visualization image with measurements
        """
        # Convert to BGR for color annotations
        if len(original.shape) == 2:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = original.copy()
        
        # Draw ellipses on original
        if iris_ellipse is not None and len(iris_ellipse) >= 5:
            cx, cy, a, b, angle = iris_ellipse
            cv2.ellipse(original_bgr, (int(cx), int(cy)), (int(a), int(b)),
                       np.rad2deg(angle), 0, 360, (0, 255, 0), 2)  # Green iris
        
        if pupil_ellipse is not None and len(pupil_ellipse) >= 5:
            cx, cy, a, b, angle = pupil_ellipse
            cv2.ellipse(original_bgr, (int(cx), int(cy)), (int(a), int(b)),
                       np.rad2deg(angle), 0, 360, (255, 0, 0), 2)  # Blue pupil
        
        # Add measurements to original
        y_offset = 30
        if pupil_from_ellipse:
            text = f"Pupil (ellipse): {pupil_from_ellipse['pupil_diameter_mm']:.2f}mm"
            cv2.putText(original_bgr, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
        
        # Frontal view (convert to BGR)
        h_orig = original_bgr.shape[0]
        
        if frontal is not None:
            frontal_bgr = cv2.cvtColor(frontal, cv2.COLOR_GRAY2BGR) if len(frontal.shape) == 2 else frontal
            frontal_resized = cv2.resize(frontal_bgr, (h_orig, h_orig))
            
            # Draw transformed ellipses on frontal view
            scale_factor = h_orig / frontal.shape[0]
            
            if iris_ellipse_frontal is not None and len(iris_ellipse_frontal) >= 5:
                cx, cy, a, b, angle = iris_ellipse_frontal
                cv2.ellipse(frontal_resized, 
                           (int(cx * scale_factor), int(cy * scale_factor)), 
                           (int(a * scale_factor), int(b * scale_factor)),
                           np.rad2deg(angle), 0, 360, (0, 255, 0), 2)  # Green iris
            
            if pupil_ellipse_frontal is not None and len(pupil_ellipse_frontal) >= 5:
                cx, cy, a, b, angle = pupil_ellipse_frontal
                cv2.ellipse(frontal_resized, 
                           (int(cx * scale_factor), int(cy * scale_factor)), 
                           (int(a * scale_factor), int(b * scale_factor)),
                           np.rad2deg(angle), 0, 360, (255, 0, 0), 2)  # Blue pupil
            
            # Add measurements to frontal
            cv2.putText(frontal_resized, "Frontal View", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if pupil_from_frontal:
                # Main measurement: diameter in mm
                text = f"Pupil: {pupil_from_frontal['pupil_diameter_mm']:.2f}mm"
                cv2.putText(frontal_resized, text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Additional info: area and equivalent diameter
                if 'pupil_area_px' in pupil_from_frontal:
                    text_area = f"Area: {pupil_from_frontal['pupil_area_px']:.0f}px"
                    cv2.putText(frontal_resized, text_area, (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    text_diam = f"D_eq: {pupil_from_frontal['pupil_diameter_from_area_px']:.1f}px"
                    cv2.putText(frontal_resized, text_diam, (10, 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            frontal_resized = np.zeros((h_orig, h_orig, 3), dtype=np.uint8)
        
        # Horizontal stack
        combined = np.hstack([original_bgr, frontal_resized])
        
        return combined


# Test function
def test_iris_processing():
    """Test iris processing with pupil measurements"""
    import matplotlib.pyplot as plt
    
    # Create synthetic eye image
    frame = np.zeros((400, 400), dtype=np.uint8)
    
    # Draw iris with texture
    cv2.circle(frame, (200, 200), 100, 150, -1)
    cv2.circle(frame, (200, 200), 40, 50, -1)  # pupil
    
    # Add some texture
    for i in range(12):
        angle = i * 30
        x1 = int(200 + 40 * np.cos(np.deg2rad(angle)))
        y1 = int(200 + 40 * np.sin(np.deg2rad(angle)))
        x2 = int(200 + 100 * np.cos(np.deg2rad(angle)))
        y2 = int(200 + 100 * np.sin(np.deg2rad(angle)))
        cv2.line(frame, (x1, y1), (x2, y2), 180, 1)
    
    # Simulate viewing angle (make elliptical)
    M = cv2.getRotationMatrix2D((200, 200), 30, 1.0)
    M[0, 0] = 1.0  # No scale X
    M[1, 1] = 0.6  # Compress Y (viewing angle)
    frame = cv2.warpAffine(frame, M, (400, 400))
    
    # Define ellipses
    iris_ellipse = np.array([200, 200, 100, 60, np.deg2rad(30)])
    pupil_ellipse = np.array([200, 200, 40, 24, np.deg2rad(30)])
    
    # Process iris (without EllSeg for simple test)
    unwrapper = IrisUnwrapper()
    results = unwrapper.process_iris(frame, iris_ellipse, pupil_ellipse)
    
    # Visualize
    viz = unwrapper.visualize_results(
        frame, 
        results['frontal_view'],
        results['pupil_from_ellipse'],
        results['pupil_from_frontal'],
        iris_ellipse,
        pupil_ellipse,
        results.get('iris_ellipse_frontal'),
        results.get('pupil_ellipse_frontal')
    )
    
    # Display
    plt.figure(figsize=(12, 6))
    plt.imshow(viz, cmap='gray')
    plt.axis('off')
    plt.title('Iris Processing: Original | Frontal View + Pupil Measurements')
    plt.tight_layout()
    plt.savefig('iris_processing_test.png', dpi=150, bbox_inches='tight')
    print("✅ Test visualization saved: iris_processing_test.png")
    
    print(f"\n📊 Frontal Info: {results['frontal_info']}")
    print(f"📊 Pupil from Ellipse: {results['pupil_from_ellipse']}")
    print(f"📊 Pupil from Frontal: {results['pupil_from_frontal']}")


if __name__ == "__main__":
    test_iris_processing()
