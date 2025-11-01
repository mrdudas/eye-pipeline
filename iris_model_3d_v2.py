#!/usr/bin/env python3
"""
3D Eye Model - Physical Model with Spherical Eyeball

This implements a physically accurate eye model where:
1. The eyeball is a sphere (radius ~12mm)
2. The iris is a circular disk on the spherical surface
3. The pupil is a smaller concentric circle
4. Both project to ellipses on the 2D image

The model parameters:
- Eye center in 3D camera coordinates: (x, y, z)
- Gaze direction: (theta, phi) - direction the eye is looking
- Pupil radius: r_pupil (in mm on the eye)
- Iris radius: r_iris (in mm on the eye)
"""

import cv2
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class EyeballModel3D:
    """
    Physical 3D model of the eyeball with iris and pupil.
    
    The eyeball is modeled as a sphere, with the iris as a circular disk
    on the front surface. When viewed from different angles, these circles
    project to ellipses on the camera image.
    """
    
    def __init__(self, img_width: int, img_height: int, 
                 camera_matrix: Optional[np.ndarray] = None):
        """
        Initialize the 3D eyeball model.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            camera_matrix: 3x3 camera intrinsic matrix (optional)
        """
        self.img_width = img_width
        self.img_height = img_height
        
        # Camera intrinsic parameters
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
            self.fx = camera_matrix[0, 0]
            self.fy = camera_matrix[1, 1]
            self.cx = camera_matrix[0, 2]
            self.cy = camera_matrix[1, 2]
        else:
            # Default: center of image, focal length = image width
            self.fx = img_width
            self.fy = img_width
            self.cx = img_width / 2
            self.cy = img_height / 2
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
        
        # Physical eye parameters (approximate, will be scaled to pixels)
        self.eyeball_radius_mm = 12.0  # Typical human eyeball radius
        self.typical_iris_radius_mm = 5.8  # Half of typical iris diameter (11.6mm)
        self.typical_pupil_radius_mm = 2.0  # Typical pupil radius (varies 2-4mm)
    
    def gaze_to_rotation_matrix(self, theta: float, phi: float, roll: float = 0.0) -> np.ndarray:
        """
        Convert gaze direction to rotation matrix.
        
        Args:
            theta: Vertical gaze angle (pitch, radians) - looking up/down
            phi: Horizontal gaze angle (yaw, radians) - looking left/right  
            roll: Roll angle (radians) - head tilt
        
        Returns:
            3x3 rotation matrix
        """
        # Rotation around X-axis (pitch - up/down)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        # Rotation around Y-axis (yaw - left/right)
        Ry = np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])
        
        # Rotation around Z-axis (roll - head tilt)
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz @ Ry @ Rx
        return Rz @ Ry @ Rx
    
    def generate_circle_on_sphere(self, radius: float, eyeball_center: np.ndarray,
                                 gaze_direction: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        Generate 3D points forming a circle on the spherical eyeball surface.
        
        Args:
            radius: Circle radius (in the same units as eyeball_radius)
            eyeball_center: 3D position of eyeball center
            gaze_direction: 3D unit vector pointing in gaze direction
            num_points: Number of points on the circle
        
        Returns:
            (num_points, 3) array of 3D points in camera coordinates
        """
        # Create circle in XY plane
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        circle_2d = np.zeros((num_points, 2))
        circle_2d[:, 0] = radius * np.cos(angles)
        circle_2d[:, 1] = radius * np.sin(angles)
        
        # The circle lies on a plane perpendicular to gaze direction
        # Find rotation matrix to align Z-axis with gaze direction
        z_axis = np.array([0, 0, 1])
        gaze_norm = gaze_direction / np.linalg.norm(gaze_direction)
        
        # Rotation axis: cross product of z_axis and gaze
        rotation_axis = np.cross(z_axis, gaze_norm)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            # Gaze is aligned with Z-axis
            R = np.eye(3) if np.dot(z_axis, gaze_norm) > 0 else np.diag([1, 1, -1])
        else:
            # Rodrigues rotation formula
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, gaze_norm), -1, 1))
            
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
        
        # Apply rotation to circle points (add z=0 coordinate)
        circle_3d_local = np.column_stack([circle_2d, np.zeros(num_points)])
        circle_3d_rotated = (R @ circle_3d_local.T).T
        
        # Translate to eyeball surface (distance = eyeball_radius from center)
        # The circle is at distance eyeball_radius in the gaze direction
        circle_3d = circle_3d_rotated + eyeball_center + self.eyeball_radius_mm * gaze_norm
        
        return circle_3d
    
    def project_points_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image using perspective projection.
        
        Args:
            points_3d: (N, 3) array of 3D points in camera coordinates
        
        Returns:
            (N, 2) array of 2D image points (some may be NaN if behind camera)
        """
        points_2d = np.zeros((len(points_3d), 2))
        
        for i, (x, y, z) in enumerate(points_3d):
            if z > 0.1:  # Point is in front of camera
                points_2d[i, 0] = self.fx * (x / z) + self.cx
                points_2d[i, 1] = self.fy * (y / z) + self.cy
            else:
                points_2d[i] = [np.nan, np.nan]
        
        return points_2d
    
    def fit_ellipse_to_points(self, points_2d: np.ndarray) -> Optional[Tuple]:
        """
        Fit an ellipse to 2D points using OpenCV.
        
        Args:
            points_2d: (N, 2) array of 2D points
        
        Returns:
            ((cx, cy), (major_axis, minor_axis), angle) or None if fitting fails
        """
        # Remove NaN points
        valid_mask = ~np.isnan(points_2d).any(axis=1)
        valid_points = points_2d[valid_mask]
        
        if len(valid_points) < 5:
            return None
        
        try:
            # OpenCV fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(valid_points.astype(np.float32))
            return ellipse
        except:
            return None
    
    def render_mask_from_parameters(self, eye_center: np.ndarray, theta: float, phi: float,
                                   r_pupil: float, r_iris: float, 
                                   scale_factor: float = 1.0) -> np.ndarray:
        """
        Render segmentation mask from 3D model parameters.
        
        Args:
            eye_center: 3D position of eyeball center [x, y, z]
            theta: Vertical gaze angle (radians)
            phi: Horizontal gaze angle (radians)
            r_pupil: Pupil radius (in pixels, will be scaled to mm)
            r_iris: Iris radius (in pixels, will be scaled to mm)
            scale_factor: Scaling factor from pixels to mm
        
        Returns:
            mask: (height, width) segmentation mask (0=bg, 2=iris, 3=pupil)
        """
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        # Convert gaze angles to direction vector
        gaze_direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(theta),
            np.cos(phi) * np.cos(theta)
        ])
        
        # Generate iris and pupil circles in 3D
        iris_3d = self.generate_circle_on_sphere(
            r_iris * scale_factor, eye_center, gaze_direction, num_points=100
        )
        pupil_3d = self.generate_circle_on_sphere(
            r_pupil * scale_factor, eye_center, gaze_direction, num_points=100
        )
        
        # Project to 2D
        iris_2d = self.project_points_to_image(iris_3d)
        pupil_2d = self.project_points_to_image(pupil_3d)
        
        # Remove NaN points
        iris_2d = iris_2d[~np.isnan(iris_2d).any(axis=1)]
        pupil_2d = pupil_2d[~np.isnan(pupil_2d).any(axis=1)]
        
        if len(iris_2d) < 3:
            return mask
        
        # Fill iris
        iris_contour = iris_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [iris_contour], 2)
        
        if len(pupil_2d) >= 3:
            # Fill pupil
            pupil_contour = pupil_2d.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pupil_contour], 3)
        
        return mask
    
    def fit_to_mask(self, ritnet_mask: np.ndarray, 
                   initial_guess: Optional[Dict] = None,
                   method: str = 'nelder-mead') -> Dict:
        """
        Fit 3D eyeball model to RITnet segmentation mask.
        
        Args:
            ritnet_mask: (H, W) segmentation mask (0=bg, 1=sclera, 2=iris, 3=pupil)
            initial_guess: Optional initial parameters
            method: Optimization method
        
        Returns:
            Dictionary with fitted parameters
        """
        # Extract initial guess from mask
        if initial_guess is None:
            initial_guess = self._estimate_initial_params(ritnet_mask)
        
        # Estimate scale factor (pixels to mm)
        # Assume typical iris is ~60-80 pixels in this image
        typical_iris_pixels = initial_guess['r_iris']
        scale_factor = self.typical_iris_radius_mm / max(typical_iris_pixels, 1)
        
        # Parameter bounds
        # [x, y, z, theta, phi, r_pupil, r_iris]
        bounds = [
            (-self.img_width, self.img_width),     # x: eye center X
            (-self.img_height, self.img_height),   # y: eye center Y  
            (50, 500),                              # z: distance to eye (mm-ish)
            (-np.pi/4, np.pi/4),                   # theta: pitch ±45°
            (-np.pi/4, np.pi/4),                   # phi: yaw ±45°
            (10, 80),                               # r_pupil: pixels
            (40, 150),                              # r_iris: pixels
        ]
        
        # Initial parameters
        x0 = np.array([
            initial_guess['cx'] - self.cx,  # Relative to image center
            initial_guess['cy'] - self.cy,
            200.0,  # Initial distance estimate
            0.0,    # Initial theta
            0.0,    # Initial phi
            initial_guess['r_pupil'],
            initial_guess['r_iris']
        ])
        
        # Loss function
        def loss_fn(params):
            x, y, z, theta, phi, r_pupil, r_iris = params
            
            # Constraints
            if r_pupil >= r_iris * 0.9:
                return 1e6
            if r_pupil < 5 or r_iris < 20:
                return 1e6
            
            eye_center = np.array([x, y, z])
            
            try:
                synthetic_mask = self.render_mask_from_parameters(
                    eye_center, theta, phi, r_pupil, r_iris, scale_factor
                )
                
                # Compute IoU loss
                loss = self._compute_loss(ritnet_mask, synthetic_mask)
                return loss
            except:
                return 1e6
        
        # Optimize
        if method == 'de':
            result = differential_evolution(loss_fn, bounds, maxiter=30, seed=42,
                                          workers=1, disp=False, polish=True)
        else:
            result = minimize(loss_fn, x0, method=method, bounds=bounds,
                            options={'maxiter': 500, 'disp': False})
        
        # Extract results
        x, y, z, theta, phi, r_pupil, r_iris = result.x
        eye_center = np.array([x, y, z])
        
        # Compute final metrics
        final_mask = self.render_mask_from_parameters(
            eye_center, theta, phi, r_pupil, r_iris, scale_factor
        )
        
        iou_pupil = self._compute_iou(ritnet_mask == 3, final_mask == 3)
        iou_iris = self._compute_iou(ritnet_mask == 2, final_mask == 2)
        
        # Get ellipse parameters
        pupil_ellipse = self._get_ellipse_from_mask(final_mask == 3)
        iris_ellipse = self._get_ellipse_from_mask(final_mask == 2)
        
        return {
            'eye_center': eye_center,
            'x': x,
            'y': y,
            'z': z,
            'theta': theta,
            'phi': phi,
            'theta_deg': np.degrees(theta),
            'phi_deg': np.degrees(phi),
            'r_pupil': r_pupil,
            'r_iris': r_iris,
            'scale_factor': scale_factor,
            'pupil_ellipse': pupil_ellipse,
            'iris_ellipse': iris_ellipse,
            'loss': result.fun,
            'iou_pupil': iou_pupil,
            'iou_iris': iou_iris,
            'cx_image': x + self.cx,
            'cy_image': y + self.cy
        }
    
    def _estimate_initial_params(self, mask: np.ndarray) -> Dict:
        """Estimate initial parameters from mask."""
        pupil_mask = (mask == 3).astype(np.uint8)
        iris_mask = (mask == 2).astype(np.uint8)
        
        if pupil_mask.sum() > 0:
            M = cv2.moments(pupil_mask)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                r_pupil = np.sqrt(pupil_mask.sum() / np.pi)
            else:
                cx, cy, r_pupil = self.img_width/2, self.img_height/2, 20
        else:
            cx, cy, r_pupil = self.img_width/2, self.img_height/2, 20
        
        if iris_mask.sum() > 0:
            r_iris = np.sqrt(iris_mask.sum() / np.pi)
        else:
            r_iris = r_pupil * 3
        
        return {'cx': cx, 'cy': cy, 'r_pupil': r_pupil, 'r_iris': r_iris}
    
    def _get_ellipse_from_mask(self, binary_mask: np.ndarray) -> Optional[Tuple]:
        """Fit ellipse to binary mask."""
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return None
        
        try:
            return cv2.fitEllipse(largest)
        except:
            return None
    
    def _compute_loss(self, mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
        """Compute 1 - mean IoU."""
        iou_pupil = self._compute_iou(mask_true == 3, mask_pred == 3)
        iou_iris = self._compute_iou(mask_true >= 2, mask_pred >= 2)
        mean_iou = (iou_pupil + iou_iris) / 2
        return 1.0 - mean_iou
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0


def demo():
    """Demo with synthetic eye."""
    import matplotlib
    matplotlib.use('Agg')
    
    img_size = 400
    model = EyeballModel3D(img_size, img_size)
    
    print("Generating synthetic eye...")
    
    # Ground truth parameters
    eye_center = np.array([0, 0, 200])  # 200mm from camera
    theta = np.radians(20)  # Looking up 20°
    phi = np.radians(-15)   # Looking left 15°
    r_pupil = 35
    r_iris = 100
    
    # Generate mask
    mask = model.render_mask_from_parameters(
        eye_center, theta, phi, r_pupil, r_iris, scale_factor=0.058
    )
    
    print("Fitting 3D model...")
    params = model.fit_to_mask(mask, method='de')
    
    print("\n=== Results ===")
    print(f"Gaze direction: θ={params['theta_deg']:.1f}° φ={params['phi_deg']:.1f}°")
    print(f"Eye position: x={params['x']:.1f}, y={params['y']:.1f}, z={params['z']:.1f}mm")
    print(f"Radii: pupil={params['r_pupil']:.1f}px, iris={params['r_iris']:.1f}px")
    print(f"IoU: Pupil={params['iou_pupil']:.3f}, Iris={params['iou_iris']:.3f}")
    
    if params['pupil_ellipse']:
        (cx, cy), (ma, mi), angle = params['pupil_ellipse']
        print(f"Pupil ellipse: center=({cx:.1f},{cy:.1f}), axes=({ma:.1f},{mi:.1f}), angle={angle:.1f}°")
    
    print("\n✅ Model fitted successfully!")


if __name__ == '__main__':
    demo()
