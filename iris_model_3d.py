#!/usr/bin/env python3
"""
3D Iris-Pupil Model and Fitting

This module implements a 3D geometric model of the iris and pupil as two concentric circles
viewed from arbitrary angles. It fits this model to RITnet segmentation masks to estimate
the 3D orientation and position of the eye.

Model:
- Pupil: Inner circle with radius r_pupil
- Iris: Outer circle with radius r_iris
- Both circles lie in the same plane (eye plane)
- Eye plane can be rotated by angles (theta, phi)
- Center position: (cx, cy) in image coordinates

Coordinate Systems:
1. Eye coordinates: 3D space where iris is a circle in XY plane
2. Camera coordinates: After rotation (theta, phi)
3. Image coordinates: After perspective projection
"""

import cv2
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class IrisPupilModel3D:
    """
    3D Model of iris and pupil as two concentric circles in 3D space.
    
    The model represents:
    - Pupil: inner circle (radius r_pupil)
    - Iris: outer circle (radius r_iris)
    - Both in the same plane at distance d from camera
    - Plane orientation: (theta, phi) rotation angles
    - Center position: (cx, cy) translation
    """
    
    def __init__(self, img_width: int, img_height: int, 
                 camera_matrix: Optional[np.ndarray] = None):
        """
        Initialize the 3D iris-pupil model.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            camera_matrix: 3x3 camera intrinsic matrix (optional)
                          If None, assumes simple pinhole camera
        """
        self.img_width = img_width
        self.img_height = img_height
        
        # Camera intrinsic parameters
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
            self.fx = camera_matrix[0, 0]
            self.fy = camera_matrix[1, 1]
            self.cx_cam = camera_matrix[0, 2]
            self.cy_cam = camera_matrix[1, 2]
        else:
            # Default: center of image, focal length = image width
            self.fx = img_width
            self.fy = img_width
            self.cx_cam = img_width / 2
            self.cy_cam = img_height / 2
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx_cam],
                [0, self.fy, self.cy_cam],
                [0, 0, 1]
            ])
    
    def rotation_matrix(self, theta: float, phi: float) -> np.ndarray:
        """
        Compute 3D rotation matrix from angles.
        
        Args:
            theta: Rotation around X-axis (pitch, looking up/down)
            phi: Rotation around Y-axis (yaw, looking left/right)
        
        Returns:
            3x3 rotation matrix
        """
        # Rotation around X-axis (pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        # Rotation around Y-axis (yaw)
        Ry = np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])
        
        # Combined rotation: first yaw, then pitch
        return Rx @ Ry
    
    def generate_circle_points(self, radius: float, num_points: int = 100) -> np.ndarray:
        """
        Generate 3D points on a circle in the XY plane.
        
        Args:
            radius: Circle radius
            num_points: Number of points to generate
        
        Returns:
            (num_points, 3) array of 3D points
        """
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=1)
    
    def project_to_image(self, points_3d: np.ndarray, theta: float, phi: float,
                        cx: float, cy: float, distance: float) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: (N, 3) array of 3D points in eye coordinates
            theta: Pitch rotation (radians)
            phi: Yaw rotation (radians)
            cx: Center X position in image (pixels)
            cy: Center Y position in image (pixels)
            distance: Distance from camera to eye plane (arbitrary units)
        
        Returns:
            (N, 2) array of 2D image points
        """
        # Rotate points
        R = self.rotation_matrix(theta, phi)
        points_rotated = (R @ points_3d.T).T  # (N, 3)
        
        # Translate to distance d
        points_camera = points_rotated + np.array([0, 0, distance])
        
        # Perspective projection
        points_2d = np.zeros((len(points_3d), 2))
        for i, (x, y, z) in enumerate(points_camera):
            if z > 0.1:  # Avoid division by zero
                points_2d[i, 0] = self.fx * (x / z) + self.cx_cam
                points_2d[i, 1] = self.fy * (y / z) + self.cy_cam
            else:
                points_2d[i] = [np.nan, np.nan]
        
        # Apply center translation
        points_2d[:, 0] += (cx - self.cx_cam)
        points_2d[:, 1] += (cy - self.cy_cam)
        
        return points_2d
    
    def render_mask(self, r_pupil: float, r_iris: float, theta: float, phi: float,
                   cx: float, cy: float, distance: float = 100.0) -> np.ndarray:
        """
        Render a synthetic segmentation mask from model parameters.
        
        Args:
            r_pupil: Pupil radius (pixels)
            r_iris: Iris radius (pixels)
            theta: Pitch rotation (radians)
            phi: Yaw rotation (radians)
            cx: Center X (pixels)
            cy: Center Y (pixels)
            distance: Distance to eye plane (default 100)
        
        Returns:
            mask: (height, width) uint8 array with values:
                  0 = background
                  1 = sclera (not modeled, always 0)
                  2 = iris
                  3 = pupil
        """
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        # Generate circles
        pupil_3d = self.generate_circle_points(r_pupil, num_points=100)
        iris_3d = self.generate_circle_points(r_iris, num_points=100)
        
        # Project to 2D
        pupil_2d = self.project_to_image(pupil_3d, theta, phi, cx, cy, distance)
        iris_2d = self.project_to_image(iris_3d, theta, phi, cx, cy, distance)
        
        # Remove NaN points
        pupil_2d = pupil_2d[~np.isnan(pupil_2d).any(axis=1)]
        iris_2d = iris_2d[~np.isnan(iris_2d).any(axis=1)]
        
        if len(iris_2d) < 3 or len(pupil_2d) < 3:
            return mask
        
        # Fill iris
        iris_contour = iris_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [iris_contour], 2)
        
        # Fill pupil (overwrites iris)
        pupil_contour = pupil_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pupil_contour], 3)
        
        return mask
    
    def fit_to_mask(self, ritnet_mask: np.ndarray, 
                   initial_guess: Optional[Dict] = None,
                   method: str = 'nelder-mead') -> Dict:
        """
        Fit 3D model to RITnet segmentation mask.
        
        Args:
            ritnet_mask: (H, W) segmentation mask with values 0-3
                        (0=background, 1=sclera, 2=iris, 3=pupil)
            initial_guess: Optional dict with keys: r_pupil, r_iris, theta, phi, cx, cy
            method: Optimization method ('nelder-mead', 'powell', 'de' for differential evolution)
        
        Returns:
            Dictionary with fitted parameters:
                - r_pupil: Pupil radius
                - r_iris: Iris radius
                - theta: Pitch angle (radians)
                - phi: Yaw angle (radians)
                - cx: Center X
                - cy: Center Y
                - distance: Fixed at 100
                - loss: Final loss value
                - iou_pupil: IoU for pupil
                - iou_iris: IoU for iris
        """
        # Extract initial guess from mask if not provided
        if initial_guess is None:
            initial_guess = self._estimate_initial_params(ritnet_mask)
        
        # Parameter bounds
        bounds = [
            (5, self.img_width / 3),      # r_pupil
            (10, self.img_width / 2),     # r_iris
            (-np.pi/3, np.pi/3),          # theta (pitch)
            (-np.pi/3, np.pi/3),          # phi (yaw)
            (0, self.img_width),          # cx
            (0, self.img_height),         # cy
        ]
        
        # Initial parameters vector
        x0 = np.array([
            initial_guess['r_pupil'],
            initial_guess['r_iris'],
            initial_guess.get('theta', 0.0),
            initial_guess.get('phi', 0.0),
            initial_guess['cx'],
            initial_guess['cy']
        ])
        
        # Loss function
        def loss_fn(params):
            r_pupil, r_iris, theta, phi, cx, cy = params
            
            # Constraints
            if r_pupil >= r_iris:
                return 1e6  # Pupil must be smaller than iris
            if r_pupil < 5 or r_iris < 10:
                return 1e6
            
            # Render synthetic mask
            synthetic_mask = self.render_mask(r_pupil, r_iris, theta, phi, cx, cy)
            
            # Compute loss (1 - IoU)
            loss = self._compute_loss(ritnet_mask, synthetic_mask)
            return loss
        
        # Optimize
        if method == 'de':
            # Differential evolution (global optimization)
            result = differential_evolution(loss_fn, bounds, maxiter=30, seed=42,
                                          workers=1, disp=True, polish=False)
        else:
            # Local optimization
            result = minimize(loss_fn, x0, method=method, bounds=bounds,
                            options={'maxiter': 500, 'disp': False})
        
        # Extract fitted parameters
        r_pupil, r_iris, theta, phi, cx, cy = result.x
        
        # Compute final metrics
        final_mask = self.render_mask(r_pupil, r_iris, theta, phi, cx, cy)
        iou_pupil = self._compute_iou(ritnet_mask == 3, final_mask == 3)
        iou_iris = self._compute_iou(ritnet_mask == 2, final_mask == 2)
        
        return {
            'r_pupil': r_pupil,
            'r_iris': r_iris,
            'theta': theta,
            'phi': phi,
            'cx': cx,
            'cy': cy,
            'distance': 100.0,
            'loss': result.fun,
            'iou_pupil': iou_pupil,
            'iou_iris': iou_iris,
            'theta_deg': np.degrees(theta),
            'phi_deg': np.degrees(phi)
        }
    
    def _estimate_initial_params(self, mask: np.ndarray) -> Dict:
        """
        Estimate initial parameters from mask.
        
        Args:
            mask: RITnet segmentation mask
        
        Returns:
            Dictionary with initial parameter estimates
        """
        # Find pupil region
        pupil_mask = (mask == 3).astype(np.uint8)
        iris_mask = (mask == 2).astype(np.uint8)
        
        # Pupil center and radius
        if pupil_mask.sum() > 0:
            M_pupil = cv2.moments(pupil_mask)
            if M_pupil['m00'] > 0:
                cx_pupil = M_pupil['m10'] / M_pupil['m00']
                cy_pupil = M_pupil['m01'] / M_pupil['m00']
                r_pupil = np.sqrt(pupil_mask.sum() / np.pi)
            else:
                cx_pupil = self.img_width / 2
                cy_pupil = self.img_height / 2
                r_pupil = 20
        else:
            cx_pupil = self.img_width / 2
            cy_pupil = self.img_height / 2
            r_pupil = 20
        
        # Iris radius
        if iris_mask.sum() > 0:
            r_iris = np.sqrt(iris_mask.sum() / np.pi)
        else:
            r_iris = r_pupil * 3  # Default ratio
        
        return {
            'r_pupil': r_pupil,
            'r_iris': r_iris,
            'theta': 0.0,
            'phi': 0.0,
            'cx': cx_pupil,
            'cy': cy_pupil
        }
    
    def _compute_loss(self, mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
        """
        Compute loss between true and predicted masks.
        
        Uses 1 - mean IoU for iris and pupil regions.
        """
        # Pupil IoU
        iou_pupil = self._compute_iou(mask_true == 3, mask_pred == 3)
        
        # Iris IoU (including pupil as part of iris)
        iou_iris = self._compute_iou(mask_true >= 2, mask_pred >= 2)
        
        # Combined loss (maximize average IoU)
        mean_iou = (iou_pupil + iou_iris) / 2
        return 1.0 - mean_iou
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union
    
    def unwrap_iris(self, image: np.ndarray, params: Dict,
                   output_size: Tuple[int, int] = (256, 64)) -> np.ndarray:
        """
        Unwrap iris to frontal view (polar coordinates).
        
        This creates a "normalized" iris image as if viewing the iris head-on,
        removing perspective distortion.
        
        Args:
            image: Original image
            params: Model parameters from fit_to_mask()
            output_size: (width, height) of output unwrapped iris
                        width = angular resolution (360 degrees)
                        height = radial resolution (pupil to iris boundary)
        
        Returns:
            Unwrapped iris image (frontal view)
        """
        r_pupil = params['r_pupil']
        r_iris = params['r_iris']
        theta = params['theta']
        phi = params['phi']
        cx = params['cx']
        cy = params['cy']
        
        width, height = output_size
        unwrapped = np.zeros((height, width, 3), dtype=np.uint8)
        
        # For each point in output (polar coordinates)
        for v in range(height):
            # Radius: from pupil boundary to iris boundary
            radius_normalized = v / height  # 0 to 1
            radius = r_pupil + radius_normalized * (r_iris - r_pupil)
            
            for u in range(width):
                # Angle: 0 to 2*pi
                angle = 2 * np.pi * u / width
                
                # 3D point on iris plane (before rotation)
                x_3d = radius * np.cos(angle)
                y_3d = radius * np.sin(angle)
                z_3d = 0.0
                
                # Apply rotation
                point_3d = np.array([x_3d, y_3d, z_3d])
                R = self.rotation_matrix(theta, phi)
                point_rotated = R @ point_3d
                
                # Project to image
                point_camera = point_rotated + np.array([0, 0, 100.0])
                x_cam, y_cam, z_cam = point_camera
                
                if z_cam > 0.1:
                    x_img = self.fx * (x_cam / z_cam) + self.cx_cam + (cx - self.cx_cam)
                    y_img = self.fy * (y_cam / z_cam) + self.cy_cam + (cy - self.cy_cam)
                    
                    # Sample from original image
                    x_img_int = int(round(x_img))
                    y_img_int = int(round(y_img))
                    
                    if 0 <= x_img_int < image.shape[1] and 0 <= y_img_int < image.shape[0]:
                        unwrapped[v, u] = image[y_img_int, x_img_int]
        
        return unwrapped
    
    def visualize_fit(self, image: np.ndarray, ritnet_mask: np.ndarray, 
                     params: Dict, output_path: Optional[str] = None):
        """
        Visualize model fitting results.
        
        Args:
            image: Original image
            ritnet_mask: RITnet segmentation mask
            params: Fitted model parameters
            output_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # RITnet mask
        colored_ritnet = self._colorize_mask(ritnet_mask)
        axes[1].imshow(colored_ritnet)
        axes[1].set_title('RITnet Mask')
        axes[1].axis('off')
        
        # Fitted model mask
        fitted_mask = self.render_mask(
            params['r_pupil'], params['r_iris'],
            params['theta'], params['phi'],
            params['cx'], params['cy']
        )
        colored_fitted = self._colorize_mask(fitted_mask)
        axes[2].imshow(colored_fitted)
        axes[2].set_title(f"3D Model\nθ={params['theta_deg']:.1f}° φ={params['phi_deg']:.1f}°")
        axes[2].axis('off')
        
        # Overlay
        overlay = image.copy()
        # Draw iris boundary
        iris_3d = self.generate_circle_points(params['r_iris'], num_points=100)
        iris_2d = self.project_to_image(iris_3d, params['theta'], params['phi'],
                                        params['cx'], params['cy'], 100.0)
        iris_2d = iris_2d[~np.isnan(iris_2d).any(axis=1)]
        iris_contour = iris_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [iris_contour], True, (0, 255, 0), 2)
        
        # Draw pupil boundary
        pupil_3d = self.generate_circle_points(params['r_pupil'], num_points=100)
        pupil_2d = self.project_to_image(pupil_3d, params['theta'], params['phi'],
                                         params['cx'], params['cy'], 100.0)
        pupil_2d = pupil_2d[~np.isnan(pupil_2d).any(axis=1)]
        pupil_contour = pupil_2d.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pupil_contour], True, (0, 0, 255), 2)
        
        # Draw center
        cv2.circle(overlay, (int(params['cx']), int(params['cy'])), 5, (255, 0, 0), -1)
        
        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f"IoU: Pupil={params['iou_pupil']:.3f} Iris={params['iou_iris']:.3f}")
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to RGB color image."""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[mask == 1] = [255, 0, 0]    # Sclera: Red
        colored[mask == 2] = [0, 255, 0]    # Iris: Green
        colored[mask == 3] = [0, 0, 255]    # Pupil: Blue
        return colored


def demo():
    """Demo of iris model fitting."""
    import sys
    import os
    
    # Prevent matplotlib from using GUI backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create synthetic mask
    img_size = 400
    model = IrisPupilModel3D(img_size, img_size)
    
    # Generate ground truth mask
    print("Generating synthetic data...")
    true_params = {
        'r_pupil': 40,
        'r_iris': 120,
        'theta': np.radians(15),  # 15 degrees pitch
        'phi': np.radians(-10),   # -10 degrees yaw
        'cx': 200,
        'cy': 200
    }
    
    synthetic_mask = model.render_mask(
        true_params['r_pupil'], true_params['r_iris'],
        true_params['theta'], true_params['phi'],
        true_params['cx'], true_params['cy']
    )
    
    # Fit model
    print("Fitting 3D model...")
    fitted_params = model.fit_to_mask(synthetic_mask, method='de')
    
    print("\n=== Ground Truth ===")
    print(f"r_pupil: {true_params['r_pupil']:.2f} px")
    print(f"r_iris: {true_params['r_iris']:.2f} px")
    print(f"theta: {np.degrees(true_params['theta']):.2f}°")
    print(f"phi: {np.degrees(true_params['phi']):.2f}°")
    
    print("\n=== Fitted Parameters ===")
    print(f"r_pupil: {fitted_params['r_pupil']:.2f} px")
    print(f"r_iris: {fitted_params['r_iris']:.2f} px")
    print(f"theta: {fitted_params['theta_deg']:.2f}°")
    print(f"phi: {fitted_params['phi_deg']:.2f}°")
    print(f"IoU Pupil: {fitted_params['iou_pupil']:.3f}")
    print(f"IoU Iris: {fitted_params['iou_iris']:.3f}")
    
    # Visualize
    dummy_image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    model.visualize_fit(dummy_image, synthetic_mask, fitted_params, 
                       output_path='iris_model_fit_demo.png')
    print("\n✅ Visualization saved to: iris_model_fit_demo.png")


if __name__ == '__main__':
    demo()
