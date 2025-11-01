#!/usr/bin/env python3
"""
Ellipse-Based Iris-Pupil Model

This model directly fits ellipses to the iris and pupil regions in the 2D image.
The ellipses represent the projection of circular iris and pupil when viewed at an angle.

Model parameters:
- Pupil: Ellipse with center (cx_p, cy_p), axes (a_p, b_p), angle θ_p
- Iris: Ellipse with center (cx_i, cy_i), axes (a_i, b_i), angle θ_i

Constraints:
- Both ellipses should be approximately concentric (same center)
- Both ellipses should have similar orientation (same angle)
- Pupil ellipse should be inside iris ellipse
"""

import cv2
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class EllipseIrisPupilModel:
    """
    Direct ellipse-based model for iris and pupil.
    
    This fits two concentric, similarly-oriented ellipses to the RITnet mask.
    Much simpler and faster than full 3D projection.
    """
    
    def __init__(self, img_width: int, img_height: int):
        """Initialize the ellipse model."""
        self.img_width = img_width
        self.img_height = img_height
    
    def render_ellipse_mask(self, cx: float, cy: float,
                          pupil_a: float, pupil_b: float, pupil_angle: float,
                          iris_a: float, iris_b: float, iris_angle: float) -> np.ndarray:
        """
        Render a mask with two ellipses.
        
        Args:
            cx, cy: Common center for both ellipses
            pupil_a, pupil_b: Pupil major/minor axes
            pupil_angle: Pupil rotation angle (degrees)
            iris_a, iris_b: Iris major/minor axes
            iris_angle: Iris rotation angle (degrees)
        
        Returns:
            mask: (H, W) array with 0=bg, 2=iris, 3=pupil
        """
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        # Draw iris
        cv2.ellipse(mask, 
                   (int(cx), int(cy)),
                   (int(iris_a), int(iris_b)),
                   iris_angle,
                   0, 360,
                   2,  # Iris color
                   -1)  # Filled
        
        # Draw pupil (overwrites iris)
        cv2.ellipse(mask,
                   (int(cx), int(cy)),
                   (int(pupil_a), int(pupil_b)),
                   pupil_angle,
                   0, 360,
                   3,  # Pupil color
                   -1)  # Filled
        
        return mask
    
    def fit_to_mask(self, ritnet_mask: np.ndarray,
                   method: str = 'nelder-mead') -> Dict:
        """
        Fit ellipse model to RITnet mask.
        
        Args:
            ritnet_mask: (H, W) segmentation mask
            method: Optimization method
        
        Returns:
            Dictionary with fitted parameters
        """
        # Extract ellipses directly from mask using OpenCV
        pupil_mask = (ritnet_mask == 3).astype(np.uint8)
        iris_mask = ((ritnet_mask == 2) | (ritnet_mask == 3)).astype(np.uint8)
        
        # Fit ellipses using OpenCV
        pupil_ellipse = self._fit_ellipse_cv(pupil_mask)
        iris_ellipse = self._fit_ellipse_cv(iris_mask)
        
        if pupil_ellipse is None or iris_ellipse is None:
            print("Warning: Could not fit ellipses directly, using centroid method")
            return self._fallback_fit(ritnet_mask)
        
        # Extract parameters
        (cx_p, cy_p), (pupil_a, pupil_b), pupil_angle = pupil_ellipse
        (cx_i, cy_i), (iris_a, iris_b), iris_angle = iris_ellipse
        
        # Use average center
        cx = (cx_p + cx_i) / 2
        cy = (cy_p + cy_i) / 2
        
        # Use average angle
        angle = (pupil_angle + iris_angle) / 2
        
        # Axes are diameters, convert to semi-axes
        pupil_a, pupil_b = pupil_a / 2, pupil_b / 2
        iris_a, iris_b = iris_a / 2, iris_b / 2
        
        # Optional: Refine with optimization
        if method != 'direct':
            params = self._refine_with_optimization(
                ritnet_mask, cx, cy, pupil_a, pupil_b, angle,
                iris_a, iris_b, angle, method
            )
        else:
            params = {
                'cx': cx,
                'cy': cy,
                'pupil_a': pupil_a,
                'pupil_b': pupil_b,
                'pupil_angle': angle,
                'iris_a': iris_a,
                'iris_b': iris_b,
                'iris_angle': angle
            }
        
        # Compute metrics
        final_mask = self.render_ellipse_mask(**params)
        iou_pupil = self._compute_iou(ritnet_mask == 3, final_mask == 3)
        iou_iris = self._compute_iou(ritnet_mask == 2, final_mask == 2)
        
        # Estimate 3D gaze from ellipse
        gaze_estimates = self._estimate_gaze_from_ellipse(
            params['pupil_a'], params['pupil_b'], params['pupil_angle']
        )
        
        return {
            **params,
            'iou_pupil': iou_pupil,
            'iou_iris': iou_iris,
            **gaze_estimates,
            'pupil_ellipse': pupil_ellipse,
            'iris_ellipse': iris_ellipse
        }
    
    def _fit_ellipse_cv(self, binary_mask: np.ndarray) -> Optional[Tuple]:
        """Fit ellipse to binary mask using OpenCV."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        if len(largest) < 5:
            return None
        
        try:
            # Fit ellipse: returns ((cx, cy), (width, height), angle)
            ellipse = cv2.fitEllipse(largest)
            return ellipse
        except:
            return None
    
    def _fallback_fit(self, mask: np.ndarray) -> Dict:
        """Fallback: use centroid and approximate circular regions."""
        pupil_mask = (mask == 3).astype(np.uint8)
        iris_mask = (mask == 2).astype(np.uint8)
        
        # Pupil
        if pupil_mask.sum() > 0:
            M = cv2.moments(pupil_mask)
            cx = M['m10'] / M['m00'] if M['m00'] > 0 else self.img_width / 2
            cy = M['m01'] / M['m00'] if M['m00'] > 0 else self.img_height / 2
            r_pupil = np.sqrt(pupil_mask.sum() / np.pi)
        else:
            cx, cy, r_pupil = self.img_width / 2, self.img_height / 2, 20
        
        # Iris
        if iris_mask.sum() > 0:
            r_iris = np.sqrt(iris_mask.sum() / np.pi)
        else:
            r_iris = r_pupil * 3
        
        return {
            'cx': cx,
            'cy': cy,
            'pupil_a': r_pupil,
            'pupil_b': r_pupil,
            'pupil_angle': 0,
            'iris_a': r_iris,
            'iris_b': r_iris,
            'iris_angle': 0,
            'iou_pupil': 0.0,
            'iou_iris': 0.0,
            'theta_deg': 0.0,
            'phi_deg': 0.0
        }
    
    def _refine_with_optimization(self, mask, cx, cy, pupil_a, pupil_b, pupil_angle,
                                 iris_a, iris_b, iris_angle, method):
        """Refine ellipse parameters using optimization."""
        # Parameter bounds
        bounds = [
            (0, self.img_width),      # cx
            (0, self.img_height),     # cy
            (5, 100),                 # pupil_a
            (5, 100),                 # pupil_b
            (-180, 180),              # pupil_angle
            (20, 200),                # iris_a
            (20, 200),                # iris_b
            (-180, 180),              # iris_angle
        ]
        
        x0 = [cx, cy, pupil_a, pupil_b, pupil_angle, iris_a, iris_b, iris_angle]
        
        def loss_fn(params):
            try:
                cx, cy, pa, pb, pangle, ia, ib, iangle = params
                
                # Constraints
                if pa >= ia or pb >= ib:
                    return 1e6
                if pa < 5 or ia < 20:
                    return 1e6
                
                synthetic = self.render_ellipse_mask(cx, cy, pa, pb, pangle, ia, ib, iangle)
                
                iou_pupil = self._compute_iou(mask == 3, synthetic == 3)
                iou_iris = self._compute_iou(mask == 2, synthetic == 2)
                
                return 1.0 - (iou_pupil + iou_iris) / 2
            except:
                return 1e6
        
        if method == 'de':
            result = differential_evolution(loss_fn, bounds, maxiter=20, seed=42, polish=True)
        else:
            result = minimize(loss_fn, x0, method='Powell', bounds=bounds,
                            options={'maxiter': 300})
        
        cx, cy, pa, pb, pangle, ia, ib, iangle = result.x
        
        return {
            'cx': cx,
            'cy': cy,
            'pupil_a': pa,
            'pupil_b': pb,
            'pupil_angle': pangle,
            'iris_a': ia,
            'iris_b': ib,
            'iris_angle': iangle
        }
    
    def _estimate_gaze_from_ellipse(self, major_axis: float, minor_axis: float, 
                                    angle: float) -> Dict:
        """
        Estimate 3D gaze direction from ellipse parameters.
        
        When a circle is viewed at an angle, it projects to an ellipse.
        The ratio minor/major ≈ cos(viewing_angle).
        """
        # Eccentricity of ellipse
        if major_axis > 0:
            ratio = minor_axis / major_axis
            ratio = np.clip(ratio, 0, 1)
            
            # Estimate tilt angle from ratio
            # cos(tilt) ≈ minor/major
            tilt_rad = np.arccos(ratio)
            tilt_deg = np.degrees(tilt_rad)
            
            # The angle of the ellipse tells us the direction of tilt
            # This is a simplified model
            theta_deg = tilt_deg * np.sin(np.radians(angle))
            phi_deg = tilt_deg * np.cos(np.radians(angle))
        else:
            theta_deg, phi_deg = 0, 0
        
        return {
            'theta_deg': theta_deg,
            'phi_deg': phi_deg,
            'tilt_deg': tilt_deg if major_axis > 0 else 0,
            'ellipse_ratio': ratio if major_axis > 0 else 1.0
        }
    
    def _compute_iou(self, mask1, mask2):
        """Compute Intersection over Union."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def visualize_fit(self, image: np.ndarray, ritnet_mask: np.ndarray,
                     params: Dict, output_path: Optional[str] = None):
        """Visualize the fitted ellipses."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original with RITnet
        colored = self._colorize_mask(ritnet_mask)
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].imshow(colored, alpha=0.4)
        axes[0].set_title('RITnet Segmentation')
        axes[0].axis('off')
        
        # Fitted ellipses
        result = image.copy()
        
        # Draw iris ellipse (green)
        cv2.ellipse(result,
                   (int(params['cx']), int(params['cy'])),
                   (int(params['iris_a']), int(params['iris_b'])),
                   params['iris_angle'],
                   0, 360,
                   (0, 255, 0),
                   2)
        
        # Draw pupil ellipse (blue)
        cv2.ellipse(result,
                   (int(params['cx']), int(params['cy'])),
                   (int(params['pupil_a']), int(params['pupil_b'])),
                   params['pupil_angle'],
                   0, 360,
                   (255, 0, 0),
                   2)
        
        # Draw center
        cv2.circle(result, (int(params['cx']), int(params['cy'])), 3, (0, 255, 255), -1)
        
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Fitted Ellipses\nIoU: P={params['iou_pupil']:.3f} I={params['iou_iris']:.3f}")
        axes[1].axis('off')
        
        # Synthetic mask
        synthetic = self.render_ellipse_mask(
            params['cx'], params['cy'],
            params['pupil_a'], params['pupil_b'], params['pupil_angle'],
            params['iris_a'], params['iris_b'], params['iris_angle']
        )
        colored_synth = self._colorize_mask(synthetic)
        axes[2].imshow(colored_synth)
        axes[2].set_title(f"Synthetic Mask\nGaze: θ={params['theta_deg']:.1f}° φ={params['phi_deg']:.1f}°")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _colorize_mask(self, mask):
        """Convert mask to RGB."""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[mask == 1] = [255, 0, 0]
        colored[mask == 2] = [0, 255, 0]
        colored[mask == 3] = [0, 0, 255]
        return colored


def demo():
    """Demo with synthetic data."""
    import matplotlib
    matplotlib.use('Agg')
    
    print("Creating synthetic ellipse mask...")
    
    img_size = 400
    model = EllipseIrisPupilModel(img_size, img_size)
    
    # Generate synthetic mask with ellipses
    params_true = {
        'cx': 200,
        'cy': 200,
        'pupil_a': 30,
        'pupil_b': 20,  # Ellipse!
        'pupil_angle': 25,
        'iris_a': 90,
        'iris_b': 70,  # Ellipse!
        'iris_angle': 25
    }
    
    mask = model.render_ellipse_mask(**params_true)
    
    print("Fitting ellipse model...")
    params_fitted = model.fit_to_mask(mask, method='direct')
    
    print("\n=== Ground Truth ===")
    print(f"Center: ({params_true['cx']}, {params_true['cy']})")
    print(f"Pupil: a={params_true['pupil_a']}, b={params_true['pupil_b']}, angle={params_true['pupil_angle']}°")
    print(f"Iris: a={params_true['iris_a']}, b={params_true['iris_b']}, angle={params_true['iris_angle']}°")
    
    print("\n=== Fitted ===")
    print(f"Center: ({params_fitted['cx']:.1f}, {params_fitted['cy']:.1f})")
    print(f"Pupil: a={params_fitted['pupil_a']:.1f}, b={params_fitted['pupil_b']:.1f}, angle={params_fitted['pupil_angle']:.1f}°")
    print(f"Iris: a={params_fitted['iris_a']:.1f}, b={params_fitted['iris_b']:.1f}, angle={params_fitted['iris_angle']:.1f}°")
    print(f"IoU: Pupil={params_fitted['iou_pupil']:.3f}, Iris={params_fitted['iou_iris']:.3f}")
    print(f"Estimated gaze: θ={params_fitted['theta_deg']:.1f}° φ={params_fitted['phi_deg']:.1f}°")
    
    # Visualize
    dummy_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    model.visualize_fit(dummy_img, mask, params_fitted, 'ellipse_fit_demo.png')
    
    print("\n✅ Saved visualization to ellipse_fit_demo.png")


if __name__ == '__main__':
    demo()
