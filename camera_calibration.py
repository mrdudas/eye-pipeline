#!/usr/bin/env python3
"""
Camera Calibration Module
Kamera geometriai korrekció sakktábla alapú kalibrációval
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


class CameraCalibrator:
    """Kamera kalibráció OpenCV calibrateCamera használatával"""
    
    def __init__(self, chessboard_size=(10, 7), square_size_mm=1.0):
        """
        Inicializálás
        
        Args:
            chessboard_size: (cols, rows) belső sarkok száma
            square_size_mm: Sakktábla négyzet mérete mm-ben
        """
        self.chessboard_size = chessboard_size
        self.square_size_mm = square_size_mm
        
        # 3D koordináták előkészítése (sakktábla sík, Z=0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                                     0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm
        
        # Calibration eredmények
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
    def calibrate_from_video(self, video_path, max_frames=30, 
                            skip_frames=10, show_detection=False):
        """
        Kalibráció videó fájlból
        
        Args:
            video_path: Videó fájl elérési útja (eye_cam.mkv)
            max_frames: Maximum hány frame-et használjon
            skip_frames: Hány frame-et uggorjon át frame-ek között
            show_detection: Mutassa-e a detektált sarkokat
            
        Returns:
            success: bool, sikeres volt-e a kalibráció
            info: dict, kalibráció részletek
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, {"error": f"Cannot open video: {video_path}"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n📹 Video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]} inner corners")
        print(f"Square size: {self.square_size_mm} mm")
        
        # Objektum és kép pontok gyűjtése
        obj_points = []  # 3D pontok a valós térben
        img_points = []  # 2D pontok a képen
        
        frame_num = 0
        detected_count = 0
        img_size = None
        
        # Criteria for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print(f"\n🔍 Detecting chessboard corners...")
        pbar = tqdm(total=max_frames, desc="Frames processed")
        
        while detected_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames
            if frame_num % skip_frames != 0:
                frame_num += 1
                continue
            
            frame_num += 1
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK
            )
            
            if ret:
                # Refine corner positions to sub-pixel accuracy
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                
                obj_points.append(self.objp)
                img_points.append(corners_refined)
                
                detected_count += 1
                pbar.update(1)
                
                # Show detection (optional)
                if show_detection:
                    display_frame = frame.copy()
                    cv2.drawChessboardCorners(
                        display_frame, 
                        self.chessboard_size, 
                        corners_refined, 
                        ret
                    )
                    cv2.putText(display_frame, 
                               f"Detected: {detected_count}/{max_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    
                    # Resize for display
                    h, w = display_frame.shape[:2]
                    scale = min(800/w, 600/h)
                    display_resized = cv2.resize(
                        display_frame, 
                        (int(w*scale), int(h*scale))
                    )
                    
                    cv2.imshow('Calibration', display_resized)
                    cv2.waitKey(50)
        
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()
        
        if detected_count == 0:
            return False, {
                "error": "No chessboard corners detected!",
                "frames_checked": frame_num,
                "detected": 0
            }
        
        print(f"\n✅ Detected chessboard in {detected_count} frames")
        print(f"📐 Image size: {img_size}")
        
        # Perform calibration
        print(f"\n🔧 Calibrating camera...")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, 
            img_points, 
            img_size, 
            None, 
            None
        )
        
        if not ret:
            return False, {"error": "Calibration failed!"}
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points_reprojected, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], 
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points_reprojected, 
                           cv2.NORM_L2) / len(img_points_reprojected)
            total_error += error
        
        mean_error = total_error / len(obj_points)
        
        # Store results
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.calibration_error = mean_error
        
        # Prepare info
        info = {
            "success": True,
            "frames_used": detected_count,
            "image_size": img_size,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coeffs": dist_coeffs.tolist(),
            "reprojection_error_px": float(mean_error),
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
            "k1": float(dist_coeffs[0, 0]),
            "k2": float(dist_coeffs[0, 1]),
            "p1": float(dist_coeffs[0, 2]),
            "p2": float(dist_coeffs[0, 3]),
            "k3": float(dist_coeffs[0, 4]) if dist_coeffs.shape[1] > 4 else 0.0,
        }
        
        print(f"\n✅ Calibration successful!")
        print(f"📊 Reprojection error: {mean_error:.4f} pixels")
        print(f"\nCamera Matrix:")
        print(f"  fx = {info['fx']:.2f} px")
        print(f"  fy = {info['fy']:.2f} px")
        print(f"  cx = {info['cx']:.2f} px")
        print(f"  cy = {info['cy']:.2f} px")
        print(f"\nDistortion Coefficients:")
        print(f"  k1 = {info['k1']:.6f}")
        print(f"  k2 = {info['k2']:.6f}")
        print(f"  p1 = {info['p1']:.6f}")
        print(f"  p2 = {info['p2']:.6f}")
        print(f"  k3 = {info['k3']:.6f}")
        
        return True, info
    
    def save_calibration(self, filename="camera_calibration.yaml"):
        """
        Kalibráció mentése YAML fájlba
        
        Args:
            filename: Fájlnév (camera_calibration.yaml)
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data to save! Run calibrate_from_video first.")
        
        data = {
            "calibration_date": str(np.datetime64('today')),
            "chessboard_size": list(self.chessboard_size),
            "square_size_mm": float(self.square_size_mm),
            "image_width": int(self.camera_matrix.shape[1]) if hasattr(self.camera_matrix, 'shape') else 0,
            "image_height": int(self.camera_matrix.shape[0]) if hasattr(self.camera_matrix, 'shape') else 0,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": self.camera_matrix.flatten().tolist()
            },
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(self.dist_coeffs[0]),
                "data": self.dist_coeffs.flatten().tolist()
            },
            "reprojection_error": float(self.calibration_error),
            "intrinsics": {
                "fx": float(self.camera_matrix[0, 0]),
                "fy": float(self.camera_matrix[1, 1]),
                "cx": float(self.camera_matrix[0, 2]),
                "cy": float(self.camera_matrix[1, 2])
            },
            "distortion": {
                "k1": float(self.dist_coeffs[0, 0]),
                "k2": float(self.dist_coeffs[0, 1]),
                "p1": float(self.dist_coeffs[0, 2]),
                "p2": float(self.dist_coeffs[0, 3]),
                "k3": float(self.dist_coeffs[0, 4]) if self.dist_coeffs.shape[1] > 4 else 0.0
            }
        }
        
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"\n💾 Calibration saved to: {filename}")
        
        return filename
    
    @staticmethod
    def load_calibration(filename="camera_calibration.yaml"):
        """
        Kalibráció betöltése YAML fájlból
        
        Args:
            filename: Fájlnév
            
        Returns:
            camera_matrix: numpy array (3x3)
            dist_coeffs: numpy array (1x5)
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"Calibration file not found: {filename}")
        
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct camera matrix
        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        
        # Reconstruct distortion coefficients
        dist_coeffs = np.array(data['distortion_coefficients']['data']).reshape(1, -1)
        
        print(f"✅ Calibration loaded from: {filename}")
        print(f"   Reprojection error: {data['reprojection_error']:.4f} px")
        
        return camera_matrix, dist_coeffs
    
    def undistort_frame(self, frame):
        """
        Frame undistort (geometriai korrekció)
        
        Args:
            frame: Input frame (BGR vagy grayscale)
            
        Returns:
            undistorted: Korrigált frame
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("No calibration data! Load calibration first.")
        
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
    
    def visualize_distortion(self, frame):
        """
        Vizualizáció: original vs undistorted
        
        Args:
            frame: Input frame
            
        Returns:
            comparison: Side-by-side összehasonlítás
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data!")
        
        undistorted = self.undistort_frame(frame)
        
        # Side by side
        h, w = frame.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        if len(frame.shape) == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            undistorted_color = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame
            undistorted_color = undistorted
        
        comparison[:, :w] = frame_color
        comparison[:, w:] = undistorted_color
        
        # Labels
        cv2.putText(comparison, "Original (Distorted)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Undistorted", (w+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Grid overlay for better visualization
        for i in range(0, h, 50):
            cv2.line(comparison, (0, i), (w*2, i), (128, 128, 128), 1)
        for j in range(0, w*2, 50):
            cv2.line(comparison, (j, 0), (j, h), (128, 128, 128), 1)
        
        return comparison


def main():
    """Standalone kalibráció futtatás"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--video', type=str, default='eye_cam.mkv',
                       help='Calibration video file')
    parser.add_argument('--output', type=str, default='camera_calibration.yaml',
                       help='Output calibration file')
    parser.add_argument('--chessboard', type=str, default='10x7',
                       help='Chessboard size (cols x rows)')
    parser.add_argument('--square-size', type=float, default=1.0,
                       help='Square size in mm')
    parser.add_argument('--max-frames', type=int, default=30,
                       help='Maximum frames to use')
    parser.add_argument('--show', action='store_true',
                       help='Show detection process')
    
    args = parser.parse_args()
    
    # Parse chessboard size
    cols, rows = map(int, args.chessboard.split('x'))
    
    # Create calibrator
    calibrator = CameraCalibrator(
        chessboard_size=(cols, rows),
        square_size_mm=args.square_size
    )
    
    # Calibrate
    success, info = calibrator.calibrate_from_video(
        args.video,
        max_frames=args.max_frames,
        show_detection=args.show
    )
    
    if success:
        # Save calibration
        calibrator.save_calibration(args.output)
        
        print(f"\n✅ Calibration complete!")
        print(f"📄 Results saved to: {args.output}")
    else:
        print(f"\n❌ Calibration failed!")
        print(f"Error: {info.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
