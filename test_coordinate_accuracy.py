#!/usr/bin/env python3
"""
Compare EllSeg vs Traditional CV detection
Test coordinate accuracy after rescaling fix
"""
import cv2
import numpy as np
from ellseg_integration import EllSegDetector

# Load video
cap = cv2.VideoCapture("eye1.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read frame")
    exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Original frame shape: {frame.shape}")
print(f"Gray frame shape: {gray.shape}")

# === TRADITIONAL CV DETECTION ===
print("\n" + "="*60)
print("Traditional CV Detection (threshold + contours)")
print("="*60)

# Simple threshold-based detection
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # Filter by area for pupil
    pupil_contours = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
    
    if len(pupil_contours) > 0:
        pupil_contour = max(pupil_contours, key=cv2.contourArea)
        
        if len(pupil_contour) >= 5:
            ellipse = cv2.fitEllipse(pupil_contour)
            cx, cy = ellipse[0]
            w, h = ellipse[1]
            angle = ellipse[2]
            
            print(f"Pupil center: ({cx:.2f}, {cy:.2f})")
            print(f"Pupil axes: {w:.2f} × {h:.2f}")
            print(f"Pupil angle: {angle:.2f}°")
            
            # Visualize
            vis_cv = frame.copy()
            cv2.ellipse(vis_cv, ellipse, (0, 0, 255), 2)
            cv2.circle(vis_cv, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(vis_cv, f"CV: ({cx:.0f}, {cy:.0f})", 
                       (int(cx) + 10, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite("comparison_cv.png", vis_cv)
            print("✅ CV visualization saved to comparison_cv.png")
        else:
            print("⚠️  Pupil contour too small")
    else:
        print("⚠️  No pupil contours found")
else:
    print("⚠️  No contours found")

# === ELLSEG DETECTION ===
print("\n" + "="*60)
print("EllSeg CNN Detection (with fixed coordinate transform)")
print("="*60)

detector = EllSegDetector(device='cpu')

if detector.model is not None:
    results = detector.detect(gray)
    
    pupil_ellipse = results['pupil_ellipse']
    iris_ellipse = results['iris_ellipse']
    
    if not np.all(pupil_ellipse == -1):
        cx, cy, a, b, angle = pupil_ellipse
        print(f"Pupil center: ({cx:.2f}, {cy:.2f})")
        print(f"Pupil semi-axes: {a:.2f} × {b:.2f}")
        print(f"Pupil full axes: {2*a:.2f} × {2*b:.2f}")
        print(f"Pupil angle: {np.rad2deg(angle):.2f}°")
    
    if not np.all(iris_ellipse == -1):
        cx, cy, a, b, angle = iris_ellipse
        print(f"\nIris center: ({cx:.2f}, {cy:.2f})")
        print(f"Iris semi-axes: {a:.2f} × {b:.2f}")
        print(f"Iris full axes: {2*a:.2f} × {2*b:.2f}")
        print(f"Iris angle: {np.rad2deg(angle):.2f}°")
    
    print(f"\nConfidence: {results['confidence']:.3f}")
    
    # Visualize
    vis_ellseg = detector.visualize(frame, results)
    cv2.imwrite("comparison_ellseg.png", vis_ellseg)
    print("✅ EllSeg visualization saved to comparison_ellseg.png")
    
    # === SIDE-BY-SIDE COMPARISON ===
    print("\n" + "="*60)
    print("Creating side-by-side comparison...")
    print("="*60)
    
    # Draw both on same image
    vis_both = frame.copy()
    
    # Draw Traditional CV (red)
    if len(pupil_contours) > 0 and len(pupil_contour) >= 5:
        ellipse_cv = cv2.fitEllipse(pupil_contour)
        cv2.ellipse(vis_both, ellipse_cv, (0, 0, 255), 2)
        cv2.circle(vis_both, (int(ellipse_cv[0][0]), int(ellipse_cv[0][1])), 
                  5, (0, 0, 255), -1)
        cv2.putText(vis_both, "CV", 
                   (int(ellipse_cv[0][0]) - 30, int(ellipse_cv[0][1]) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw EllSeg (green)
    if not np.all(pupil_ellipse == -1):
        cx, cy, a, b, angle = pupil_ellipse
        cv2.ellipse(vis_both, 
                   (int(cx), int(cy)), 
                   (int(a), int(b)), 
                   np.rad2deg(angle), 
                   0, 360, (0, 255, 0), 2)
        cv2.circle(vis_both, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.putText(vis_both, "EllSeg", 
                   (int(cx) + 10, int(cy) + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("comparison_both.png", vis_both)
    print("✅ Combined comparison saved to comparison_both.png")
    
    # Calculate offset if both detected
    if len(pupil_contours) > 0 and len(pupil_contour) >= 5 and not np.all(pupil_ellipse == -1):
        ellipse_cv = cv2.fitEllipse(pupil_contour)
        cv_cx, cv_cy = ellipse_cv[0]
        ellseg_cx, ellseg_cy = pupil_ellipse[0], pupil_ellipse[1]
        
        offset_x = ellseg_cx - cv_cx
        offset_y = ellseg_cy - cv_cy
        offset_dist = np.sqrt(offset_x**2 + offset_y**2)
        
        print(f"\n📊 Coordinate Offset:")
        print(f"   ΔX = {offset_x:.2f} px")
        print(f"   ΔY = {offset_y:.2f} px")
        print(f"   Distance = {offset_dist:.2f} px")
        
        if offset_dist < 5:
            print("   ✅ Excellent alignment (<5px)")
        elif offset_dist < 10:
            print("   ✓ Good alignment (<10px)")
        else:
            print(f"   ⚠️  Significant offset (>{offset_dist:.0f}px)")

else:
    print("⚠️  EllSeg model not loaded")

print("\n" + "="*60)
print("Comparison complete!")
print("="*60)
