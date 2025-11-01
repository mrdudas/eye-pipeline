#!/usr/bin/env python3
"""
Test EllSeg on a frame from eye1.mp4
"""
import cv2
import numpy as np
from ellseg_integration import EllSegDetector

# Create detector
print("Creating EllSeg detector...")
detector = EllSegDetector(device='cpu')

if detector.model is None:
    print("❌ Model not loaded!")
    exit(1)

# Load video
video_path = "eye1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    exit(1)

print(f"✅ Video loaded: {video_path}")

# Get first frame
ret, frame = cap.read()
if not ret:
    print("❌ Could not read first frame")
    exit(1)

print(f"Frame shape: {frame.shape}")

# Convert to grayscale
if len(frame.shape) == 3:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    gray = frame

print("Running EllSeg detection...")

# Detect
results = detector.detect(gray)

print("\n" + "="*60)
print("EllSeg Detection Results:")
print("="*60)
print(f"Pupil ellipse: {results['pupil_ellipse']}")
print(f"Iris ellipse:  {results['iris_ellipse']}")
print(f"Confidence:    {results['confidence']:.3f}")
print(f"Segmentation map shape: {results['seg_map'].shape}")
print(f"Segmentation classes: {np.unique(results['seg_map'])}")
print()
print(f"Pupil pixels: {np.sum(results['seg_map'] == 2)}")
print(f"Iris pixels:  {np.sum(results['seg_map'] == 1)}")
print("="*60)

# Visualize
vis_frame = detector.visualize(frame, results)

# Save
cv2.imwrite("ellseg_test_output.png", vis_frame)
print("\n✅ Visualization saved to: ellseg_test_output.png")

# Display (if possible)
try:
    cv2.imshow("EllSeg Test", vis_frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
except:
    print("(Display not available)")

cap.release()
print("\n✅ Test complete!")
