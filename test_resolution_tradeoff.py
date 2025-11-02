#!/usr/bin/env python3
"""Test different input resolutions for EllSeg to find optimal speed/accuracy trade-off"""

import time
import cv2
import numpy as np
from ellseg_integration import EllSegDetector
import torch

# Load a sample frame
print("Loading test frame...")
cap = cv2.VideoCapture('eye1.mp4')
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Test different resolutions
resolutions = [
    (320, 240, "Current (baseline)"),
    (256, 192, "20% reduction"),
    (224, 168, "30% reduction"),
    (192, 144, "43% reduction"),
    (160, 120, "56% reduction"),
]

print("\n" + "="*80)
print("TESTING DIFFERENT INPUT RESOLUTIONS")
print("="*80)

device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'

results = []

for width, height, description in resolutions:
    print(f"\n{'─'*80}")
    print(f"Testing: {width}x{height} ({description})")
    print(f"{'─'*80}")
    
    # Initialize detector with custom resolution
    detector = EllSegDetector(device=device)
    
    # Warm-up
    _ = detector.preprocess_frame(gray, target_size=(width, height))
    
    # Time preprocessing with this resolution
    preprocess_times = []
    for i in range(10):
        t0 = time.perf_counter()
        input_tensor, transform_info = detector.preprocess_frame(gray, target_size=(width, height))
        elapsed = (time.perf_counter() - t0) * 1000
        preprocess_times.append(elapsed)
    
    # Time full detection with this resolution
    detection_times = []
    last_result = None
    for i in range(10):
        t0 = time.perf_counter()
        # Manually call detect with custom preprocessing
        input_tensor, transform_info = detector.preprocess_frame(gray, target_size=(width, height))
        with torch.no_grad():
            x4, x3, x2, x1, x = detector.model.enc(input_tensor)
            latent = torch.mean(x.flatten(start_dim=2), -1)
            elOut = detector.model.elReg(x, 0)
            seg_out = detector.model.dec(x4, x3, x2, x1, x)
        
        seg_map = detector._get_segmentation_map(seg_out)
        pupil_ellipse = detector._fit_ellipse_to_mask(seg_map, class_id=2)
        iris_ellipse = detector._fit_ellipse_to_mask(seg_map, class_id=1)
        pupil_ellipse = detector._rescale_ellipse(pupil_ellipse, transform_info)
        iris_ellipse = detector._rescale_ellipse(iris_ellipse, transform_info)
        
        elapsed = (time.perf_counter() - t0) * 1000
        detection_times.append(elapsed)
        
        if i == 9:  # Save last result for comparison
            last_result = (pupil_ellipse, iris_ellipse)
    
    avg_time = np.mean(detection_times)
    fps = 1000 / avg_time
    
    print(f"  Preprocessing: {np.mean(preprocess_times):.2f}ms ± {np.std(preprocess_times):.2f}ms")
    print(f"  Detection:     {avg_time:.2f}ms ± {np.std(detection_times):.2f}ms")
    print(f"  FPS:           {fps:.2f}")
    
    if last_result[0] is not None and not np.all(last_result[0] == -1):
        pupil_cx, pupil_cy, pupil_a, pupil_b, pupil_angle = last_result[0]
        print(f"  Pupil center:  ({pupil_cx:.1f}, {pupil_cy:.1f})")
        print(f"  Pupil axes:    {pupil_a:.1f} x {pupil_b:.1f}")
    else:
        print(f"  ⚠️  Pupil not detected!")
    
    if last_result[1] is not None and not np.all(last_result[1] == -1):
        iris_cx, iris_cy, iris_a, iris_b, iris_angle = last_result[1]
        print(f"  Iris center:   ({iris_cx:.1f}, {iris_cy:.1f})")
        print(f"  Iris axes:     {iris_a:.1f} x {iris_b:.1f}")
    else:
        print(f"  ⚠️  Iris not detected!")
    
    results.append({
        'resolution': (width, height),
        'description': description,
        'time': avg_time,
        'fps': fps,
        'pupil': last_result[0],
        'iris': last_result[1]
    })

# Summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"{'Resolution':<15} {'Description':<20} {'Time':<12} {'FPS':<8} {'Speedup':<10}")
print("─"*80)

baseline_time = results[0]['time']
baseline_pupil = results[0]['pupil']

for r in results:
    speedup = baseline_time / r['time']
    
    # Check if detection still works
    detection_ok = "✅" if (r['pupil'] is not None and not np.all(r['pupil'] == -1)) else "❌"
    
    # Compare ellipse params (if both detected)
    if baseline_pupil is not None and r['pupil'] is not None:
        if not np.all(baseline_pupil == -1) and not np.all(r['pupil'] == -1):
            # Calculate difference in center position
            diff_cx = abs(baseline_pupil[0] - r['pupil'][0])
            diff_cy = abs(baseline_pupil[1] - r['pupil'][1])
            diff_center = np.sqrt(diff_cx**2 + diff_cy**2)
            
            # Calculate difference in axes
            diff_a = abs(baseline_pupil[2] - r['pupil'][2])
            diff_b = abs(baseline_pupil[3] - r['pupil'][3])
            
            accuracy_note = f"(Δcenter={diff_center:.1f}px, Δaxes={diff_a:.1f},{diff_b:.1f}px)"
        else:
            accuracy_note = ""
    else:
        accuracy_note = ""
    
    res_str = f"{r['resolution'][0]}x{r['resolution'][1]}"
    print(f"{res_str:<15} {r['description']:<20} {r['time']:>8.2f}ms  {r['fps']:>6.2f}  {speedup:>6.2f}x {detection_ok} {accuracy_note}")

print("="*80)

# Recommendation
print("\n💡 RECOMMENDATION:")
best_compromise = results[2]  # 224x168
print(f"   Use {best_compromise['resolution'][0]}x{best_compromise['resolution'][1]} for best speed/accuracy trade-off")
print(f"   Expected speedup: {baseline_time / best_compromise['time']:.2f}x")
print(f"   Expected FPS: {best_compromise['fps']:.1f} (was {results[0]['fps']:.1f})")
print(f"   Detection quality: Should be similar to baseline")
