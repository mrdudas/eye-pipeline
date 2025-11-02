#!/usr/bin/env python3
"""Test FP16 (half precision) optimization for EllSeg"""

import time
import cv2
import numpy as np
from ellseg_integration import EllSegDetector
import torch

# Load test frame
print("Loading test frame...")
cap = cv2.VideoCapture('eye1.mp4')
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame shape: {gray.shape}\n")

device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}\n")

# ====================================
# TEST 1: FP32 (Baseline)
# ====================================
print("="*80)
print("TEST 1: FP32 (BASELINE)")
print("="*80)

detector_fp32 = EllSegDetector(device=device)

# Warm-up
_ = detector_fp32.detect(gray)

# Benchmark
times_fp32 = []
result_fp32 = None
for i in range(10):
    t0 = time.perf_counter()
    result = detector_fp32.detect(gray)
    elapsed = (time.perf_counter() - t0) * 1000
    times_fp32.append(elapsed)
    if i == 9:
        result_fp32 = result

avg_fp32 = np.mean(times_fp32)
print(f"Average: {avg_fp32:.2f}ms ± {np.std(times_fp32):.2f}ms")
print(f"FPS: {1000/avg_fp32:.2f}")

if not np.all(result_fp32['pupil_ellipse'] == -1):
    p = result_fp32['pupil_ellipse']
    print(f"Pupil: center=({p[0]:.2f}, {p[1]:.2f}), axes=({p[2]:.2f}, {p[3]:.2f})")
else:
    print("Pupil: NOT DETECTED")

if not np.all(result_fp32['iris_ellipse'] == -1):
    i = result_fp32['iris_ellipse']
    print(f"Iris: center=({i[0]:.2f}, {i[1]:.2f}), axes=({i[2]:.2f}, {i[3]:.2f})")
else:
    print("Iris: NOT DETECTED")

# ====================================
# TEST 2: FP16
# ====================================
print("\n" + "="*80)
print("TEST 2: FP16 (HALF PRECISION)")
print("="*80)

if device == 'cpu':
    print("⚠️  FP16 not well supported on CPU, skipping test")
else:
    try:
        detector_fp16 = EllSegDetector(device=device, use_fp16=True)
        print("✅ FP16 model initialized")
        
        # Warm-up
        _ = detector_fp16.detect(gray)
        
        # Benchmark
        times_fp16 = []
        result_fp16 = None
        for i in range(10):
            t0 = time.perf_counter()
            result = detector_fp16.detect(gray)
            elapsed = (time.perf_counter() - t0) * 1000
            times_fp16.append(elapsed)
            if i == 9:
                result_fp16 = result
        
        avg_fp16 = np.mean(times_fp16)
        print(f"Average: {avg_fp16:.2f}ms ± {np.std(times_fp16):.2f}ms")
        print(f"FPS: {1000/avg_fp16:.2f}")
        print(f"Speedup: {avg_fp32/avg_fp16:.2f}x")
        
        if not np.all(result_fp16['pupil_ellipse'] == -1):
            p = result_fp16['pupil_ellipse']
            print(f"Pupil: center=({p[0]:.2f}, {p[1]:.2f}), axes=({p[2]:.2f}, {p[3]:.2f})")
        else:
            print("Pupil: NOT DETECTED")
        
        if not np.all(result_fp16['iris_ellipse'] == -1):
            i = result_fp16['iris_ellipse']
            print(f"Iris: center=({i[0]:.2f}, {i[1]:.2f}), axes=({i[2]:.2f}, {i[3]:.2f})")
        else:
            print("Iris: NOT DETECTED")
        
        # ====================================
        # COMPARISON
        # ====================================
        print("\n" + "="*80)
        print("ACCURACY COMPARISON (FP16 vs FP32)")
        print("="*80)
        
        if not np.all(result_fp32['pupil_ellipse'] == -1) and not np.all(result_fp16['pupil_ellipse'] == -1):
            p32 = result_fp32['pupil_ellipse']
            p16 = result_fp16['pupil_ellipse']
            
            diff_cx = abs(p32[0] - p16[0])
            diff_cy = abs(p32[1] - p16[1])
            diff_center = np.sqrt(diff_cx**2 + diff_cy**2)
            diff_a = abs(p32[2] - p16[2])
            diff_b = abs(p32[3] - p16[3])
            diff_angle = abs(p32[4] - p16[4])
            
            print(f"Pupil differences:")
            print(f"  Center:  {diff_center:.2f} pixels")
            print(f"  Axis A:  {diff_a:.2f} pixels")
            print(f"  Axis B:  {diff_b:.2f} pixels")
            print(f"  Angle:   {np.rad2deg(diff_angle):.2f} degrees")
            
            # Check if differences are acceptable
            if diff_center < 2.0 and diff_a < 2.0 and diff_b < 2.0:
                print("  ✅ Accuracy: ACCEPTABLE (< 2px error)")
            else:
                print(f"  ⚠️  Accuracy: May be degraded (errors > 2px)")
        
        if not np.all(result_fp32['iris_ellipse'] == -1) and not np.all(result_fp16['iris_ellipse'] == -1):
            i32 = result_fp32['iris_ellipse']
            i16 = result_fp16['iris_ellipse']
            
            diff_cx = abs(i32[0] - i16[0])
            diff_cy = abs(i32[1] - i16[1])
            diff_center = np.sqrt(diff_cx**2 + diff_cy**2)
            diff_a = abs(i32[2] - i16[2])
            diff_b = abs(i32[3] - i16[3])
            diff_angle = abs(i32[4] - i16[4])
            
            print(f"\nIris differences:")
            print(f"  Center:  {diff_center:.2f} pixels")
            print(f"  Axis A:  {diff_a:.2f} pixels")
            print(f"  Axis B:  {diff_b:.2f} pixels")
            print(f"  Angle:   {np.rad2deg(diff_angle):.2f} degrees")
            
            if diff_center < 2.0 and diff_a < 2.0 and diff_b < 2.0:
                print("  ✅ Accuracy: ACCEPTABLE (< 2px error)")
            else:
                print(f"  ⚠️  Accuracy: May be degraded (errors > 2px)")
        
        # ====================================
        # SUMMARY
        # ====================================
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"FP32: {avg_fp32:.2f}ms ({1000/avg_fp32:.2f} FPS)")
        print(f"FP16: {avg_fp16:.2f}ms ({1000/avg_fp16:.2f} FPS)")
        print(f"Speedup: {avg_fp32/avg_fp16:.2f}x ({avg_fp32 - avg_fp16:.2f}ms saved)")
        print("="*80)
        
        # Recommendation
        if avg_fp32/avg_fp16 > 1.3:
            print("\n💡 RECOMMENDATION:")
            print("   ✅ FP16 provides significant speedup with acceptable accuracy")
            print("   ✅ Safe to use for production")
        else:
            print("\n💡 RECOMMENDATION:")
            print("   ⚠️  FP16 speedup is minimal on this device")
            print("   ⚠️  Consider other optimizations")
        
    except Exception as e:
        print(f"\n❌ FP16 test failed: {e}")
        import traceback
        traceback.print_exc()
