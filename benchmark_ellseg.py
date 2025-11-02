#!/usr/bin/env python3
"""Benchmark EllSeg detection performance"""

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

if not ret:
    print("Error: Could not load video frame")
    exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"Frame shape: {gray.shape}")

# Initialize detector
print('\nInitializing EllSeg detector...')
device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
detector = EllSegDetector(device=device)

# Warm-up run
print('Warm-up run...')
_ = detector.detect(gray)

# Time detection
print('\n' + '='*60)
print('TIMING 10 DETECTION RUNS')
print('='*60)
times = []
for i in range(10):
    t0 = time.perf_counter()
    results = detector.detect(gray)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
    print(f'  Run {i+1}: {elapsed:.2f}ms')

print('\n' + '='*60)
print('STATISTICS')
print('='*60)
print(f'Average: {np.mean(times):.2f}ms')
print(f'Min: {np.min(times):.2f}ms')
print(f'Max: {np.max(times):.2f}ms')
print(f'Std: {np.std(times):.2f}ms')
print(f'FPS: {1000/np.mean(times):.2f}')
print('='*60)

# Now profile individual components
print('\n' + '='*60)
print('PROFILING INDIVIDUAL COMPONENTS')
print('='*60)

# Test preprocessing time
preprocess_times = []
for i in range(10):
    t0 = time.perf_counter()
    input_tensor, transform_info = detector.preprocess_frame(gray)
    elapsed = (time.perf_counter() - t0) * 1000
    preprocess_times.append(elapsed)

print(f'Preprocessing: {np.mean(preprocess_times):.2f}ms ± {np.std(preprocess_times):.2f}ms')

# Test model forward pass time (without preprocessing)
input_tensor, transform_info = detector.preprocess_frame(gray)
forward_times = []
for i in range(10):
    t0 = time.perf_counter()
    with torch.no_grad():
        x4, x3, x2, x1, x = detector.model.enc(input_tensor)
        latent = torch.mean(x.flatten(start_dim=2), -1)
        elOut = detector.model.elReg(x, 0)
        seg_out = detector.model.dec(x4, x3, x2, x1, x)
    elapsed = (time.perf_counter() - t0) * 1000
    forward_times.append(elapsed)

print(f'Model Forward: {np.mean(forward_times):.2f}ms ± {np.std(forward_times):.2f}ms')

# Test postprocessing time
seg_out_sample = seg_out
postprocess_times = []
for i in range(10):
    t0 = time.perf_counter()
    seg_map = detector._get_segmentation_map(seg_out_sample)
    pupil_ellipse = detector._fit_ellipse_to_mask(seg_map, class_id=2)
    iris_ellipse = detector._fit_ellipse_to_mask(seg_map, class_id=1)
    pupil_ellipse = detector._rescale_ellipse(pupil_ellipse, transform_info)
    iris_ellipse = detector._rescale_ellipse(iris_ellipse, transform_info)
    seg_map = detector._rescale_segmap(seg_map, transform_info)
    elapsed = (time.perf_counter() - t0) * 1000
    postprocess_times.append(elapsed)

print(f'Postprocessing: {np.mean(postprocess_times):.2f}ms ± {np.std(postprocess_times):.2f}ms')

total_components = np.mean(preprocess_times) + np.mean(forward_times) + np.mean(postprocess_times)
print(f'\nTotal (sum of components): {total_components:.2f}ms')
print(f'Overhead: {np.mean(times) - total_components:.2f}ms')

print('\n' + '='*60)
print('BREAKDOWN')
print('='*60)
print(f'Preprocessing:  {np.mean(preprocess_times)/np.mean(times)*100:5.1f}%')
print(f'Model Forward:  {np.mean(forward_times)/np.mean(times)*100:5.1f}%')
print(f'Postprocessing: {np.mean(postprocess_times)/np.mean(times)*100:5.1f}%')
print('='*60)
