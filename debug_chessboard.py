#!/usr/bin/env python3
"""
Debug script a sakktábla detektáláshoz
"""

import cv2
import numpy as np

def analyze_video_and_detect_chessboard(video_path):
    """Részletes analízis"""
    cap = cv2.VideoCapture(video_path)
    
    # Video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total}")
    
    # Try different chessboard sizes
    possible_sizes = [
        (10, 7), (7, 10),  # Original
        (9, 6), (6, 9),    # 10x7 - 1
        (8, 5), (5, 8),    # Smaller
        (11, 8), (8, 11),  # Larger
        (7, 5), (5, 7),    # Much smaller
    ]
    
    print(f"\n🔍 Trying to detect chessboard...")
    print(f"Testing {len(possible_sizes)} different sizes...\n")
    
    # Test on multiple frames
    test_frames = [0, 100, 500, 1000, 1500]
    
    for frame_idx in test_frames:
        if frame_idx >= total:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print(f"Frame {frame_idx}:")
        
        for size in possible_sizes:
            ret = cv2.findChessboardCorners(
                gray, size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret[0]:
                print(f"  ✅ Found {size[0]}x{size[1]} chessboard!")
                
                # Save image with detected corners
                display = frame.copy()
                cv2.drawChessboardCorners(display, size, ret[1], ret[0])
                cv2.imwrite(f'chessboard_detected_frame{frame_idx}_{size[0]}x{size[1]}.png', display)
                print(f"     Saved: chessboard_detected_frame{frame_idx}_{size[0]}x{size[1]}.png")
                
                cap.release()
                return size
        
        print(f"  ❌ No chessboard detected in frame {frame_idx}")
    
    cap.release()
    
    print(f"\n💡 Suggestions:")
    print(f"   1. Check if the chessboard is clearly visible")
    print(f"   2. Try different lighting conditions")
    print(f"   3. Ensure the chessboard is flat and not distorted")
    print(f"   4. The chessboard size might be different than expected")
    
    return None

if __name__ == "__main__":
    result = analyze_video_and_detect_chessboard('eye_cam.mkv')
    if result:
        print(f"\n✅ Success! Use chessboard size: {result}")
    else:
        print(f"\n❌ Could not detect any chessboard pattern")
