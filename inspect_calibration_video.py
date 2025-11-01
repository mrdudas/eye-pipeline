#!/usr/bin/env python3
"""
Vizualizálja az eye_cam.mkv videó frame-jeit
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_frames(video_path, num_frames=9):
    """Mutat néhány frame-et a videóból"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))
    
    cap.release()
    
    # Plot
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f'eye_cam.mkv - Sample Frames', fontsize=16)
    
    for i, (idx, frame) in enumerate(frames):
        r, c = divmod(i, cols)
        axes[r, c].imshow(frame)
        axes[r, c].set_title(f'Frame {idx}')
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig('eye_cam_frames.png', dpi=150)
    print("✅ Saved to: eye_cam_frames.png")
    plt.show()

if __name__ == "__main__":
    show_frames('eye_cam.mkv')
