#!/usr/bin/env python3
"""
Test RITnet on eye1.mp4
"""

import sys
sys.path.append('./RITnet')

import cv2
import torch
import numpy as np
from models import model_dict
import matplotlib.pyplot as plt

def preprocess_image(image):
    """
    Preprocess image according to RITnet requirements:
    1. Convert to grayscale
    2. Gamma correction (0.8)
    3. CLAHE (clipLimit=1.5, tileGridSize=(8,8))
    4. Normalization (mean=0.5, std=0.5)
    5. Resize to 640x400
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Gamma correction
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Resize to 640x400
    resized = cv2.resize(gray, (640, 400))
    
    # Normalize
    normalized = (resized / 255.0 - 0.5) / 0.5
    
    # Convert to tensor: [1, 1, 400, 640]
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
    
    return tensor, gray

def postprocess_mask(output, original_size):
    """
    Convert RITnet output to segmentation mask
    Output shape: [1, 4, 400, 640]
    Classes: 0=background, 1=sclera, 2=iris, 3=pupil
    """
    # Get class predictions
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize back to original size
    pred_resized = cv2.resize(pred.astype(np.uint8), 
                             (original_size[1], original_size[0]),
                             interpolation=cv2.INTER_NEAREST)
    
    return pred_resized

def extract_eyelid_boundaries(mask, pupil_center):
    """
    Extract upper and lower eyelid boundaries from segmentation mask
    """
    # Find sclera, iris, and pupil regions (non-background)
    eye_region = (mask > 0).astype(np.uint8) * 255
    
    # Find contours of eye region
    contours, _ = cv2.findContours(eye_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Extract top and bottom points along horizontal line through pupil
    cx, cy = pupil_center
    
    # Find topmost and bottommost points
    topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
    
    return topmost, bottommost

def visualize_results(original, mask, pupil_center, eyelid_top, eyelid_bottom):
    """
    Visualize segmentation results with eyelid boundaries
    """
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 0, 0]    # Sclera: Red
    colored_mask[mask == 2] = [0, 255, 0]    # Iris: Green
    colored_mask[mask == 3] = [0, 0, 255]    # Pupil: Blue
    
    # Overlay on original
    overlay = cv2.addWeighted(original, 0.6, colored_mask, 0.4, 0)
    
    # Draw pupil center
    if pupil_center:
        cv2.circle(overlay, pupil_center, 5, (255, 255, 0), -1)
    
    # Draw eyelid boundaries
    if eyelid_top:
        cv2.circle(overlay, eyelid_top, 5, (0, 255, 255), -1)
        cv2.putText(overlay, 'Upper', eyelid_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    if eyelid_bottom:
        cv2.circle(overlay, eyelid_bottom, 5, (0, 255, 255), -1)
        cv2.putText(overlay, 'Lower', eyelid_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw horizontal axis if both boundaries found
    if eyelid_top and eyelid_bottom:
        cv2.line(overlay, eyelid_top, eyelid_bottom, (0, 255, 255), 2)
    
    return overlay

def main():
    print("Loading RITnet model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model_dict['densenet']
    model.load_state_dict(torch.load('./RITnet/best_model.pkl', map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load video
    print("\nLoading video...")
    cap = cv2.VideoCapture('eye1.mp4')
    
    if not cap.isOpened():
        print("ERROR: Cannot open video!")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Test on frame 1000
    test_frame_idx = 1000
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Cannot read frame!")
        cap.release()
        return
    
    print(f"\nProcessing frame {test_frame_idx}...")
    print(f"Frame shape: {frame.shape}")
    
    # Convert to grayscale for visualization
    if len(frame.shape) == 3:
        gray_vis = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_vis = frame.copy()
    
    # Preprocess
    input_tensor, preprocessed = preprocess_image(frame)
    input_tensor = input_tensor.to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    
    # Postprocess
    mask = postprocess_mask(output, frame.shape[:2])
    
    # Find pupil center
    pupil_mask = (mask == 3).astype(np.uint8) * 255
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pupil_center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            pupil_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            print(f"Pupil center: {pupil_center}")
    
    # Extract eyelid boundaries
    eyelid_top, eyelid_bottom = extract_eyelid_boundaries(mask, pupil_center if pupil_center else (200, 200))
    
    if eyelid_top:
        print(f"Upper eyelid: {eyelid_top}")
    if eyelid_bottom:
        print(f"Lower eyelid: {eyelid_bottom}")
    
    # Visualize
    print("\nCreating visualization...")
    result = visualize_results(cv2.cvtColor(gray_vis, cv2.COLOR_GRAY2BGR), mask, 
                               pupil_center, eyelid_top, eyelid_bottom)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(gray_vis, cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # Preprocessed
    axes[0, 1].imshow(preprocessed, cmap='gray')
    axes[0, 1].set_title('Preprocessed (Gamma + CLAHE)')
    axes[0, 1].axis('off')
    
    # Segmentation mask
    axes[0, 2].imshow(mask, cmap='tab10')
    axes[0, 2].set_title('Segmentation Mask\n0=bg, 1=sclera, 2=iris, 3=pupil')
    axes[0, 2].axis('off')
    
    # Individual masks
    axes[1, 0].imshow((mask == 1).astype(np.uint8) * 255, cmap='gray')
    axes[1, 0].set_title('Sclera (class 1)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow((mask == 2).astype(np.uint8) * 255, cmap='gray')
    axes[1, 1].set_title('Iris (class 2)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(result)
    axes[1, 2].set_title('Final Result with Eyelid Boundaries')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('ritnet_test_result.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to: ritnet_test_result.png")
    
    plt.show()
    
    # Save result video frame
    cv2.imwrite('ritnet_frame_result.png', result)
    print("Frame result saved to: ritnet_frame_result.png")
    
    cap.release()
    print("\nDone!")

if __name__ == "__main__":
    main()
