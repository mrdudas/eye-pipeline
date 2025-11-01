"""
MediaPipe Iris Landmarks Részletes Vizualizáció
Megnézzük, hogy pontosan hol vannak a landmark pontok
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_mediapipe_landmarks():
    """MediaPipe landmark pontok részletes vizualizációja"""
    
    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Videó betöltése
    cap = cv2.VideoCapture('eye1.mp4')
    
    # Első frame
    ret, frame = cap.read()
    if not ret:
        print("Nem sikerült a frame betöltése")
        return
    
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Feldolgozás
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        print("Nem található arc/iris")
        return
    
    landmarks = results.multi_face_landmarks[0]
    
    # Iris landmark indices
    # MediaPipe Face Mesh 468-477: iris landmarks
    left_iris = list(range(468, 473))   # 468-472: bal iris
    right_iris = list(range(473, 478))  # 473-477: jobb iris
    
    print("MediaPipe Iris Landmarks:")
    print(f"  Bal iris: {left_iris}")
    print(f"  Jobb iris: {right_iris}")
    
    # Vizualizáció
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MediaPipe Iris Landmarks Részletes Elemzés', fontsize=16, fontweight='bold')
    
    # 1. Eredeti frame
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title('Eredeti Frame')
    ax.axis('off')
    
    # 2. Összes iris pont
    frame_all = frame.copy()
    all_iris_points = []
    for idx in range(468, 478):
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        all_iris_points.append([x, y])
        cv2.circle(frame_all, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(frame_all, str(idx), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Centrum (átlag)
    all_iris_points = np.array(all_iris_points)
    center = np.mean(all_iris_points, axis=0).astype(int)
    cv2.circle(frame_all, tuple(center), 5, (0, 0, 255), -1)
    cv2.putText(frame_all, "Center", (center[0]+10, center[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    ax = axes[0, 1]
    ax.imshow(cv2.cvtColor(frame_all, cv2.COLOR_BGR2RGB))
    ax.set_title('Összes Iris Landmark (468-477)')
    ax.axis('off')
    
    # 3. Bal iris
    frame_left = frame.copy()
    left_points = []
    for idx in left_iris:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        left_points.append([x, y])
        cv2.circle(frame_left, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(frame_left, str(idx), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if left_points:
        left_points = np.array(left_points)
        center_left = np.mean(left_points, axis=0).astype(int)
        cv2.circle(frame_left, tuple(center_left), 5, (0, 0, 255), -1)
    
    ax = axes[0, 2]
    ax.imshow(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
    ax.set_title('Bal Iris (468-472)')
    ax.axis('off')
    
    # 4. Jobb iris
    frame_right = frame.copy()
    right_points = []
    for idx in right_iris:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        right_points.append([x, y])
        cv2.circle(frame_right, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(frame_right, str(idx), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if right_points:
        right_points = np.array(right_points)
        center_right = np.mean(right_points, axis=0).astype(int)
        cv2.circle(frame_right, tuple(center_right), 5, (0, 0, 255), -1)
    
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))
    ax.set_title('Jobb Iris (473-477)')
    ax.axis('off')
    
    # 5. Grayscale frame pupilla-val
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Küszöbölés a pupilla megtalálásához
    _, binary = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    ax = axes[1, 1]
    ax.imshow(binary, cmap='gray')
    ax.set_title('Pupilla (küszöbölés)')
    ax.axis('off')
    
    # 6. Overlay: MediaPipe vs valós pupilla
    frame_overlay = frame.copy()
    
    # MediaPipe centrum
    cv2.circle(frame_overlay, tuple(center), 5, (0, 255, 0), -1)
    cv2.putText(frame_overlay, "MediaPipe", (center[0]+10, center[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Pupilla detektálás (hagyományos CV)
    # Morfológiai műveletek
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)
    
    # Kontúrok
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Legnagyobb kontúr = pupilla
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Ellipszis illesztés
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            pupil_center = tuple(map(int, ellipse[0]))
            
            cv2.ellipse(frame_overlay, ellipse, (255, 0, 0), 2)
            cv2.circle(frame_overlay, pupil_center, 5, (255, 0, 0), -1)
            cv2.putText(frame_overlay, "Pupilla (CV)", (pupil_center[0]+10, pupil_center[1]+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Távolság
            distance = np.sqrt((center[0] - pupil_center[0])**2 + (center[1] - pupil_center[1])**2)
            
            # Vonal a két pont között
            cv2.line(frame_overlay, tuple(center), pupil_center, (255, 255, 0), 2)
            mid_point = ((center[0] + pupil_center[0])//2, (center[1] + pupil_center[1])//2)
            cv2.putText(frame_overlay, f"{distance:.1f}px", mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    ax = axes[1, 2]
    ax.imshow(cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB))
    ax.set_title('MediaPipe vs Valós Pupilla')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/mediapipe_landmark_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Vizualizáció mentve: output/mediapipe_landmark_analysis.png")
    plt.show()
    
    cap.release()
    face_mesh.close()
    
    print("\n" + "=" * 70)
    print("📊 KÖVETKEZTETÉS:")
    print("=" * 70)
    print("MediaPipe IRIS landmarks-t detektál, nem PUPILLA-t!")
    print("Az iris nagyobb, mint a pupilla, más a centruma.")
    print("\n💡 MEGOLDÁS:")
    print("1. Használd a hagyományos CV-t a pupilla detektáláshoz")
    print("2. MediaPipe-ot csak iris tracking-hez használd")
    print("3. Kombináld a kettőt: MediaPipe stabilizáció + CV pupilla")
    print("=" * 70)


if __name__ == "__main__":
    visualize_mediapipe_landmarks()
