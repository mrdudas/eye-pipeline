"""
MediaPipe Iris - Gyors Teszt
Google MediaPipe iris landmark detection tesztelése az eye1.mp4-en
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path


def test_mediapipe_iris():
    """MediaPipe Iris tesztelése"""
    print("="*60)
    print("MEDIAPIPE IRIS - TESZT")
    print("="*60)
    
    # MediaPipe Face Mesh inicializálás
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Iris landmarks engedélyezése
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Videó betöltése
    cap = cv2.VideoCapture('eye1.mp4')
    
    # Teszt frame-ek
    test_frames = [0, 10, 20, 30, 40]
    results_data = []
    
    print("\nTesztelés...")
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ Frame {frame_num}: nem sikerült beolvasni")
            continue
        
        # MediaPipe feldolgozás (RGB kell!)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            print(f"✅ Frame {frame_num}: Arc detektálva!")
            landmarks = results.multi_face_landmarks[0]
            
            # Iris landmarks (468-477)
            h, w = frame.shape[:2]
            iris_points = []
            
            for idx in range(468, 478):  # 468-477 = iris landmarks
                lm = landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                iris_points.append((x, y))
            
            results_data.append({
                'frame': frame_num,
                'detected': True,
                'iris_points': iris_points,
                'image': frame.copy()
            })
        else:
            print(f"❌ Frame {frame_num}: Nincs arc detektálva")
            results_data.append({
                'frame': frame_num,
                'detected': False,
                'image': frame.copy()
            })
    
    cap.release()
    face_mesh.close()
    
    # Eredmények kiértékelése
    print("\n" + "="*60)
    print("EREDMÉNYEK")
    print("="*60)
    
    detected_count = sum(1 for r in results_data if r['detected'])
    print(f"\nDetektált arcok: {detected_count}/{len(test_frames)}")
    
    if detected_count == 0:
        print("\n❌ MEDIAIPIPE NEM ALKALMAS")
        print("   Probléma: Close-up eye videó, nincs teljes arc")
        print("   Megoldás: DeepVOG vagy más eye-specifikus modell")
        return False, results_data
    
    if detected_count < len(test_frames):
        print(f"\n⚠️  MEDIAPIPE RÉSZBEN MŰKÖDIK ({detected_count}/{len(test_frames)})")
        print("   Néhány frame-en nem talál arcot")
    else:
        print("\n✅ MEDIAPIPE MŰKÖDIK!")
        print("   Minden tesztelt frame-en talált arcot")
    
    return detected_count > 0, results_data


def visualize_results(results_data):
    """Eredmények vizualizálása"""
    if not results_data:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MediaPipe Iris Detection - Test Results', 
                 fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results_data[:5]):
        row = idx // 3
        col = idx % 3
        
        img = result['image']
        
        if result['detected']:
            # Rajzoljuk az iris pontokat
            iris_points = result['iris_points']
            for x, y in iris_points:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            
            # Iris centrum (átlag)
            if iris_points:
                cx = int(np.mean([p[0] for p in iris_points]))
                cy = int(np.mean([p[1] for p in iris_points]))
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
                cv2.circle(img, (cx, cy), 20, (255, 0, 0), 2)
            
            title = f"Frame {result['frame']} ✓"
            color = 'green'
        else:
            title = f"Frame {result['frame']} ✗"
            color = 'red'
        
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(title, fontweight='bold', color=color)
        axes[row, col].axis('off')
    
    # Üres panel
    if len(results_data) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path('output/mediapipe_test_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVizualizáció mentve: {output_path}")
    plt.close()


def compare_with_traditional():
    """Összehasonlítás a hagyományos módszerrel"""
    print("\n" + "="*60)
    print("ÖSSZEHASONLÍTÁS: MediaPipe vs Hagyományos CV")
    print("="*60)
    
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    face_mesh.close()
    
    # Hagyományos CV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 100)
    
    # Vizualizáció
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('MediaPipe vs Hagyományos CV', fontsize=16, fontweight='bold')
    
    # Eredeti
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Eredeti Kép', fontweight='bold')
    axes[0, 0].axis('off')
    
    # MediaPipe
    mp_viz = frame.copy()
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        for idx in range(468, 478):
            lm = landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(mp_viz, (x, y), 2, (0, 255, 0), -1)
        
        status = "✅ DETEKTÁLT"
    else:
        status = "❌ NEM DETEKTÁLT"
    
    axes[0, 1].imshow(cv2.cvtColor(mp_viz, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'MediaPipe Iris\n{status}', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hagyományos - Enhanced
    axes[1, 0].imshow(enhanced, cmap='gray')
    axes[1, 0].set_title('Hagyományos: CLAHE', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Hagyományos - Edges
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title(f'Hagyományos: Canny Edges\n({np.sum(edges>0)} pixels)', 
                        fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path('output/mediapipe_vs_traditional.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Összehasonlítás mentve: {output_path}")
    plt.close()


def main():
    """Fő függvény"""
    Path('output').mkdir(exist_ok=True)
    
    # Teszt
    success, results = test_mediapipe_iris()
    
    # Vizualizáció
    if results:
        visualize_results(results)
    
    # Összehasonlítás
    compare_with_traditional()
    
    print("\n" + "="*60)
    if success:
        print("✅ MEDIAPIPE MŰKÖDIK AZ EYE1.MP4-EN!")
        print("\nKövetkező lépések:")
        print("1. Teljes integráció a pipeline-ba")
        print("2. Teljes videó feldolgozása")
        print("3. Pontosság mérése")
    else:
        print("❌ MEDIAPIPE NEM ALKALMAS")
        print("\nKövetkező lépések:")
        print("1. DeepVOG telepítése")
        print("2. Eye-specifikus modell használata")
    print("="*60)


if __name__ == "__main__":
    main()
