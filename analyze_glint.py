"""
Glint detektálás vizualizáció és finomhangolás
Megvizsgáljuk hogyan működik a fényes spotok (glint) eltávolítása
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_glint_detection(frame_number=0):
    """
    Glint detektálás lépésenkénti vizualizációja
    """
    # Videó megnyitása
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Nem sikerült beolvasni a képkockát!")
        return
    
    # Szürkeárnyalatos konverzió
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Különböző threshold értékek tesztelése
    thresholds = [200, 220, 240, 250]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Glint Detektálás Analízis - Képkocka #{frame_number}', 
                 fontsize=14, fontweight='bold')
    
    # Eredeti kép
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Eredeti (RGB)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Szürkeárnyalatos
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Szürkeárnyalatos', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hisztogram
    axes[0, 2].hist(gray.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    axes[0, 2].set_title('Intenzitás Hisztogram', fontweight='bold')
    axes[0, 2].set_xlabel('Pixel érték')
    axes[0, 2].set_ylabel('Gyakoriság')
    axes[0, 2].axvline(x=240, color='r', linestyle='--', label='Threshold: 240')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Statisztikák
    stats_text = f"Min: {gray.min()}\n"
    stats_text += f"Max: {gray.max()}\n"
    stats_text += f"Átlag: {gray.mean():.1f}\n"
    stats_text += f"Std: {gray.std():.1f}\n"
    stats_text += f"Median: {np.median(gray):.1f}"
    axes[0, 3].text(0.1, 0.5, stats_text, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='center')
    axes[0, 3].set_title('Statisztikák', fontweight='bold')
    axes[0, 3].axis('off')
    
    # Threshold tesztelés
    for i, thresh in enumerate(thresholds):
        # Binary threshold
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        
        # Blob számolás
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        num_blobs = num_labels - 1  # Háttér nélkül
        
        # Vizualizáció
        axes[1, i].imshow(binary, cmap='gray')
        axes[1, i].set_title(f'Threshold: {thresh}\nBlobs: {num_blobs}', fontweight='bold')
        axes[1, i].axis('off')
        
        # Maszk vizualizáció az eredeti képen
        mask_viz = gray.copy()
        mask_viz[binary > 0] = 255
        axes[2, i].imshow(mask_viz, cmap='gray')
        axes[2, i].set_title(f'Detektált Glint (T={thresh})', fontweight='bold')
        axes[2, i].axis('off')
        
        # Blob paraméterek kiírása
        if num_blobs > 0:
            blob_info = f"Blobs: {num_blobs}\n"
            blob_info += f"Területek: {stats[1:, cv2.CC_STAT_AREA].tolist()}"
            print(f"\nThreshold {thresh}: {blob_info}")
    
    plt.tight_layout()
    output_path = Path('output/glint_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGlint analízis mentve: {output_path}")
    
    return gray


def test_glint_removal_methods(frame_number=0):
    """
    Különböző glint eltávolítási módszerek összehasonlítása
    """
    # Videó megnyitása
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Nem sikerült beolvasni a képkockát!")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Threshold alapú maszkolás
    threshold = 240
    _, glint_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Maszk finomítás (morfológiai műveletek)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glint_mask_dilated = cv2.dilate(glint_mask, kernel, iterations=1)
    
    # 2. Különböző inpainting módszerek
    inpaint_telea = cv2.inpaint(gray, glint_mask_dilated, 3, cv2.INPAINT_TELEA)
    inpaint_ns = cv2.inpaint(gray, glint_mask_dilated, 3, cv2.INPAINT_NS)
    
    # 3. Median helyettesítés
    median_replacement = gray.copy()
    median_value = np.median(gray)
    median_replacement[glint_mask > 0] = median_value
    
    # 4. Gaussian blur alapú
    blur_replacement = gray.copy()
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    blur_replacement[glint_mask_dilated > 0] = blurred[glint_mask_dilated > 0]
    
    # Vizualizáció
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Glint Eltávolítási Módszerek Összehasonlítása - Képkocka #{frame_number}',
                 fontsize=14, fontweight='bold')
    
    # Első sor: eredeti és maszkok
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Eredeti', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(glint_mask, cmap='gray')
    axes[0, 1].set_title('Glint Maszk (T=240)', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(glint_mask_dilated, cmap='gray')
    axes[0, 2].set_title('Maszk Dilatált', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gray, cmap='gray')
    axes[0, 3].contour(glint_mask_dilated, colors='red', linewidths=2)
    axes[0, 3].set_title('Detektált Régiók', fontweight='bold')
    axes[0, 3].axis('off')
    
    # Második sor: eltávolítási módszerek
    axes[1, 0].imshow(inpaint_telea, cmap='gray')
    axes[1, 0].set_title('Inpainting (Telea)', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(inpaint_ns, cmap='gray')
    axes[1, 1].set_title('Inpainting (Navier-Stokes)', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(median_replacement, cmap='gray')
    axes[1, 2].set_title(f'Median Helyettesítés ({median_value:.0f})', fontweight='bold')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(blur_replacement, cmap='gray')
    axes[1, 3].set_title('Gaussian Blur Helyettesítés', fontweight='bold')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    output_path = Path('output/glint_removal_methods.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Glint eltávolítási módszerek mentve: {output_path}")
    
    return {
        'original': gray,
        'mask': glint_mask_dilated,
        'telea': inpaint_telea,
        'ns': inpaint_ns,
        'median': median_replacement,
        'blur': blur_replacement
    }


def test_blob_detection(frame_number=0):
    """
    LoG (Laplacian of Gaussian) blob detektálás tesztelése
    """
    from skimage import feature
    
    # Videó megnyitása
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Nem sikerült beolvasni a képkockát!")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    
    # LoG blob detektálás
    blobs = feature.blob_log(gray_float, min_sigma=1, max_sigma=5, 
                            num_sigma=10, threshold=0.8)
    
    print(f"\nLoG Blob Detection: {len(blobs)} blob detektálva")
    
    # Vizualizáció
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'LoG Blob Detektálás - Képkocka #{frame_number}',
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Eredeti kép', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r * np.sqrt(2), color='red', linewidth=2, fill=False)
        axes[1].add_patch(c)
    axes[1].set_title(f'Detektált Blobs: {len(blobs)}', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = Path('output/blob_log_detection.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"LoG blob detektálás mentve: {output_path}")
    
    return blobs


def analyze_multiple_frames():
    """
    Több képkocka elemzése a glint viselkedés megértéséhez
    """
    cap = cv2.VideoCapture('eye1.mp4')
    frame_indices = [0, 10, 20, 30, 40]
    
    fig, axes = plt.subplots(len(frame_indices), 4, figsize=(16, 4*len(frame_indices)))
    fig.suptitle('Glint Analízis Több Képkockán', fontsize=14, fontweight='bold')
    
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, glint_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glint_mask_dilated = cv2.dilate(glint_mask, kernel, iterations=1)
        
        # Inpainting
        inpainted = cv2.inpaint(gray, glint_mask_dilated, 3, cv2.INPAINT_TELEA)
        
        # Vizualizáció
        axes[idx, 0].imshow(gray, cmap='gray')
        axes[idx, 0].set_title(f'Frame {frame_num}', fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(glint_mask_dilated, cmap='gray')
        axes[idx, 1].set_title('Glint Maszk', fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(inpainted, cmap='gray')
        axes[idx, 2].set_title('Inpainted', fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Különbség
        diff = cv2.absdiff(gray, inpainted)
        axes[idx, 3].imshow(diff, cmap='hot')
        axes[idx, 3].set_title('Különbség', fontweight='bold')
        axes[idx, 3].axis('off')
    
    cap.release()
    
    plt.tight_layout()
    output_path = Path('output/glint_multiple_frames.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Többképkockás glint analízis mentve: {output_path}")


def main():
    """Fő függvény - minden elemzés futtatása"""
    print("="*60)
    print("GLINT DETEKTÁLÁS ÉS ELTÁVOLÍTÁS ANALÍZIS")
    print("="*60)
    
    # Output mappa biztosítása
    Path('output').mkdir(exist_ok=True)
    
    # 1. Glint detektálás vizualizáció
    print("\n1. Glint detektálás különböző threshold értékekkel...")
    visualize_glint_detection(frame_number=0)
    
    # 2. Eltávolítási módszerek összehasonlítása
    print("\n2. Glint eltávolítási módszerek összehasonlítása...")
    test_glint_removal_methods(frame_number=0)
    
    # 3. LoG blob detektálás
    print("\n3. LoG blob detektálás tesztelése...")
    test_blob_detection(frame_number=0)
    
    # 4. Több képkocka elemzése
    print("\n4. Több képkocka elemzése...")
    analyze_multiple_frames()
    
    print("\n" + "="*60)
    print("ELEMZÉS BEFEJEZVE!")
    print("Eredmények az output/ mappában:")
    print("  - glint_analysis.png")
    print("  - glint_removal_methods.png")
    print("  - blob_log_detection.png")
    print("  - glint_multiple_frames.png")
    print("="*60)
    
    # Megjelenítés
    plt.show()


if __name__ == "__main__":
    main()
