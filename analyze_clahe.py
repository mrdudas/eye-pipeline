"""
Histogram Equalization és CLAHE Analízis
Kontraszt javítás módszerek összehasonlítása
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_histogram_methods(frame_number=0):
    """
    Különböző hisztogram módszerek összehasonlítása
    """
    # Videó betöltése
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Nem sikerült betölteni a képkockát!")
        return
    
    # Szürkeárnyalatos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Eredeti
    original = gray.copy()
    
    # 2. Globális Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)
    
    # 3. CLAHE különböző paraméterekkel
    clahe_params = [
        (1.0, (8, 8), "CLAHE (1.0, 8x8)"),
        (2.0, (8, 8), "CLAHE (2.0, 8x8) - DEFAULT"),
        (3.0, (8, 8), "CLAHE (3.0, 8x8)"),
        (2.0, (4, 4), "CLAHE (2.0, 4x4)"),
        (2.0, (16, 16), "CLAHE (2.0, 16x16)"),
    ]
    
    clahe_results = []
    for clip_limit, tile_size, label in clahe_params:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        result = clahe.apply(gray)
        clahe_results.append((result, label, clip_limit, tile_size))
    
    # Vizualizáció
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'Histogram Equalization vs CLAHE - Frame #{frame_number}',
                 fontsize=16, fontweight='bold')
    
    # Első sor: Eredeti + hisztogramja + Globális HE + hisztogramja
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Eredeti', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].hist(original.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[0, 1].set_title('Eredeti Hisztogram', fontweight='bold')
    axes[0, 1].set_xlabel('Pixel érték')
    axes[0, 1].set_ylabel('Gyakoriság')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].imshow(hist_eq, cmap='gray')
    axes[0, 2].set_title('Globális Histogram EQ', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].hist(hist_eq.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axes[0, 3].set_title('Global HE Hisztogram', fontweight='bold')
    axes[0, 3].set_xlabel('Pixel érték')
    axes[0, 3].set_ylabel('Gyakoriság')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Következő sorok: CLAHE variációk
    for idx, (result, label, clip, tile) in enumerate(clahe_results):
        row = (idx // 2) + 1
        col = (idx % 2) * 2
        
        # Kép
        axes[row, col].imshow(result, cmap='gray')
        axes[row, col].set_title(label, fontweight='bold')
        axes[row, col].axis('off')
        
        # Hisztogram
        axes[row, col + 1].hist(result.ravel(), bins=256, range=(0, 256), 
                               color='orange', alpha=0.7)
        axes[row, col + 1].set_title(f'{label} - Histogram', fontweight='bold')
        axes[row, col + 1].set_xlabel('Pixel érték')
        axes[row, col + 1].set_ylabel('Gyakoriság')
        axes[row, col + 1].grid(True, alpha=0.3)
        
        # Statisztikák hozzáadása
        stats_text = f'Mean: {result.mean():.1f}\nStd: {result.std():.1f}'
        axes[row, col + 1].text(0.7, 0.95, stats_text, 
                               transform=axes[row, col + 1].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = Path('output/histogram_clahe_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Hisztogram összehasonlítás mentve: {output_path}")
    
    return original, hist_eq, clahe_results


def detailed_clahe_analysis(frame_number=0):
    """
    CLAHE részletes elemzése különböző paraméterekkel
    """
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE paraméter grid
    clip_limits = [1.0, 2.0, 3.0, 5.0]
    tile_sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    
    fig, axes = plt.subplots(len(clip_limits), len(tile_sizes), 
                            figsize=(16, 12))
    fig.suptitle('CLAHE Parameter Grid - Clip Limit vs Tile Size',
                 fontsize=16, fontweight='bold')
    
    for i, clip_limit in enumerate(clip_limits):
        for j, tile_size in enumerate(tile_sizes):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            result = clahe.apply(gray)
            
            axes[i, j].imshow(result, cmap='gray')
            axes[i, j].set_title(f'Clip:{clip_limit}, Tile:{tile_size}',
                                fontsize=9)
            axes[i, j].axis('off')
            
            # Kontraszt mérés
            contrast = result.std()
            axes[i, j].text(0.5, 0.05, f'σ={contrast:.1f}',
                          transform=axes[i, j].transAxes,
                          ha='center', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Címkék a tengelyekre
    fig.text(0.5, 0.02, 'Tile Grid Size →', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Clip Limit →', va='center', rotation='vertical', 
            fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    output_path = Path('output/clahe_parameter_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"CLAHE paraméter grid mentve: {output_path}")


def edge_detection_comparison(frame_number=0):
    """
    Összehasonlítás: hogyan befolyásolja a preprocessing az edge detection-t
    """
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Különböző preprocessing
    hist_eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)
    
    # Edge detection mindegyiken
    canny_params = (30, 100)
    
    edges_original = cv2.Canny(gray, *canny_params)
    edges_hist_eq = cv2.Canny(hist_eq, *canny_params)
    edges_clahe = cv2.Canny(clahe_result, *canny_params)
    
    # Vizualizáció
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Hatása az Edge Detection-re',
                 fontsize=16, fontweight='bold')
    
    # Első sor: preprocesszált képek
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Eredeti', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(hist_eq, cmap='gray')
    axes[0, 1].set_title('Histogram Equalization', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(clahe_result, cmap='gray')
    axes[0, 2].set_title('CLAHE (2.0, 8x8)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Második sor: edges
    axes[1, 0].imshow(edges_original, cmap='gray')
    axes[1, 0].set_title(f'Canny Edges\nDetected: {np.sum(edges_original>0)}px', 
                        fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges_hist_eq, cmap='gray')
    axes[1, 1].set_title(f'Canny Edges (HE)\nDetected: {np.sum(edges_hist_eq>0)}px',
                        fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(edges_clahe, cmap='gray')
    axes[1, 2].set_title(f'Canny Edges (CLAHE)\nDetected: {np.sum(edges_clahe>0)}px',
                        fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path('output/preprocessing_edge_detection.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Edge detection összehasonlítás mentve: {output_path}")


def pupil_region_focus(frame_number=0):
    """
    Pupilla régió vizsgálata különböző preprocessing módszerekkel
    """
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing módszerek
    methods = {
        'Eredeti': gray,
        'Hist EQ': cv2.equalizeHist(gray),
        'CLAHE (1.0)': cv2.createCLAHE(1.0, (8,8)).apply(gray),
        'CLAHE (2.0)': cv2.createCLAHE(2.0, (8,8)).apply(gray),
        'CLAHE (3.0)': cv2.createCLAHE(3.0, (8,8)).apply(gray),
        'CLAHE (5.0)': cv2.createCLAHE(5.0, (8,8)).apply(gray),
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Módszerek - Teljes Kép és Pupilla Régió Hisztogramok',
                 fontsize=16, fontweight='bold')
    
    # Pupilla régió közelítése (középső rész, általában itt van a pupilla)
    h, w = gray.shape
    roi = (slice(h//4, 3*h//4), slice(w//4, 3*w//4))
    
    for idx, (name, img) in enumerate(methods.items()):
        row = idx // 3
        col = idx % 3
        
        # Kép megjelenítése ROI jelöléssel
        axes[row, col].imshow(img, cmap='gray')
        
        # ROI téglalap
        rect_y, rect_x = h//4, w//4
        rect_h, rect_w = h//2, w//2
        from matplotlib.patches import Rectangle
        rect = Rectangle((rect_x, rect_y), rect_w, rect_h,
                        linewidth=2, edgecolor='red', facecolor='none')
        axes[row, col].add_patch(rect)
        
        axes[row, col].set_title(name, fontweight='bold')
        axes[row, col].axis('off')
        
        # ROI hisztogram hozzáadása
        roi_region = img[roi]
        axes[row, col].text(0.02, 0.98, 
                          f'ROI μ={roi_region.mean():.1f}\nROI σ={roi_region.std():.1f}',
                          transform=axes[row, col].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                          fontsize=9)
    
    plt.tight_layout()
    output_path = Path('output/preprocessing_pupil_focus.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Pupilla régió összehasonlítás mentve: {output_path}")


def multiple_frames_clahe_test():
    """
    Több képkocka tesztelése a legjobb CLAHE paraméterek megtalálásához
    """
    cap = cv2.VideoCapture('eye1.mp4')
    frame_numbers = [0, 10, 20, 30, 40, 50]
    
    # CLAHE beállítások tesztelése
    configs = [
        (None, None, 'Eredeti'),
        (1.0, (8, 8), 'CLAHE 1.0'),
        (2.0, (8, 8), 'CLAHE 2.0 ✓'),
        (3.0, (8, 8), 'CLAHE 3.0'),
    ]
    
    fig, axes = plt.subplots(len(frame_numbers), len(configs), 
                            figsize=(16, 3*len(frame_numbers)))
    fig.suptitle('CLAHE Tesztelés Több Képkockán',
                 fontsize=16, fontweight='bold')
    
    for i, frame_num in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for j, (clip, tile, label) in enumerate(configs):
            if clip is None:
                result = gray
            else:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                result = clahe.apply(gray)
            
            axes[i, j].imshow(result, cmap='gray')
            title = f'{label}' if i == 0 else ''
            axes[i, j].set_title(title, fontweight='bold')
            axes[i, j].set_ylabel(f'Frame {frame_num}' if j == 0 else '', 
                                 fontweight='bold')
            axes[i, j].axis('off')
            
            # Kontraszt
            contrast = result.std()
            axes[i, j].text(0.5, 0.05, f'σ={contrast:.1f}',
                          transform=axes[i, j].transAxes,
                          ha='center', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    cap.release()
    
    plt.tight_layout()
    output_path = Path('output/clahe_multiple_frames.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Többképkockás CLAHE teszt mentve: {output_path}")


def print_statistics_summary():
    """Összefoglaló statisztikák"""
    print("\n" + "="*60)
    print("HISTOGRAM EQUALIZATION ÉS CLAHE - ÖSSZEFOGLALÓ")
    print("="*60)
    
    print("\n📊 Módszerek:")
    print("\n1. GLOBÁLIS HISTOGRAM EQUALIZATION")
    print("   ✅ Előnyök:")
    print("      - Teljes dinamikus tartomány kihasználása")
    print("      - Egyszerű és gyors")
    print("   ❌ Hátrányok:")
    print("      - Túl agresszív, lokális részletek elvesztése")
    print("      - Zaj felerősítése")
    print("      - Nem jó változó megvilágítású képekhez")
    
    print("\n2. CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("   ✅ Előnyök:")
    print("      - Lokális kontraszt javítás")
    print("      - Zaj kontroll (clip limit)")
    print("      - Jobb pupilla perem megőrzés")
    print("      - Adaptív tile-ok → változó megvilágítás kezelése")
    print("   ❌ Hátrányok:")
    print("      - Lassabb mint globális HE")
    print("      - Paraméter hangolás szükséges")
    
    print("\n⚙️ OPTIMÁLIS CLAHE BEÁLLÍTÁSOK:")
    print("   • Clip Limit: 2.0 (1.0 = konzervatív, 3.0+ = agresszív)")
    print("   • Tile Grid: 8x8 (4x4 = finomabb, 16x16 = durva)")
    print("   • Ajánlott: clipLimit=2.0, tileGridSize=(8,8)")
    
    print("\n🎯 PUPILLA DETEKTÁLÁSHOZ:")
    print("   ✅ CLAHE (2.0, 8x8) - AJÁNLOTT")
    print("      - Jobb edge detection")
    print("      - Pupilla kontúr élesebb")
    print("      - Kevesebb false edge")
    
    print("\n" + "="*60)


def main():
    """Fő függvény - minden elemzés"""
    print("="*60)
    print("HISTOGRAM EQUALIZATION ÉS CLAHE ANALÍZIS")
    print("="*60)
    
    Path('output').mkdir(exist_ok=True)
    
    print("\n1. Hisztogram módszerek összehasonlítása...")
    analyze_histogram_methods(frame_number=10)
    
    print("\n2. CLAHE paraméter grid...")
    detailed_clahe_analysis(frame_number=10)
    
    print("\n3. Edge detection összehasonlítás...")
    edge_detection_comparison(frame_number=10)
    
    print("\n4. Pupilla régió fókusz...")
    pupil_region_focus(frame_number=10)
    
    print("\n5. Többképkockás CLAHE teszt...")
    multiple_frames_clahe_test()
    
    print_statistics_summary()
    
    print("\n✅ ELEMZÉS BEFEJEZVE!")
    print("\nGenerált fájlok az output/ mappában:")
    print("  - histogram_clahe_comparison.png")
    print("  - clahe_parameter_grid.png")
    print("  - preprocessing_edge_detection.png")
    print("  - preprocessing_pupil_focus.png")
    print("  - clahe_multiple_frames.png")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    main()
