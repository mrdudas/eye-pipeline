"""
CLAHE Gyors Analízis - Csak fájlba mentés, nincs megjelenítés
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display backend
import matplotlib.pyplot as plt
from pathlib import Path


def quick_clahe_comparison():
    """Gyors összehasonlítás mentéssel"""
    print("Videó betöltése...")
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    cap.release()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Különböző módszerek
    print("Preprocessing módszerek alkalmazása...")
    hist_eq = cv2.equalizeHist(gray)
    clahe_1 = cv2.createCLAHE(1.0, (8,8)).apply(gray)
    clahe_2 = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    clahe_3 = cv2.createCLAHE(3.0, (8,8)).apply(gray)
    
    # Edge detection
    edges_orig = cv2.Canny(gray, 30, 100)
    edges_he = cv2.Canny(hist_eq, 30, 100)
    edges_c1 = cv2.Canny(clahe_1, 30, 100)
    edges_c2 = cv2.Canny(clahe_2, 30, 100)
    edges_c3 = cv2.Canny(clahe_3, 30, 100)
    
    # Vizualizáció 1: Preprocesszált képek
    print("Vizualizációk készítése...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Módszerek Összehasonlítása', fontsize=16, fontweight='bold')
    
    images = [
        (gray, 'Eredeti'),
        (hist_eq, 'Histogram EQ'),
        (clahe_1, 'CLAHE (1.0, 8x8)'),
        (clahe_2, 'CLAHE (2.0, 8x8) ✓'),
        (clahe_3, 'CLAHE (3.0, 8x8)'),
    ]
    
    for idx, (img, title) in enumerate(images):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].axis('off')
        
        # Stats
        stats = f'μ={img.mean():.1f}, σ={img.std():.1f}'
        axes[row, col].text(0.5, 0.05, stats, transform=axes[row, col].transAxes,
                          ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    axes[1, 2].axis('off')  # Üres
    
    plt.tight_layout()
    out1 = Path('output/clahe_quick_comparison.png')
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"✓ Mentve: {out1}")
    
    # Vizualizáció 2: Edge detection
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Edge Detection - Preprocessing Hatása', fontsize=16, fontweight='bold')
    
    edges = [
        (edges_orig, 'Eredeti'),
        (edges_he, 'Hist EQ'),
        (edges_c1, 'CLAHE 1.0'),
        (edges_c2, 'CLAHE 2.0 ✓'),
        (edges_c3, 'CLAHE 3.0'),
    ]
    
    for idx, (edge, title) in enumerate(edges):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(edge, cmap='gray')
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].axis('off')
        
        # Edge pixel count
        edge_pixels = np.sum(edge > 0)
        axes[row, col].text(0.5, 0.05, f'Edges: {edge_pixels}px',
                          transform=axes[row, col].transAxes,
                          ha='center', bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out2 = Path('output/clahe_edge_detection.png')
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"✓ Mentve: {out2}")
    
    # Statisztikák
    print("\n" + "="*60)
    print("STATISZTIKÁK")
    print("="*60)
    print(f"\n{'Módszer':<20} {'Mean':<10} {'Std':<10} {'Edges':<10}")
    print("-"*60)
    
    stats_data = [
        ('Eredeti', gray, edges_orig),
        ('Histogram EQ', hist_eq, edges_he),
        ('CLAHE (1.0)', clahe_1, edges_c1),
        ('CLAHE (2.0)', clahe_2, edges_c2),
        ('CLAHE (3.0)', clahe_3, edges_c3),
    ]
    
    for name, img, edge in stats_data:
        print(f"{name:<20} {img.mean():<10.1f} {img.std():<10.1f} {np.sum(edge>0):<10}")
    
    print("\n✅ AJÁNLÁS: CLAHE (2.0, 8x8)")
    print("   - Jó kontraszt javítás")
    print("   - Optimális edge detection")
    print("   - Nem túl agresszív")
    print("="*60)


def clahe_parameter_sweep():
    """CLAHE paraméterek végigpróbálása"""
    print("\nCLAHE Paraméter Sweep...")
    cap = cv2.VideoCapture('eye1.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    cap.release()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clip_limits = [1.0, 2.0, 3.0, 5.0]
    tile_sizes = [(4,4), (8,8), (16,16)]
    
    fig, axes = plt.subplots(len(clip_limits), len(tile_sizes), figsize=(12, 12))
    fig.suptitle('CLAHE Parameter Grid', fontsize=14, fontweight='bold')
    
    for i, clip in enumerate(clip_limits):
        for j, tile in enumerate(tile_sizes):
            clahe = cv2.createCLAHE(clip, tile)
            result = clahe.apply(gray)
            
            axes[i,j].imshow(result, cmap='gray')
            axes[i,j].set_title(f'C={clip}, T={tile}', fontsize=9)
            axes[i,j].axis('off')
            
            # Kontraszt
            axes[i,j].text(0.5, 0.05, f'σ={result.std():.1f}',
                         transform=axes[i,j].transAxes, ha='center', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    out = Path('output/clahe_parameter_sweep.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✓ Mentve: {out}")


def main():
    print("="*60)
    print("CLAHE GYORS ANALÍZIS")
    print("="*60)
    
    Path('output').mkdir(exist_ok=True)
    
    quick_clahe_comparison()
    clahe_parameter_sweep()
    
    print("\n✅ ELEMZÉS BEFEJEZVE!")
    print("\nKészült:")
    print("  - output/clahe_quick_comparison.png")
    print("  - output/clahe_edge_detection.png")
    print("  - output/clahe_parameter_sweep.png")
    print("="*60)


if __name__ == "__main__":
    main()
