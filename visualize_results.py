"""
Eredmények vizualizációja és statisztikák
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(json_path='output/pupil_data.json'):
    """Eredmények betöltése"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_statistics(data):
    """Statisztikai ábrák készítése"""
    detections = data['detections']
    
    # Adatok kinyerése
    frames = [d['frame'] for d in detections]
    diameters = [d['axes'][1] for d in detections]  # minor axis = diameter
    confidence = [d['confidence'] for d in detections]
    center_x = [d['center'][0] for d in detections]
    center_y = [d['center'][1] for d in detections]
    
    # 4 panel ábra
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pupilla Detektálás - Statisztikák', fontsize=16, fontweight='bold')
    
    # 1. Pupilla átmérő
    ax1 = axes[0, 0]
    ax1.plot(frames, diameters, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Képkocka', fontsize=11)
    ax1.set_ylabel('Átmérő (pixel)', fontsize=11)
    ax1.set_title('Pupilla Átmérő Időben', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Átlag: {np.mean(diameters):.2f} px\nStd: {np.std(diameters):.2f} px',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Konfidencia
    ax2 = axes[0, 1]
    ax2.plot(frames, confidence, 'g-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Képkocka', fontsize=11)
    ax2.set_ylabel('Konfidencia', fontsize=11)
    ax2.set_title('Detektálási Konfidencia', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    ax2.text(0.02, 0.98, f'Átlag: {np.mean(confidence):.3f}\nMin: {np.min(confidence):.3f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 3. Pupilla központ mozgása
    ax3 = axes[1, 0]
    scatter = ax3.scatter(center_x, center_y, c=frames, cmap='viridis', 
                         s=20, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax3.set_xlabel('X pozíció (pixel)', fontsize=11)
    ax3.set_ylabel('Y pozíció (pixel)', fontsize=11)
    ax3.set_title('Pupilla Középpont Trajektória', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Y tengely megfordítása (kép koordináták)
    plt.colorbar(scatter, ax=ax3, label='Képkocka')
    
    # 4. Átmérő eloszlás (hisztogram)
    ax4 = axes[1, 1]
    ax4.hist(diameters, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Átmérő (pixel)', fontsize=11)
    ax4.set_ylabel('Gyakoriság', fontsize=11)
    ax4.set_title('Pupilla Átmérő Eloszlás', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axvline(np.mean(diameters), color='red', linestyle='--', 
                linewidth=2, label=f'Átlag: {np.mean(diameters):.1f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Mentés
    output_path = Path('output/statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Statisztikák mentve: {output_path}")
    
    return fig


def print_summary(data):
    """Összefoglaló statisztikák kiírása"""
    detections = data['detections']
    video_info = data['video_info']
    
    print("\n" + "="*60)
    print("PUPILLA DETEKTÁLÁS - ÖSSZEFOGLALÓ")
    print("="*60)
    
    print("\nVideó Információk:")
    print(f"  Fájl: {video_info['path']}")
    print(f"  Felbontás: {video_info['resolution'][0]}x{video_info['resolution'][1]}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Összes képkocka: {video_info['total_frames']}")
    
    print(f"\nDetektálás:")
    print(f"  Feldolgozott képkockák: {len(detections)}")
    print(f"  Sikeres detektálás: {len(detections)} ({100:.1f}%)")
    
    if detections:
        diameters = [d['axes'][1] for d in detections]
        confidence = [d['confidence'] for d in detections]
        
        print(f"\nPupilla Átmérő (pixel):")
        print(f"  Átlag: {np.mean(diameters):.2f}")
        print(f"  Std: {np.std(diameters):.2f}")
        print(f"  Min: {np.min(diameters):.2f}")
        print(f"  Max: {np.max(diameters):.2f}")
        
        print(f"\nKonfidencia:")
        print(f"  Átlag: {np.mean(confidence):.3f}")
        print(f"  Min: {np.min(confidence):.3f}")
        print(f"  Max: {np.max(confidence):.3f}")
        
        # Mozgás analízis
        center_x = [d['center'][0] for d in detections]
        center_y = [d['center'][1] for d in detections]
        
        movement_x = np.std(center_x)
        movement_y = np.std(center_y)
        
        print(f"\nPupilla Mozgás:")
        print(f"  X irány variancia: {movement_x:.2f} pixel")
        print(f"  Y irány variancia: {movement_y:.2f} pixel")
    
    print("\n" + "="*60)


def main():
    """Fő függvény"""
    # Eredmények betöltése
    data = load_results()
    
    # Összefoglaló
    print_summary(data)
    
    # Vizualizáció
    if len(data['detections']) > 0:
        plot_statistics(data)
        print("\nVizualizáció kész! Megnyitás...")
        plt.show()
    else:
        print("\nNincs megjeleníthető adat!")


if __name__ == "__main__":
    main()
