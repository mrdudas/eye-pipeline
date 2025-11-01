"""
Részletes elemzés: MediaPipe fluktuáció időben
Megnézi az egész 50 frame-t frame-by-frame
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def analyze_temporal_fluctuation():
    """Időbeli fluktuáció elemzés"""
    
    # Adatok betöltése
    with open('output/ai_pupil_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detections = data['detections']
    
    frames = [d['frame'] for d in detections]
    diameters = [d['diameter'] for d in detections]
    centers_x = [d['center'][0] for d in detections]
    centers_y = [d['center'][1] for d in detections]
    confidences = [d['confidence'] for d in detections]
    
    print("=" * 70)
    print("🔍 MEDIAPIPE FLUKTUÁCIÓ ELEMZÉS")
    print("=" * 70)
    
    print(f"\nÖsszesítés ({len(detections)} frame):")
    print(f"  Átmérő: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} px")
    print(f"  Min-Max: {np.min(diameters):.2f} - {np.max(diameters):.2f} px")
    print(f"  Tartomány: {np.max(diameters) - np.min(diameters):.2f} px")
    
    # Frame-to-frame változások
    diameter_diffs = np.abs(np.diff(diameters))
    print(f"\nFrame-to-frame változások:")
    print(f"  Átlag: {np.mean(diameter_diffs):.2f} px")
    print(f"  Max: {np.max(diameter_diffs):.2f} px")
    print(f"  > 5px változások: {np.sum(diameter_diffs > 5)}/{len(diameter_diffs)}")
    print(f"  > 10px változások: {np.sum(diameter_diffs > 10)}/{len(diameter_diffs)}")
    
    # Legnagyobb ugrások
    print(f"\n🔴 Top 5 legnagyobb átmérő ugrás:")
    sorted_indices = np.argsort(diameter_diffs)[::-1][:5]
    for i, idx in enumerate(sorted_indices, 1):
        print(f"  {i}. Frame {idx} → {idx+1}: "
              f"{diameters[idx]:.1f} → {diameters[idx+1]:.1f} px "
              f"(Δ={diameter_diffs[idx]:.1f} px)")
    
    # Első 10 frame részletes
    print(f"\n📊 Első 10 frame részletes:")
    print(f"{'Frame':<8} {'Átmérő':<10} {'Változás':<10} {'Center X':<10} {'Center Y':<10} {'Conf':<8}")
    print("-" * 70)
    for i in range(min(10, len(diameters))):
        change = diameter_diffs[i-1] if i > 0 else 0
        print(f"{frames[i]:<8} {diameters[i]:<10.2f} {change:<10.2f} "
              f"{centers_x[i]:<10.2f} {centers_y[i]:<10.2f} {confidences[i]:<8.3f}")
    
    # Vizualizáció
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('MediaPipe Időbeli Fluktuáció Elemzés', fontsize=14, fontweight='bold')
    
    # 1. Átmérő időben
    ax = axes[0]
    ax.plot(frames, diameters, 'b-o', linewidth=2, markersize=4)
    ax.axhline(np.mean(diameters), color='red', linestyle='--', label=f'Átlag: {np.mean(diameters):.1f}px')
    ax.fill_between(frames, 
                     np.mean(diameters) - np.std(diameters), 
                     np.mean(diameters) + np.std(diameters), 
                     alpha=0.2, color='red', label=f'±1σ: {np.std(diameters):.1f}px')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő (pixel)')
    ax.set_title('Pupilla Átmérő Időben')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Frame-to-frame változás
    ax = axes[1]
    ax.plot(diameter_diffs, 'g-o', linewidth=2, markersize=4)
    ax.axhline(np.mean(diameter_diffs), color='red', linestyle='--', label=f'Átlag: {np.mean(diameter_diffs):.1f}px')
    ax.axhline(5, color='orange', linestyle=':', label='5px küszöb')
    ax.axhline(10, color='red', linestyle=':', label='10px küszöb')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő változás (pixel)')
    ax.set_title('Frame-to-Frame Átmérő Változás')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Pozíció trajektória
    ax = axes[2]
    scatter = ax.scatter(centers_x, centers_y, c=frames, cmap='viridis', s=50)
    ax.plot(centers_x, centers_y, 'k-', alpha=0.3, linewidth=1)
    # Első és utolsó frame kiemelése
    ax.scatter(centers_x[0], centers_y[0], c='green', s=200, marker='*', label='Start', edgecolor='black', linewidth=2)
    ax.scatter(centers_x[-1], centers_y[-1], c='red', s=200, marker='X', label='End', edgecolor='black', linewidth=2)
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.set_title('Pupilla Centrum Trajektória')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.colorbar(scatter, ax=ax, label='Frame')
    
    plt.tight_layout()
    plt.savefig('output/mediapipe_fluctuation_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ Grafikon mentve: output/mediapipe_fluctuation_analysis.png")
    plt.show()
    
    # Következtetés
    print("\n" + "=" * 70)
    print("🎯 KÖVETKEZTETÉS:")
    print("=" * 70)
    
    if np.std(diameters) < 5:
        print("✅ KIVÁLÓ stabilitás (σ < 5px)")
    elif np.std(diameters) < 10:
        print("✅ JÓ stabilitás (σ < 10px)")
    else:
        print("⚠️  KÖZEPES stabilitás (σ > 10px)")
    
    if np.mean(diameter_diffs) < 3:
        print("✅ ALACSONY frame-to-frame fluktuáció (< 3px)")
    elif np.mean(diameter_diffs) < 5:
        print("✅ KÖZEPES frame-to-frame fluktuáció (< 5px)")
    else:
        print("⚠️  MAGAS frame-to-frame fluktuáció (> 5px)")
    
    large_jumps = np.sum(diameter_diffs > 10)
    if large_jumps > 0:
        print(f"⚠️  {large_jumps} nagy ugrás (>10px) - lehet temporal smoothing szükséges")
    
    print("\n💡 Javaslatok:")
    if np.mean(diameter_diffs) > 3:
        print("  - Temporal smoothing (Kalman filter vagy Savitzky-Golay)")
        print("  - Outlier detection és kiszűrés")
    if np.std(diameters) > 10:
        print("  - Videó minőség ellenőrzése")
        print("  - Esetleg más AI modell (DeepVOG, RITnet)")
    
    print("=" * 70)


if __name__ == "__main__":
    analyze_temporal_fluctuation()
