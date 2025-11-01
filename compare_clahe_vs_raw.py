"""
Összehasonlítás: AI CLAHE preprocessing vs AI RAW input
Megmutatja a histogram equalization hatását a MediaPipe stabilitására
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(file_path):
    """JSON eredmények betöltése"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['detections']


def analyze_stability(detections, label):
    """Stabilitás elemzés"""
    if not detections:
        print(f"{label}: Nincs adat")
        return None
    
    frames = [d['frame'] for d in detections]
    diameters = [d['diameter'] for d in detections]
    centers_x = [d['center'][0] for d in detections]
    centers_y = [d['center'][1] for d in detections]
    confidences = [d['confidence'] for d in detections]
    
    print(f"\n{label}:")
    print(f"  Detektálva: {len(detections)} frame")
    print(f"  Átmérő: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} px")
    print(f"  Pozíció X std: {np.std(centers_x):.2f} px")
    print(f"  Pozíció Y std: {np.std(centers_y):.2f} px")
    print(f"  Konfidencia: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    
    # Frame-to-frame változás (fluktuáció)
    diameter_diffs = np.abs(np.diff(diameters))
    position_diffs = np.sqrt(np.diff(centers_x)**2 + np.diff(centers_y)**2)
    
    print(f"  Frame-to-frame átmérő változás: {np.mean(diameter_diffs):.2f} ± {np.std(diameter_diffs):.2f} px")
    print(f"  Frame-to-frame pozíció változás: {np.mean(position_diffs):.2f} ± {np.std(position_diffs):.2f} px")
    
    return {
        'frames': frames,
        'diameters': diameters,
        'centers_x': centers_x,
        'centers_y': centers_y,
        'confidences': confidences,
        'diameter_diffs': diameter_diffs,
        'position_diffs': position_diffs
    }


def plot_comparison(data_clahe, data_raw):
    """Összehasonlító grafikonok"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MediaPipe: CLAHE Preprocessing vs RAW Input', fontsize=16, fontweight='bold')
    
    # 1. Átmérő időben
    ax = axes[0, 0]
    if data_clahe:
        ax.plot(data_clahe['frames'], data_clahe['diameters'], 
                'b-', alpha=0.7, label='CLAHE', linewidth=2)
    if data_raw:
        ax.plot(data_raw['frames'], data_raw['diameters'], 
                'r-', alpha=0.7, label='RAW', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő (pixel)')
    ax.set_title('Pupilla Átmérő')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Konfidencia időben
    ax = axes[0, 1]
    if data_clahe:
        ax.plot(data_clahe['frames'], data_clahe['confidences'], 
                'b-', alpha=0.7, label='CLAHE', linewidth=2)
    if data_raw:
        ax.plot(data_raw['frames'], data_raw['confidences'], 
                'r-', alpha=0.7, label='RAW', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Konfidencia')
    ax.set_title('Detektálási Konfidencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Átmérő eloszlás
    ax = axes[0, 2]
    if data_clahe:
        ax.hist(data_clahe['diameters'], bins=30, alpha=0.5, 
                label=f"CLAHE (σ={np.std(data_clahe['diameters']):.1f})", color='blue')
    if data_raw:
        ax.hist(data_raw['diameters'], bins=30, alpha=0.5, 
                label=f"RAW (σ={np.std(data_raw['diameters']):.1f})", color='red')
    ax.set_xlabel('Átmérő (pixel)')
    ax.set_ylabel('Gyakoriság')
    ax.set_title('Átmérő Eloszlás')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Frame-to-frame átmérő változás
    ax = axes[1, 0]
    if data_clahe:
        ax.plot(data_clahe['diameter_diffs'], 'b-', alpha=0.7, 
                label=f"CLAHE (μ={np.mean(data_clahe['diameter_diffs']):.1f})", linewidth=2)
    if data_raw:
        ax.plot(data_raw['diameter_diffs'], 'r-', alpha=0.7, 
                label=f"RAW (μ={np.mean(data_raw['diameter_diffs']):.1f})", linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő változás (pixel)')
    ax.set_title('Frame-to-Frame Átmérő Fluktuáció')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Frame-to-frame pozíció változás
    ax = axes[1, 1]
    if data_clahe:
        ax.plot(data_clahe['position_diffs'], 'b-', alpha=0.7, 
                label=f"CLAHE (μ={np.mean(data_clahe['position_diffs']):.1f})", linewidth=2)
    if data_raw:
        ax.plot(data_raw['position_diffs'], 'r-', alpha=0.7, 
                label=f"RAW (μ={np.mean(data_raw['position_diffs']):.1f})", linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Pozíció változás (pixel)')
    ax.set_title('Frame-to-Frame Pozíció Fluktuáció')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Pozíció scatter
    ax = axes[1, 2]
    if data_clahe:
        ax.scatter(data_clahe['centers_x'], data_clahe['centers_y'], 
                  c='blue', alpha=0.5, s=20, label='CLAHE')
    if data_raw:
        ax.scatter(data_raw['centers_x'], data_raw['centers_y'], 
                  c='red', alpha=0.5, s=20, label='RAW')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.set_title('Pupilla Centrum Pozíció')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('output/clahe_vs_raw_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Grafikon mentve: output/clahe_vs_raw_comparison.png")
    plt.show()


def main():
    """Fő összehasonlítás"""
    print("=" * 70)
    print("MediaPipe Stabilitás: CLAHE Preprocessing vs RAW Input")
    print("=" * 70)
    
    # Adatok betöltése
    data_clahe = None
    data_raw = None
    
    if Path('output/ai_pupil_data.json').exists():
        detections_clahe = load_results('output/ai_pupil_data.json')
        data_clahe = analyze_stability(detections_clahe, "🔵 AI + CLAHE")
    else:
        print("⚠️  Nincs ai_pupil_data.json - futtasd: python ai_pupil_pipeline.py")
    
    if Path('output/ai_raw_pupil_data.json').exists():
        detections_raw = load_results('output/ai_raw_pupil_data.json')
        data_raw = analyze_stability(detections_raw, "🔴 AI RAW")
    else:
        print("⚠️  Nincs ai_raw_pupil_data.json - futtasd: python ai_pupil_pipeline_raw.py")
    
    # Összehasonlítás
    if data_clahe and data_raw:
        print("\n" + "=" * 70)
        print("📊 ÖSSZEHASONLÍTÁS")
        print("=" * 70)
        
        clahe_std = np.std(data_clahe['diameters'])
        raw_std = np.std(data_raw['diameters'])
        
        clahe_ftf = np.mean(data_clahe['diameter_diffs'])
        raw_ftf = np.mean(data_raw['diameter_diffs'])
        
        print(f"\nÁtmérő stabilitás (std):")
        print(f"  CLAHE: {clahe_std:.2f} px")
        print(f"  RAW:   {raw_std:.2f} px")
        if clahe_std > raw_std:
            improvement = (clahe_std - raw_std) / clahe_std * 100
            print(f"  ✅ RAW {improvement:.1f}% STABILABB!")
        else:
            degradation = (raw_std - clahe_std) / raw_std * 100
            print(f"  ⚠️  CLAHE {degradation:.1f}% stabilabb")
        
        print(f"\nFrame-to-frame fluktuáció:")
        print(f"  CLAHE: {clahe_ftf:.2f} px")
        print(f"  RAW:   {raw_ftf:.2f} px")
        if clahe_ftf > raw_ftf:
            improvement = (clahe_ftf - raw_ftf) / clahe_ftf * 100
            print(f"  ✅ RAW {improvement:.1f}% KEVESEBB fluktuáció!")
        else:
            degradation = (raw_ftf - clahe_ftf) / raw_ftf * 100
            print(f"  ⚠️  CLAHE {degradation:.1f}% kevesebb fluktuáció")
        
        print("\n" + "=" * 70)
        print("🎯 KÖVETKEZTETÉS:")
        if raw_std < clahe_std and raw_ftf < clahe_ftf:
            print("✅ RAW INPUT AJÁNLOTT - MediaPipe nem igényel előfeldolgozást!")
            print("   A CLAHE histogram equalization RONTJA a stabilitást!")
        elif clahe_std < raw_std:
            print("⚠️  CLAHE jobb - de ellenőrizd a videó minőségét!")
        else:
            print("🤔 Vegyes eredmények - további tesztelés szükséges")
        print("=" * 70)
        
        # Grafikonok
        plot_comparison(data_clahe, data_raw)
    else:
        print("\n⚠️  Mindkét verzió eredménye szükséges az összehasonlításhoz!")
        print("\n1. Futtasd: python ai_pupil_pipeline.py")
        print("2. Futtasd: python ai_pupil_pipeline_raw.py")
        print("3. Futtasd újra ezt a scriptet")


if __name__ == "__main__":
    main()
