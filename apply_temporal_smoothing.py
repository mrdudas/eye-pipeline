"""
Temporal Smoothing: Kalman Filter és Savitzky-Golay
Csökkenti a frame-to-frame fluktuációt
"""

import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from typing import List, Tuple


class KalmanFilter1D:
    """Egyszerű 1D Kalman filter"""
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1, initial_value=0, initial_estimate_error=1):
        self.process_variance = process_variance  # Q
        self.measurement_variance = measurement_variance  # R
        self.estimate = initial_value
        self.estimate_error = initial_estimate_error
    
    def update(self, measurement):
        """Következő mérés feldolgozása"""
        # Prediction
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate


def apply_kalman_filter(values: List[float], process_var=1e-3, measurement_var=0.1) -> np.ndarray:
    """Kalman filter alkalmazása"""
    kf = KalmanFilter1D(
        process_variance=process_var,
        measurement_variance=measurement_var,
        initial_value=values[0],
        initial_estimate_error=1.0
    )
    
    filtered = []
    for v in values:
        filtered.append(kf.update(v))
    
    return np.array(filtered)


def apply_savgol_filter(values: List[float], window_length=7, polyorder=2) -> np.ndarray:
    """Savitzky-Golay filter"""
    if len(values) < window_length:
        window_length = len(values) if len(values) % 2 == 1 else len(values) - 1
    
    if window_length < polyorder + 2:
        return np.array(values)
    
    return savgol_filter(values, window_length, polyorder)


def apply_gaussian_filter(values: List[float], sigma=2.0) -> np.ndarray:
    """Gaussian smoothing"""
    return gaussian_filter1d(values, sigma=sigma)


def smooth_pupil_data(input_file: str, output_file: str, method='kalman', **kwargs):
    """
    Pupilla adatok simítása
    
    Args:
        input_file: Input JSON fájl
        output_file: Output JSON fájl
        method: 'kalman', 'savgol', vagy 'gaussian'
        **kwargs: Módszer-specifikus paraméterek
    """
    # Adatok betöltése
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detections = data['detections']
    
    # Adatok kinyerése
    frames = [d['frame'] for d in detections]
    diameters = [d['diameter'] for d in detections]
    centers_x = [d['center'][0] for d in detections]
    centers_y = [d['center'][1] for d in detections]
    radii = [d['radius'] for d in detections]
    confidences = [d['confidence'] for d in detections]
    
    # Simítás
    print(f"Simítási módszer: {method.upper()}")
    
    if method == 'kalman':
        process_var = kwargs.get('process_var', 1e-3)
        measurement_var = kwargs.get('measurement_var', 0.1)
        print(f"  Process variance: {process_var}")
        print(f"  Measurement variance: {measurement_var}")
        
        diameters_smooth = apply_kalman_filter(diameters, process_var, measurement_var)
        centers_x_smooth = apply_kalman_filter(centers_x, process_var, measurement_var)
        centers_y_smooth = apply_kalman_filter(centers_y, process_var, measurement_var)
    
    elif method == 'savgol':
        window = kwargs.get('window_length', 7)
        polyorder = kwargs.get('polyorder', 2)
        print(f"  Window length: {window}")
        print(f"  Polynomial order: {polyorder}")
        
        diameters_smooth = apply_savgol_filter(diameters, window, polyorder)
        centers_x_smooth = apply_savgol_filter(centers_x, window, polyorder)
        centers_y_smooth = apply_savgol_filter(centers_y, window, polyorder)
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 2.0)
        print(f"  Sigma: {sigma}")
        
        diameters_smooth = apply_gaussian_filter(diameters, sigma)
        centers_x_smooth = apply_gaussian_filter(centers_x, sigma)
        centers_y_smooth = apply_gaussian_filter(centers_y, sigma)
    
    else:
        raise ValueError(f"Ismeretlen módszer: {method}")
    
    # Statisztikák
    print(f"\n📊 Simítás előtt:")
    print(f"  Átmérő std: {np.std(diameters):.2f} px")
    print(f"  Frame-to-frame változás: {np.mean(np.abs(np.diff(diameters))):.2f} px")
    
    print(f"\n📊 Simítás után:")
    print(f"  Átmérő std: {np.std(diameters_smooth):.2f} px")
    print(f"  Frame-to-frame változás: {np.mean(np.abs(np.diff(diameters_smooth))):.2f} px")
    
    improvement_std = (1 - np.std(diameters_smooth) / np.std(diameters)) * 100
    improvement_ftf = (1 - np.mean(np.abs(np.diff(diameters_smooth))) / np.mean(np.abs(np.diff(diameters)))) * 100
    
    print(f"\n✨ Javulás:")
    print(f"  Std: {improvement_std:.1f}% csökkenés")
    print(f"  Frame-to-frame: {improvement_ftf:.1f}% csökkenés")
    
    # Új detekciós lista
    smoothed_detections = []
    for i, detection in enumerate(detections):
        new_detection = detection.copy()
        new_detection['diameter'] = float(diameters_smooth[i])
        new_detection['radius'] = float(diameters_smooth[i] / 2)
        new_detection['center'] = [float(centers_x_smooth[i]), float(centers_y_smooth[i])]
        new_detection['method'] = f"{detection['method']}_smoothed_{method}"
        new_detection['original_diameter'] = float(diameters[i])
        new_detection['original_center'] = [float(centers_x[i]), float(centers_y[i])]
        smoothed_detections.append(new_detection)
    
    # Mentés
    output_data = data.copy()
    output_data['method'] = f"{data['method']} + {method.upper()} smoothing"
    output_data['smoothing'] = {
        'method': method,
        'parameters': kwargs
    }
    output_data['detections'] = smoothed_detections
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Simított adatok mentve: {output_file}")
    
    return diameters, diameters_smooth, centers_x, centers_x_smooth, centers_y, centers_y_smooth


def compare_smoothing_methods(input_file: str):
    """Különböző simítási módszerek összehasonlítása"""
    
    # Adatok betöltése
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detections = data['detections']
    frames = [d['frame'] for d in detections]
    diameters = [d['diameter'] for d in detections]
    
    # Különböző simítások
    methods = {
        'Original': diameters,
        'Kalman (strict)': apply_kalman_filter(diameters, 1e-4, 0.1),
        'Kalman (balanced)': apply_kalman_filter(diameters, 1e-3, 0.1),
        'Kalman (loose)': apply_kalman_filter(diameters, 1e-2, 0.1),
        'Savitzky-Golay (w=5)': apply_savgol_filter(diameters, 5, 2),
        'Savitzky-Golay (w=7)': apply_savgol_filter(diameters, 7, 2),
        'Savitzky-Golay (w=11)': apply_savgol_filter(diameters, 11, 2),
        'Gaussian (σ=1)': apply_gaussian_filter(diameters, 1.0),
        'Gaussian (σ=2)': apply_gaussian_filter(diameters, 2.0),
        'Gaussian (σ=3)': apply_gaussian_filter(diameters, 3.0),
    }
    
    # Vizualizáció
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Temporal Smoothing Módszerek Összehasonlítása', fontsize=14, fontweight='bold')
    
    # 1. Minden módszer
    ax = axes[0, 0]
    for name, values in methods.items():
        if name == 'Original':
            ax.plot(frames, values, 'k-', alpha=0.3, linewidth=2, label=name)
        else:
            ax.plot(frames, values, alpha=0.7, linewidth=1.5, label=name)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő (pixel)')
    ax.set_title('Összes Simítási Módszer')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Csak Kalman
    ax = axes[0, 1]
    ax.plot(frames, diameters, 'k-', alpha=0.3, linewidth=2, label='Original')
    ax.plot(frames, methods['Kalman (strict)'], 'b-', linewidth=2, label='Kalman (strict)')
    ax.plot(frames, methods['Kalman (balanced)'], 'g-', linewidth=2, label='Kalman (balanced)')
    ax.plot(frames, methods['Kalman (loose)'], 'r-', linewidth=2, label='Kalman (loose)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő (pixel)')
    ax.set_title('Kalman Filter Paraméterek')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Csak Savitzky-Golay
    ax = axes[1, 0]
    ax.plot(frames, diameters, 'k-', alpha=0.3, linewidth=2, label='Original')
    ax.plot(frames, methods['Savitzky-Golay (w=5)'], 'b-', linewidth=2, label='w=5')
    ax.plot(frames, methods['Savitzky-Golay (w=7)'], 'g-', linewidth=2, label='w=7')
    ax.plot(frames, methods['Savitzky-Golay (w=11)'], 'r-', linewidth=2, label='w=11')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő (pixel)')
    ax.set_title('Savitzky-Golay Filter Window Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Csak Gaussian
    ax = axes[1, 1]
    ax.plot(frames, diameters, 'k-', alpha=0.3, linewidth=2, label='Original')
    ax.plot(frames, methods['Gaussian (σ=1)'], 'b-', linewidth=2, label='σ=1')
    ax.plot(frames, methods['Gaussian (σ=2)'], 'g-', linewidth=2, label='σ=2')
    ax.plot(frames, methods['Gaussian (σ=3)'], 'r-', linewidth=2, label='σ=3')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Átmérő (pixel)')
    ax.set_title('Gaussian Filter Sigma Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/smoothing_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Grafikon mentve: output/smoothing_comparison.png")
    plt.show()
    
    # Statisztikai összehasonlítás
    print("\n" + "=" * 80)
    print("📊 STATISZTIKAI ÖSSZEHASONLÍTÁS")
    print("=" * 80)
    print(f"{'Módszer':<25} {'Std':<10} {'FtF Átlag':<12} {'FtF Max':<10}")
    print("-" * 80)
    
    for name, values in methods.items():
        std = np.std(values)
        ftf_mean = np.mean(np.abs(np.diff(values)))
        ftf_max = np.max(np.abs(np.diff(values)))
        print(f"{name:<25} {std:<10.2f} {ftf_mean:<12.2f} {ftf_max:<10.2f}")
    
    print("=" * 80)


def main():
    """Fő függvény"""
    input_file = 'output/ai_pupil_data.json'
    
    print("=" * 70)
    print("🌊 TEMPORAL SMOOTHING - Fluktuáció Csökkentés")
    print("=" * 70)
    
    # 1. Összehasonlítás
    print("\n1️⃣  Módszerek összehasonlítása...\n")
    compare_smoothing_methods(input_file)
    
    # 2. Legjobb módszer alkalmazása
    print("\n2️⃣  Legjobb módszer alkalmazása...\n")
    
    # Kalman (balanced) - általában legjobb real-time tracking-hez
    smooth_pupil_data(
        input_file,
        'output/ai_pupil_data_smoothed_kalman.json',
        method='kalman',
        process_var=1e-3,
        measurement_var=0.1
    )
    
    # Savitzky-Golay - jó post-processing-hez
    smooth_pupil_data(
        input_file,
        'output/ai_pupil_data_smoothed_savgol.json',
        method='savgol',
        window_length=7,
        polyorder=2
    )
    
    print("\n" + "=" * 70)
    print("✅ Smoothing befejezve!")
    print("=" * 70)


if __name__ == "__main__":
    main()
