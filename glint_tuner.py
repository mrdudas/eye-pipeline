"""
Interaktív Glint Parameter Tuner
Valós időben állítható paraméterekkel tesztelhető a glint detektálás
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from pathlib import Path


class GlintTuner:
    def __init__(self, video_path='eye1.mp4', frame_number=0):
        self.video_path = video_path
        self.frame_number = frame_number
        
        # Videó betöltése
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Nem sikerült betölteni a képkockát!")
        
        self.original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Kezdeti paraméterek
        self.threshold = 240
        self.morph_size = 3
        self.morph_iterations = 1
        self.inpaint_radius = 3
        self.method = 'telea'  # 'telea', 'ns', 'median', 'blur'
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI létrehozása"""
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Képek
        self.ax_original = self.fig.add_subplot(gs[0, 0])
        self.ax_mask = self.fig.add_subplot(gs[0, 1])
        self.ax_result = self.fig.add_subplot(gs[0, 2])
        
        self.ax_diff = self.fig.add_subplot(gs[1, 0])
        self.ax_histogram = self.fig.add_subplot(gs[1, 1])
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        
        # Kontrolok
        ax_threshold = self.fig.add_subplot(gs[2, 0])
        ax_morph_size = self.fig.add_subplot(gs[2, 1])
        ax_morph_iter = self.fig.add_subplot(gs[2, 2])
        
        # Sliders
        self.slider_threshold = Slider(ax_threshold, 'Threshold', 180, 255, 
                                       valinit=self.threshold, valstep=1)
        self.slider_morph_size = Slider(ax_morph_size, 'Morph Size', 1, 9, 
                                        valinit=self.morph_size, valstep=2)
        self.slider_morph_iter = Slider(ax_morph_iter, 'Morph Iter', 0, 5, 
                                        valinit=self.morph_iterations, valstep=1)
        
        # Callback-ek
        self.slider_threshold.on_changed(self.update)
        self.slider_morph_size.on_changed(self.update)
        self.slider_morph_iter.on_changed(self.update)
        
        self.update(None)
        
    def process(self):
        """Glint feldolgozás aktuális paraméterekkel"""
        # Threshold
        _, mask = cv2.threshold(self.original, int(self.threshold), 255, cv2.THRESH_BINARY)
        
        # Morfológiai műveletek
        if self.morph_size > 0 and self.morph_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (int(self.morph_size), int(self.morph_size)))
            mask = cv2.dilate(mask, kernel, iterations=int(self.morph_iterations))
        
        # Inpainting
        if self.method == 'telea':
            result = cv2.inpaint(self.original, mask, int(self.inpaint_radius), 
                                cv2.INPAINT_TELEA)
        elif self.method == 'ns':
            result = cv2.inpaint(self.original, mask, int(self.inpaint_radius), 
                                cv2.INPAINT_NS)
        elif self.method == 'median':
            result = self.original.copy()
            median_val = np.median(self.original)
            result[mask > 0] = median_val
        elif self.method == 'blur':
            result = self.original.copy()
            blurred = cv2.GaussianBlur(self.original, (15, 15), 0)
            result[mask > 0] = blurred[mask > 0]
        
        return mask, result
        
    def update(self, val):
        """Vizualizáció frissítése"""
        self.threshold = self.slider_threshold.val
        self.morph_size = self.slider_morph_size.val
        self.morph_iterations = self.slider_morph_iter.val
        
        # Feldolgozás
        mask, result = self.process()
        
        # Különbség
        diff = cv2.absdiff(self.original, result)
        
        # Statisztikák
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        num_blobs = num_labels - 1
        glint_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        glint_percent = (glint_pixels / total_pixels) * 100
        
        # Képek frissítése
        self.ax_original.clear()
        self.ax_original.imshow(self.original, cmap='gray')
        self.ax_original.set_title('Eredeti kép', fontweight='bold')
        self.ax_original.axis('off')
        
        self.ax_mask.clear()
        self.ax_mask.imshow(mask, cmap='gray')
        self.ax_mask.set_title(f'Glint Maszk\nBlobs: {num_blobs}', fontweight='bold')
        self.ax_mask.axis('off')
        
        self.ax_result.clear()
        self.ax_result.imshow(result, cmap='gray')
        self.ax_result.set_title('Eredmény (Glint eltávolítva)', fontweight='bold')
        self.ax_result.axis('off')
        
        self.ax_diff.clear()
        self.ax_diff.imshow(diff, cmap='hot')
        self.ax_diff.set_title('Különbség (Eredeti - Eredmény)', fontweight='bold')
        self.ax_diff.axis('off')
        
        # Hisztogram
        self.ax_histogram.clear()
        self.ax_histogram.hist(self.original.ravel(), bins=256, range=(0, 256), 
                              color='blue', alpha=0.5, label='Eredeti')
        self.ax_histogram.hist(result.ravel(), bins=256, range=(0, 256), 
                              color='green', alpha=0.5, label='Eredmény')
        self.ax_histogram.axvline(x=self.threshold, color='r', linestyle='--', 
                                 linewidth=2, label=f'Threshold: {int(self.threshold)}')
        self.ax_histogram.set_title('Intenzitás Eloszlás', fontweight='bold')
        self.ax_histogram.set_xlabel('Pixel érték')
        self.ax_histogram.set_ylabel('Gyakoriság')
        self.ax_histogram.legend()
        self.ax_histogram.grid(True, alpha=0.3)
        
        # Statisztikák szöveg
        self.ax_stats.clear()
        stats_text = f"GLINT STATISZTIKÁK\n\n"
        stats_text += f"Threshold: {int(self.threshold)}\n"
        stats_text += f"Morph kernel: {int(self.morph_size)}x{int(self.morph_size)}\n"
        stats_text += f"Morph iter: {int(self.morph_iterations)}\n\n"
        stats_text += f"Detektált blobs: {num_blobs}\n"
        stats_text += f"Glint pixelek: {glint_pixels}\n"
        stats_text += f"Glint arány: {glint_percent:.2f}%\n\n"
        stats_text += f"Eredeti:\n"
        stats_text += f"  Min: {self.original.min()}\n"
        stats_text += f"  Max: {self.original.max()}\n"
        stats_text += f"  Átlag: {self.original.mean():.1f}\n\n"
        stats_text += f"Eredmény:\n"
        stats_text += f"  Min: {result.min()}\n"
        stats_text += f"  Max: {result.max()}\n"
        stats_text += f"  Átlag: {result.mean():.1f}\n"
        
        self.ax_stats.text(0.1, 0.5, stats_text, fontsize=10, 
                          family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                          verticalalignment='center')
        self.ax_stats.set_title('Statisztikák', fontweight='bold')
        self.ax_stats.axis('off')
        
        self.fig.canvas.draw_idle()
        
    def show(self):
        """UI megjelenítése"""
        plt.show()


def compare_frames_side_by_side():
    """
    Több képkocka összehasonlítása a jelenlegi és optimalizált paraméterekkel
    """
    cap = cv2.VideoCapture('eye1.mp4')
    frame_numbers = [0, 5, 10, 15, 20, 25]
    
    fig, axes = plt.subplots(len(frame_numbers), 5, figsize=(18, 3*len(frame_numbers)))
    fig.suptitle('Glint Eltávolítás - Optimalizált Paraméterek', 
                 fontsize=14, fontweight='bold')
    
    # Optimalizált paraméterek
    threshold = 240
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    iterations = 1
    inpaint_radius = 3
    
    for idx, frame_num in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Feldolgozás
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask_dilated = cv2.dilate(mask, kernel, iterations=iterations)
        inpainted = cv2.inpaint(gray, mask_dilated, inpaint_radius, cv2.INPAINT_TELEA)
        diff = cv2.absdiff(gray, inpainted)
        
        # CLAHE az eredményen
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inpainted)
        
        # Vizualizáció
        axes[idx, 0].imshow(gray, cmap='gray')
        axes[idx, 0].set_title(f'Frame {frame_num}' if idx == 0 else '', fontweight='bold')
        axes[idx, 0].set_ylabel(f'#{frame_num}', fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask_dilated, cmap='gray')
        axes[idx, 1].set_title('Maszk' if idx == 0 else '', fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(inpainted, cmap='gray')
        axes[idx, 2].set_title('Inpainted' if idx == 0 else '', fontweight='bold')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(enhanced, cmap='gray')
        axes[idx, 3].set_title('+ CLAHE' if idx == 0 else '', fontweight='bold')
        axes[idx, 3].axis('off')
        
        axes[idx, 4].imshow(diff, cmap='hot')
        axes[idx, 4].set_title('Különbség' if idx == 0 else '', fontweight='bold')
        axes[idx, 4].axis('off')
    
    cap.release()
    
    plt.tight_layout()
    output_path = Path('output/glint_optimized_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Optimalizált összehasonlítás mentve: {output_path}")
    plt.show()


def main():
    """Fő függvény"""
    print("="*60)
    print("GLINT PARAMETER TUNER")
    print("="*60)
    print("\nHasználat:")
    print("  - Állítsd a slider-eket az optimális paraméterek megtalálásához")
    print("  - Figyeld a detektált blobs számát és a glint arányt")
    print("  - Nézd meg a különbség képet (piros = változás)")
    print("\n" + "="*60)
    
    # Interaktív tuner
    try:
        tuner = GlintTuner(frame_number=10)
        tuner.show()
    except Exception as e:
        print(f"Interaktív tuner hiba: {e}")
    
    # Összehasonlító vizualizáció
    print("\nÖsszehasonlító vizualizáció készítése...")
    compare_frames_side_by_side()


if __name__ == "__main__":
    main()
