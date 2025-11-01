"""
GLINT Removal Teljes Videó Feldolgozás
Csak a glint eltávolítást mutatja vizuálisan
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm


class GlintOnlyProcessor:
    """Csak glint removal vizualizáció"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.video_path = self.config['video']['input_path']
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Videó megnyitása
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Nem sikerült megnyitni: {self.video_path}")
        
        # Videó info
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Videó: {self.video_path}")
        print(f"  Felbontás: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Képkockák: {self.frame_count}")
        print(f"  Időtartam: {self.frame_count/self.fps:.1f} másodperc")
    
    def remove_glints(self, image):
        """Glint eltávolítás az aktuális konfigurációval"""
        glint_cfg = self.config['preprocessing']['glint_removal']
        threshold = glint_cfg['threshold']
        
        # Threshold
        _, glint_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Blob szűrés
        glint_mask = self._filter_glint_blobs(glint_mask, glint_cfg)
        
        # Morfológiai műveletek
        kernel_size = glint_cfg.get('morph_kernel_size', 3)
        morph_iter = glint_cfg.get('morph_iterations', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        glint_mask_dilated = cv2.dilate(glint_mask, kernel, iterations=morph_iter)
        
        # Inpainting
        method = glint_cfg.get('inpainting_method', 'telea')
        radius = glint_cfg.get('inpainting_radius', 3)
        
        if method == 'ns':
            result = cv2.inpaint(image, glint_mask_dilated, radius, cv2.INPAINT_NS)
        else:
            result = cv2.inpaint(image, glint_mask_dilated, radius, cv2.INPAINT_TELEA)
        
        return result, glint_mask, glint_mask_dilated
    
    def _filter_glint_blobs(self, mask, glint_cfg):
        """Blob szűrés"""
        min_area = glint_cfg.get('min_area', 5)
        max_area = glint_cfg.get('max_area', 200)
        min_circularity = glint_cfg.get('min_circularity', 0.3)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if min_area <= area <= max_area:
                blob_mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = contours[0]
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity >= min_circularity:
                            filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def create_visualization_frame(self, original, result, mask, mask_dilated, frame_num):
        """4-panel vizualizáció létrehozása"""
        # Colormap a különbséghez
        diff = cv2.absdiff(original, result)
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
        
        # Maszk overlay az eredetire
        mask_overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        mask_overlay[mask_dilated > 0] = [0, 255, 255]  # Cyan a glint területek
        
        # Címkék hozzáadása
        def add_label(img, text, position=(10, 30)):
            labeled = img.copy()
            cv2.putText(labeled, text, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(labeled, text, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            return labeled
        
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        top_left = add_label(original_bgr, "1. Eredeti")
        top_right = add_label(mask_overlay, "2. Detektalt Glint (cyan)")
        bottom_left = add_label(result_bgr, "3. Glint Eltavolitva")
        bottom_right = add_label(diff_colored, "4. Kulonbseg (hot)")
        
        # Frame szám az összes panelre
        frame_text = f"Frame: {frame_num}/{self.frame_count}"
        for img in [top_left, top_right, bottom_left, bottom_right]:
            cv2.putText(img, frame_text, (10, self.height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, frame_text, (10, self.height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Statisztikák
        glint_pixels = np.sum(mask_dilated > 0)
        glint_percent = (glint_pixels / (self.width * self.height)) * 100
        stats_text = f"Glint: {glint_pixels}px ({glint_percent:.2f}%)"
        
        cv2.putText(bottom_right, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bottom_right, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 2x2 grid összeállítása
        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def process_video(self):
        """Teljes videó feldolgozása"""
        # Kimeneti videó beállítása
        output_width = self.width * 2
        output_height = self.height * 2
        
        output_path = self.output_dir / 'glint_removal_full_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, 
                                (output_width, output_height))
        
        print(f"\nFeldolgozás indul...")
        print(f"Kimenet: {output_path}")
        print(f"Kimeneti felbontás: {output_width}x{output_height}")
        print(f"Morfológiai iterációk: {self.config['preprocessing']['glint_removal']['morph_iterations']}")
        
        # Statisztikák gyűjtése
        total_glint_pixels = 0
        max_glint_pixels = 0
        max_glint_frame = 0
        
        try:
            for frame_num in tqdm(range(self.frame_count), desc="Feldolgozás"):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Szürkeárnyalatos
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Glint removal
                result, mask, mask_dilated = self.remove_glints(gray)
                
                # Statisztikák
                glint_pixels = np.sum(mask_dilated > 0)
                total_glint_pixels += glint_pixels
                
                if glint_pixels > max_glint_pixels:
                    max_glint_pixels = glint_pixels
                    max_glint_frame = frame_num
                
                # Vizualizáció
                viz_frame = self.create_visualization_frame(
                    gray, result, mask, mask_dilated, frame_num)
                
                # Írás
                writer.write(viz_frame)
        
        finally:
            self.cap.release()
            writer.release()
        
        # Statisztikák kiírása
        avg_glint_pixels = total_glint_pixels / self.frame_count
        avg_glint_percent = (avg_glint_pixels / (self.width * self.height)) * 100
        max_glint_percent = (max_glint_pixels / (self.width * self.height)) * 100
        
        print("\n" + "="*60)
        print("FELDOLGOZÁS BEFEJEZVE!")
        print("="*60)
        print(f"\nKimenet: {output_path}")
        print(f"Felbontás: {output_width}x{output_height}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Feldolgozott képkockák: {self.frame_count}")
        
        print(f"\nGLINT Statisztikák:")
        print(f"  Átlagos glint terület: {avg_glint_pixels:.0f} pixel ({avg_glint_percent:.2f}%)")
        print(f"  Maximum glint terület: {max_glint_pixels} pixel ({max_glint_percent:.2f}%)")
        print(f"  Maximum frame: {max_glint_frame}")
        
        print(f"\nKonfiguráció:")
        glint_cfg = self.config['preprocessing']['glint_removal']
        print(f"  Threshold: {glint_cfg['threshold']}")
        print(f"  Morph kernel: {glint_cfg['morph_kernel_size']}x{glint_cfg['morph_kernel_size']}")
        print(f"  Morph iterációk: {glint_cfg['morph_iterations']}")
        print(f"  Min area: {glint_cfg['min_area']} pixel")
        print(f"  Max area: {glint_cfg['max_area']} pixel")
        print(f"  Min circularity: {glint_cfg['min_circularity']}")
        print(f"  Inpainting: {glint_cfg['inpainting_method']}")
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nFájlméret: {file_size_mb:.1f} MB")
        print("="*60)


def main():
    print("="*60)
    print("GLINT REMOVAL - TELJES VIDEÓ FELDOLGOZÁS")
    print("="*60)
    
    processor = GlintOnlyProcessor()
    processor.process_video()
    
    print("\n✅ Videó elkészült!")
    print("   Nézd meg: output/glint_removal_full_video.mp4")


if __name__ == "__main__":
    main()
