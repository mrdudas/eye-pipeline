"""
Eye Tracking Pipeline - Main Module
Pupilla felismerés és követés videóból
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import json


@dataclass
class PupilData:
    """Pupilla adatok tárolására"""
    frame_number: int
    center_x: float
    center_y: float
    major_axis: float
    minor_axis: float
    angle: float
    confidence: float
    
    def to_dict(self):
        return {
            'frame': self.frame_number,
            'center': [self.center_x, self.center_y],
            'axes': [self.major_axis, self.minor_axis],
            'angle': self.angle,
            'confidence': self.confidence
        }


class EyeTrackingPipeline:
    """Fő pipeline osztály a pupilla detektáláshoz"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Pipeline inicializálás
        
        Args:
            config_path: Konfiguráció fájl elérési útja
        """
        self.config = self._load_config(config_path)
        self.video_path = self.config['video']['input_path']
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Videó megnyitása
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Nem sikerült megnyitni a videót: {self.video_path}")
        
        # Videó tulajdonságok
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Videó betöltve: {self.video_path}")
        print(f"  - Felbontás: {self.width}x{self.height}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Képkockák: {self.frame_count}")
        
        # Kamera kalibráció betöltése ha elérhető
        self.camera_matrix = None
        self.dist_coeffs = None
        if self.config['calibration']['enabled']:
            self._load_calibration()
        
        # CLAHE inicializálás
        if self.config['preprocessing']['clahe']['enabled']:
            clip = self.config['preprocessing']['clahe']['clip_limit']
            grid = tuple(self.config['preprocessing']['clahe']['tile_grid_size'])
            self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        
        # Blob detector inicializálás
        self._init_blob_detector()
        
        # Eredmények tárolása
        self.results = []
    
    def _load_config(self, config_path: str) -> dict:
        """Konfiguráció betöltése"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_calibration(self):
        """Kamera kalibráció betöltése"""
        cal = self.config['calibration']
        if all([cal['fx'], cal['fy'], cal['cx'], cal['cy']]):
            self.camera_matrix = np.array([
                [cal['fx'], 0, cal['cx']],
                [0, cal['fy'], cal['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.array(cal['distortion_coeffs'], dtype=np.float32)
            print("Kamera kalibráció betöltve")
    
    def _init_blob_detector(self):
        """Blob detector inicializálás a pupilla kezdeti detektálásához"""
        params = cv2.SimpleBlobDetector_Params()
        blob_cfg = self.config['pupil_detection']['blob']
        
        params.minThreshold = blob_cfg['min_threshold']
        params.maxThreshold = blob_cfg['max_threshold']
        params.filterByArea = True
        params.minArea = blob_cfg['min_area']
        params.maxArea = blob_cfg['max_area']
        params.filterByCircularity = True
        params.minCircularity = blob_cfg['circularity']
        params.filterByConvexity = False
        params.filterByInertia = False
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Képkocka előfeldolgozása
        
        Args:
            frame: Bemeneti képkocka
            
        Returns:
            Előfeldolgozott képkocka
        """
        # Szürkeárnyalatos konverzió
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Kamera torzítás korrekció
        if self.camera_matrix is not None:
            gray = cv2.undistort(gray, self.camera_matrix, self.dist_coeffs)
        
        # Glint eltávolítás
        if self.config['preprocessing']['glint_removal']['enabled']:
            gray = self._remove_glints(gray)
        
        # CLAHE kontrasztjavítás
        if self.config['preprocessing']['clahe']['enabled']:
            gray = self.clahe.apply(gray)
        
        return gray
    
    def _remove_glints(self, image: np.ndarray) -> np.ndarray:
        """
        Glint (fényes reflektálódások) eltávolítása - OPTIMALIZÁLT
        
        Args:
            image: Bemeneti szürkeárnyalatos kép
            
        Returns:
            Glint-mentes kép
        """
        glint_cfg = self.config['preprocessing']['glint_removal']
        threshold = glint_cfg['threshold']
        
        # Nagyon fényes pixelek detektálása
        _, glint_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Blob szűrés - csak valódi glint-ek maradnak
        glint_mask = self._filter_glint_blobs(glint_mask)
        
        # Ha inpainting engedélyezett
        if glint_cfg['inpainting']:
            radius = glint_cfg['inpainting_radius']
            
            # Maszk finomítás morfológiai műveletekkel
            kernel_size = glint_cfg.get('morph_kernel_size', 3)
            morph_iter = glint_cfg.get('morph_iterations', 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (kernel_size, kernel_size))
            glint_mask = cv2.dilate(glint_mask, kernel, iterations=morph_iter)
            
            # Inpainting módszer választása
            method = glint_cfg.get('inpainting_method', 'telea')
            if method == 'ns':
                result = cv2.inpaint(image, glint_mask, radius, cv2.INPAINT_NS)
            else:  # telea (default)
                result = cv2.inpaint(image, glint_mask, radius, cv2.INPAINT_TELEA)
            return result
        else:
            # Egyszerű maszkolás
            result = image.copy()
            result[glint_mask > 0] = np.median(image)
            return result
    
    def _filter_glint_blobs(self, mask: np.ndarray) -> np.ndarray:
        """
        Glint blob szűrés terület és forma alapján
        
        Args:
            mask: Bináris maszk a threshold után
            
        Returns:
            Szűrt maszk csak valódi glint-ekkel
        """
        glint_cfg = self.config['preprocessing']['glint_removal']
        min_area = glint_cfg.get('min_area', 5)
        max_area = glint_cfg.get('max_area', 200)
        min_circularity = glint_cfg.get('min_circularity', 0.3)
        
        # Connected components analízis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        
        # Új szűrt maszk
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 0 = háttér
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Területszűrés
            if min_area <= area <= max_area:
                # Circularity számítás
                blob_mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = contours[0]
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Circularity szűrés
                        if circularity >= min_circularity:
                            filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def detect_pupil(self, frame: np.ndarray, preprocessed: np.ndarray, 
                     frame_num: int) -> Optional[PupilData]:
        """
        Pupilla detektálás egy képkockában
        
        Args:
            frame: Eredeti képkocka (vizualizációhoz)
            preprocessed: Előfeldolgozott képkocka
            frame_num: Képkocka száma
            
        Returns:
            PupilData vagy None ha nem sikerült a detektálás
        """
        # Edge detection
        canny_cfg = self.config['pupil_detection']['canny']
        edges = cv2.Canny(preprocessed, 
                         canny_cfg['low_threshold'], 
                         canny_cfg['high_threshold'])
        
        # Kontúrok keresése
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Legnagyobb kontúr kiválasztása (feltételezve hogy ez a pupilla)
        # Szűrés terület alapján
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        if not valid_contours:
            return None
        
        # Kontúr kiválasztása terület alapján
        best_contour = max(valid_contours, key=cv2.contourArea)
        
        # Ellipszis illesztés
        if len(best_contour) >= 5:  # Minimum 5 pont kell az ellipszishez
            ellipse = cv2.fitEllipse(best_contour)
            
            # Ellipszis paraméterek
            (cx, cy), (ma, mi), angle = ellipse
            
            # Konfidencia számítás (egyszerű verzió)
            # Jobb algoritmus: RANSAC alapú illesztés
            area_contour = cv2.contourArea(best_contour)
            area_ellipse = np.pi * (ma/2) * (mi/2)
            confidence = min(area_contour / area_ellipse, 1.0) if area_ellipse > 0 else 0
            
            return PupilData(
                frame_number=frame_num,
                center_x=cx,
                center_y=cy,
                major_axis=ma,
                minor_axis=mi,
                angle=angle,
                confidence=confidence
            )
        
        return None
    
    def annotate_frame(self, frame: np.ndarray, pupil_data: Optional[PupilData]) -> np.ndarray:
        """
        Képkocka annotálása a detektált pupilla adatokkal
        
        Args:
            frame: Eredeti képkocka
            pupil_data: Detektált pupilla adatok
            
        Returns:
            Annotált képkocka
        """
        annotated = frame.copy()
        
        if pupil_data is not None:
            # Ellipszis rajzolása
            center = (int(pupil_data.center_x), int(pupil_data.center_y))
            axes = (int(pupil_data.major_axis/2), int(pupil_data.minor_axis/2))
            angle = pupil_data.angle
            
            cv2.ellipse(annotated, center, axes, angle, 0, 360, (0, 255, 0), 2)
            
            # Központ rajzolása
            cv2.circle(annotated, center, 3, (0, 0, 255), -1)
            
            # Konfidencia kiírása
            text = f"Conf: {pupil_data.confidence:.2f}"
            cv2.putText(annotated, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Átmérő kiírása (pixelben)
            diameter = pupil_data.minor_axis
            text_diameter = f"D: {diameter:.1f}px"
            cv2.putText(annotated, text_diameter, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "No pupil detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def process_video(self):
        """Teljes videó feldolgozása"""
        from tqdm import tqdm
        
        start_frame = self.config['video']['start_frame']
        end_frame = self.config['video']['end_frame']
        if end_frame is None:
            end_frame = self.frame_count
        
        # Videó pozíció beállítása
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Kimeneti videó létrehozása ha szükséges
        if self.config['output']['save_annotated']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = self.output_dir / 'annotated_output.mp4'
            self.video_writer = cv2.VideoWriter(
                str(out_path), fourcc, self.fps, (self.width, self.height)
            )
        
        print(f"\nVideofeldolgozás: {start_frame} -> {end_frame}")
        
        for frame_num in tqdm(range(start_frame, end_frame), desc="Feldolgozás"):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Előfeldolgozás
            preprocessed = self.preprocess_frame(frame)
            
            # Pupilla detektálás
            pupil_data = self.detect_pupil(frame, preprocessed, frame_num)
            
            if pupil_data is not None:
                self.results.append(pupil_data)
            
            # Annotált kép mentése/megjelenítése
            if self.config['output']['save_annotated'] or self.config['output']['show_preview']:
                annotated = self.annotate_frame(frame, pupil_data)
                
                if self.config['output']['save_annotated']:
                    self.video_writer.write(annotated)
                
                if self.config['output']['show_preview']:
                    cv2.imshow('Pupil Detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        print(f"\nFeldolgozás befejezve. {len(self.results)}/{end_frame-start_frame} képkockában találtunk pupillát.")
    
    def save_results(self):
        """Eredmények mentése JSON fájlba"""
        if self.config['output']['save_data']:
            output_file = self.output_dir / 'pupil_data.json'
            
            data = {
                'video_info': {
                    'path': self.video_path,
                    'fps': self.fps,
                    'resolution': [self.width, self.height],
                    'total_frames': self.frame_count
                },
                'detections': [p.to_dict() for p in self.results]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Eredmények mentve: {output_file}")
    
    def cleanup(self):
        """Erőforrások felszabadítása"""
        self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Pipeline futtatása"""
        try:
            self.process_video()
            self.save_results()
        finally:
            self.cleanup()


def main():
    """Fő futtatási függvény"""
    pipeline = EyeTrackingPipeline("config.yaml")
    pipeline.run()


if __name__ == "__main__":
    main()
