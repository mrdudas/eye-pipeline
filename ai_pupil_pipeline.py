"""
AI-alapú Pupilla Detektálás - MediaPipe Iris Integration
"""

import cv2
import numpy as np
import mediapipe as mp
import yaml
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class PupilDataAI:
    """AI-alapú pupilla adatok"""
    frame_number: int
    center_x: float
    center_y: float
    radius: float  # Becsült sugár az iris pontokból
    iris_points: list  # Összes iris landmark
    confidence: float
    method: str = "mediapipe"
    
    def to_dict(self):
        return {
            'frame': self.frame_number,
            'center': [self.center_x, self.center_y],
            'radius': self.radius,
            'diameter': self.radius * 2,
            'iris_points': self.iris_points,
            'confidence': self.confidence,
            'method': self.method
        }


class MediaPipePupilDetector:
    """MediaPipe alapú pupilla/iris detektálás"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicializálás"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("MediaPipe Iris detector inicializálva")
    
    def detect_pupil(self, frame: np.ndarray, frame_num: int) -> Optional[PupilDataAI]:
        """
        Pupilla/iris detektálás egy képkockában
        
        Args:
            frame: BGR képkocka
            frame_num: Képkocka száma
            
        Returns:
            PupilDataAI vagy None
        """
        # MediaPipe RGB-t igényel
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Feldolgozás
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Első arc landmarks
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Iris landmarks: 468-477 (10 pont)
        # 468-471: jobb szem iris
        # 473-476: bal szem iris
        # Mivel close-up, általában csak egy szem látszik
        
        iris_points = []
        for idx in range(468, 478):
            lm = landmarks.landmark[idx]
            x = lm.x * w
            y = lm.y * h
            iris_points.append([float(x), float(y)])
        
        if not iris_points:
            return None
        
        # Iris centrum (átlag)
        iris_points_np = np.array(iris_points)
        center_x = float(np.mean(iris_points_np[:, 0]))
        center_y = float(np.mean(iris_points_np[:, 1]))
        
        # Sugár becslése (std vagy max távolság a centrumtól)
        distances = np.sqrt(
            (iris_points_np[:, 0] - center_x)**2 + 
            (iris_points_np[:, 1] - center_y)**2
        )
        radius = float(np.mean(distances))
        
        # Konfidencia (MediaPipe nem ad direkt confidence-t, használjuk a pontok konzisztenciáját)
        # Ha a pontok közel vannak egymáshoz (kis std) → jó detektálás
        confidence = 1.0 / (1.0 + np.std(distances) / radius)  # 0-1 range
        confidence = float(min(max(confidence, 0.0), 1.0))
        
        return PupilDataAI(
            frame_number=frame_num,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            iris_points=iris_points,
            confidence=confidence,
            method="mediapipe"
        )
    
    def cleanup(self):
        """Erőforrások felszabadítása"""
        self.face_mesh.close()


class AIEyeTrackingPipeline:
    """AI-alapú Eye Tracking Pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Pipeline inicializálás"""
        self.config = self._load_config(config_path)
        self.video_path = self.config['video']['input_path']
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Videó
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Nem sikerült megnyitni: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Videó: {self.video_path}")
        print(f"  Felbontás: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Képkockák: {self.frame_count}")
        
        # AI detector
        self.detector = MediaPipePupilDetector(config_path)
        
        # Eredmények
        self.results = []
    
    def _load_config(self, config_path: str) -> dict:
        """Konfiguráció betöltése"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def annotate_frame(self, frame: np.ndarray, pupil_data: Optional[PupilDataAI]) -> np.ndarray:
        """Képkocka annotálása"""
        annotated = frame.copy()
        
        if pupil_data is not None:
            # Iris pontok
            for point in pupil_data.iris_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
            
            # Centrum
            center = (int(pupil_data.center_x), int(pupil_data.center_y))
            cv2.circle(annotated, center, 5, (0, 0, 255), -1)
            
            # Sugár kör
            radius = int(pupil_data.radius)
            cv2.circle(annotated, center, radius, (255, 0, 0), 2)
            
            # Szöveg
            cv2.putText(annotated, f"MediaPipe Iris", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Conf: {pupil_data.confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"D: {pupil_data.radius*2:.1f}px", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "No detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def process_video(self):
        """Teljes videó feldolgozása"""
        start_frame = self.config['video']['start_frame']
        end_frame = self.config['video']['end_frame']
        if end_frame is None:
            end_frame = self.frame_count
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Kimeneti videó
        if self.config['output']['save_annotated']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = self.output_dir / 'ai_annotated_output.mp4'
            self.video_writer = cv2.VideoWriter(
                str(out_path), fourcc, self.fps, (self.width, self.height)
            )
        
        print(f"\nAI feldolgozás: {start_frame} -> {end_frame}")
        
        for frame_num in tqdm(range(start_frame, end_frame), desc="AI Detection"):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # AI detektálás
            pupil_data = self.detector.detect_pupil(frame, frame_num)
            
            if pupil_data is not None:
                self.results.append(pupil_data)
            
            # Annotált kép
            if self.config['output']['save_annotated'] or self.config['output']['show_preview']:
                annotated = self.annotate_frame(frame, pupil_data)
                
                if self.config['output']['save_annotated']:
                    self.video_writer.write(annotated)
                
                if self.config['output']['show_preview']:
                    cv2.imshow('AI Pupil Detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        print(f"\nFeldolgozás befejezve: {len(self.results)}/{end_frame-start_frame} detektálva")
    
    def save_results(self):
        """Eredmények mentése"""
        if self.config['output']['save_data']:
            output_file = self.output_dir / 'ai_pupil_data.json'
            
            data = {
                'video_info': {
                    'path': self.video_path,
                    'fps': self.fps,
                    'resolution': [self.width, self.height],
                    'total_frames': self.frame_count
                },
                'method': 'MediaPipe Iris Detection',
                'detections': [p.to_dict() for p in self.results]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"AI eredmények mentve: {output_file}")
    
    def cleanup(self):
        """Cleanup"""
        self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.detector.cleanup()
    
    def run(self):
        """Pipeline futtatása"""
        try:
            self.process_video()
            self.save_results()
        finally:
            self.cleanup()


def main():
    """Fő függvény"""
    pipeline = AIEyeTrackingPipeline("config.yaml")
    pipeline.run()


if __name__ == "__main__":
    main()
