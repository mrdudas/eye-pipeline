#!/usr/bin/env python3
"""
Interactive Pipeline Tuner GUI
Interaktív felület a pupilla detektálási pipeline beállításához
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import yaml
from pathlib import Path
from threading import Thread
import json
from tqdm import tqdm
import sys
import subprocess
import platform
import time
import traceback
import torch

# EllSeg integration (primary detection method)
from ellseg_integration import EllSegDetector

# Camera calibration
from camera_calibration import CameraCalibrator

# Iris unwrapping
from iris_unwrapping import IrisUnwrapper


class PipelineTunerGUI:
    """Interaktív pipeline beállító GUI"""
    
    def __init__(self, video_path="eye1.mp4", config_path="config.yaml"):
        """GUI inicializálás"""
        self.video_path = video_path
        self.config_path = config_path
        
        # Detect dark mode for better color scheme
        self.is_dark_mode = self._detect_dark_mode()
        self.info_color = "yellow" if self.is_dark_mode else "blue"
        
        # Videó betöltése
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Nem sikerült megnyitni: {video_path}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Jelenlegi frame
        self.current_frame_num = 0
        self.original_frame = None
        self.processed_frame = None
        
        # Auto-play control
        self.is_playing = False
        self.play_interval = 80  # milliseconds (10 fps default)
        self.play_timer = None
        
        # Thread lock for video capture
        from threading import Lock
        self.video_lock = Lock()
        
        # EllSeg Model for robust ellipse detection
        print("Loading EllSeg model...")
        try:
            # Auto-detect CUDA/MPS availability
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon
                print("🚀 Using Apple Metal Performance Shaders (MPS)")
            else:
                device = 'cpu'
                print("⚠️  Using CPU (no GPU detected)")
            
            self.ellseg_detector = EllSegDetector(device=device)
            self.ellseg_available = self.ellseg_detector.model is not None
            if self.ellseg_available:
                print("✅ EllSeg loaded successfully")
            else:
                print("⚠️  EllSeg model not available")
        except Exception as e:
            print(f"Warning: Could not load EllSeg: {e}")
            self.ellseg_detector = None
            self.ellseg_available = False
        
        # Iris unwrapper
        self.iris_unwrapper = IrisUnwrapper()
        self.unwrapped_frontal = None
        self.unwrapped_polar = None
        self.unwrap_info = {}
        print("✅ Iris unwrapper initialized")
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_loaded = False
        self.load_camera_calibration()
        
        # GUI létrehozása
        self.create_gui()
        
        # Load config file at startup (if exists)
        self.load_settings_at_startup()
        
        # Első frame betöltése
        self.load_frame(0)
        self.update_preview()
    
    def create_gui(self):
        """GUI felület létrehozása"""
        self.root = tk.Tk()
        self.root.title("Pupilla Detection Pipeline Tuner")
        self.root.geometry("1600x900")
        
        # Fő konténer
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bal oldal: Kontrollok
        left_panel = ttk.Frame(main_container, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Jobb oldal: Képek
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # === KONTROLLOK ===
        self.create_controls(left_panel)
        
        # === KÉPEK ===
        self.create_image_displays(right_panel)
    
    def create_controls(self, parent):
        """Kontroll elemek létrehozása"""
        
        #Scrollable canvas
        canvas = tk.Canvas(parent, width=380)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # === 0. CAMERA CALIBRATION ===
        section0 = ttk.LabelFrame(scrollable_frame, text="0. Camera Undistortion", padding=10)
        section0.pack(fill=tk.X, pady=5)
        
        if self.calibration_loaded:
            ttk.Label(section0, text="✅ Calibration Loaded", foreground="green").pack()
            ttk.Label(section0, text=f"fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}",
                     font=("Courier", 9)).pack()
        else:
            ttk.Label(section0, text="⚠️ No Calibration", foreground="orange").pack()
        
        self.undistort_enabled = tk.BooleanVar(value=self.calibration_loaded)
        ttk.Checkbutton(section0, text="Enable Undistortion", 
                       variable=self.undistort_enabled,
                       command=self.update_preview).pack()
        
        calibration_btn_frame = ttk.Frame(section0)
        calibration_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(calibration_btn_frame, text="📹 Run Calibration", 
                  command=self.run_calibration_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(calibration_btn_frame, text="📂 Load Calibration", 
                  command=self.load_calibration_dialog).pack(side=tk.LEFT, padx=2)
        
        # === 1. IMAGE SELECTION ===
        section1 = ttk.LabelFrame(scrollable_frame, text="1. Image Selection", padding=10)
        section1.pack(fill=tk.X, pady=5)
        
        ttk.Label(section1, text=f"Total frames: {self.frame_count}").pack()
        ttk.Label(section1, text=f"FPS: {self.fps:.2f}").pack()
        
        # Frame control with play button
        frame_control = ttk.Frame(section1)
        frame_control.pack(pady=5)
        
        self.play_button = ttk.Button(frame_control, text="▶ Play", command=self.toggle_play, width=8)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.frame_slider = tk.Scale(
            frame_control, from_=0, to=self.frame_count-1, 
            orient=tk.HORIZONTAL, length=300,
            label="Frame Number",
            command=self.on_frame_change
        )
        self.frame_slider.pack(side=tk.LEFT)
        
        # === 2. GLINT REMOVAL ===
        section2 = ttk.LabelFrame(scrollable_frame, text="2. Glint Removal", padding=10)
        section2.pack(fill=tk.X, pady=5)
        
        self.glint_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(section2, text="Enable Glint Removal", 
                       variable=self.glint_enabled,
                       command=self.update_preview).pack()
        
        self.glint_threshold = tk.IntVar(value=240)
        self.create_slider(section2, "Threshold:", self.glint_threshold, 200, 255)
        
        self.glint_min_area = tk.IntVar(value=5)
        self.create_slider(section2, "Min Area:", self.glint_min_area, 1, 50)
        
        self.glint_max_area = tk.IntVar(value=200)
        self.create_slider(section2, "Max Area:", self.glint_max_area, 50, 500)
        
        self.glint_iterations = tk.IntVar(value=3)
        self.create_slider(section2, "Morph Iterations:", self.glint_iterations, 1, 10)
        
        # === 3. NOISE REDUCTION ===
        section3 = ttk.LabelFrame(scrollable_frame, text="3. Noise Reduction", padding=10)
        section3.pack(fill=tk.X, pady=5)
        
        self.noise_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(section3, text="Enable Noise Reduction", 
                       variable=self.noise_enabled,
                       command=self.update_preview).pack()
        
        self.noise_method = tk.StringVar(value="bilateral")
        ttk.Radiobutton(section3, text="Bilateral Filter", 
                       variable=self.noise_method, value="bilateral",
                       command=self.update_preview).pack(anchor=tk.W)
        ttk.Radiobutton(section3, text="Gaussian Blur", 
                       variable=self.noise_method, value="gaussian",
                       command=self.update_preview).pack(anchor=tk.W)
        ttk.Radiobutton(section3, text="Median Blur", 
                       variable=self.noise_method, value="median",
                       command=self.update_preview).pack(anchor=tk.W)
        
        self.noise_strength = tk.IntVar(value=5)
        self.create_slider(section3, "Strength:", self.noise_strength, 1, 15)
        
        # === 4. CLAHE / HISTOGRAM ===
        section4 = ttk.LabelFrame(scrollable_frame, text="4. CLAHE / Histogram", padding=10)
        section4.pack(fill=tk.X, pady=5)
        
        self.clahe_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(section4, text="Enable CLAHE", 
                       variable=self.clahe_enabled,
                       command=self.update_preview).pack()
        
        self.clahe_clip_limit = tk.DoubleVar(value=2.0)
        self.create_slider(section4, "Clip Limit:", self.clahe_clip_limit, 0.5, 10.0, resolution=0.1)
        
        self.clahe_tile_size = tk.IntVar(value=8)
        self.create_slider(section4, "Tile Size:", self.clahe_tile_size, 4, 32)
        
        # === 5. PUPIL DETECTION ===
        section5 = ttk.LabelFrame(scrollable_frame, text="5. Pupil & Iris Detection (Traditional CV)", padding=10)
        section5.pack(fill=tk.X, pady=5)
        
        # Pupil detection
        ttk.Label(section5, text="Pupil Detection:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        self.pupil_threshold = tk.IntVar(value=50)
        self.create_slider(section5, "Pupil Threshold:", self.pupil_threshold, 20, 100)
        
        self.pupil_min_area = tk.IntVar(value=100)
        self.create_slider(section5, "Pupil Min Area:", self.pupil_min_area, 50, 1000)
        
        self.pupil_morph_size = tk.IntVar(value=5)
        self.create_slider(section5, "Pupil Morph Kernel:", self.pupil_morph_size, 3, 15)
        
        # Iris detection (NEW!)
        ttk.Separator(section5, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(section5, text="Iris Detection (NEW!):", font=("Arial", 9, "bold"), 
                 foreground=self.info_color).pack(anchor=tk.W)
        
        self.iris_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(section5, text="Enable Iris Detection", 
                       variable=self.iris_enabled,
                       command=self.update_preview).pack()
        
        self.iris_threshold = tk.IntVar(value=80)
        self.create_slider(section5, "Iris Threshold:", self.iris_threshold, 50, 150)
        
        self.iris_min_area = tk.IntVar(value=5000)
        self.create_slider(section5, "Iris Min Area:", self.iris_min_area, 2000, 20000)
        
        self.iris_max_area = tk.IntVar(value=50000)
        self.create_slider(section5, "Iris Max Area:", self.iris_max_area, 10000, 100000)
        
        # === 5.5 ELLSEG ROBUST ELLIPSE DETECTION (NEW!) ===
        section5_5 = ttk.LabelFrame(scrollable_frame, text="⭐ 5.5. EllSeg Robust Detection (NEW!)", padding=10)
        section5_5.pack(fill=tk.X, pady=5)
        
        ttk.Label(section5_5, text="CNN-based ellipse detection (handles occlusions)", 
                 foreground=self.info_color, font=("Arial", 9)).pack()
        
        if self.ellseg_available:
            ttk.Label(section5_5, text="✅ EllSeg Model Loaded", foreground="green").pack()
        else:
            ttk.Label(section5_5, text="⚠️  EllSeg not available", foreground="orange").pack()
            ttk.Label(section5_5, text="Run: bash setup_ellseg.sh", 
                     font=("Courier", 8)).pack()
        
        self.ellseg_enabled = tk.BooleanVar(value=True)  # Enabled by default (RECOMMENDED)
        ttk.Checkbutton(section5_5, text="Enable EllSeg Detection", 
                       variable=self.ellseg_enabled,
                       command=self.update_preview).pack()
        
        self.ellseg_show_segmentation = tk.BooleanVar(value=True)
        ttk.Checkbutton(section5_5, text="Show Segmentation Overlay", 
                       variable=self.ellseg_show_segmentation,
                       command=self.update_preview).pack()
        
        self.ellseg_info_label = tk.StringVar(value="EllSeg not enabled")
        ttk.Label(section5_5, textvariable=self.ellseg_info_label, 
                 font=("Courier", 8), foreground="darkgreen").pack()
        
        # === ACTION BUTTONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(action_frame, text="🔄 Update Preview", 
                  command=self.update_preview).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🔍 Show Transformation Steps", 
                  command=self.show_transformation_steps).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🧪 Test on 50 Frames", 
                  command=lambda: self.run_test(50)).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🧪 Test on 100 Frames", 
                  command=lambda: self.run_test(100)).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🧪 Test on 1000 Frames", 
                  command=lambda: self.run_test(1000)).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🎬 Process Full Video", 
                  command=self.process_full_video).pack(fill=tk.X, pady=2)
        
        # Last video path
        self.last_video_path = None
        
        self.open_video_btn = ttk.Button(action_frame, text="🎬 Open Last Video", 
                  command=self.open_last_video, state='disabled')
        self.open_video_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="💾 Save Settings", 
                  command=self.save_settings).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="📂 Load Settings", 
                  command=self.load_settings).pack(fill=tk.X, pady=2)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(scrollable_frame, textvariable=self.status_var, 
                 foreground=self.info_color, font=("Arial", 10, "bold")).pack(pady=10)
    
    def create_slider(self, parent, label, variable, from_, to, resolution=1):
        """Slider létrehozása címkével"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
        
        value_label = ttk.Label(frame, text=str(variable.get()), width=8)
        value_label.pack(side=tk.RIGHT)
        
        slider = tk.Scale(
            frame, from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL, variable=variable,
            showvalue=False,
            command=lambda v: [value_label.config(text=f"{float(v):.1f}"), self.update_preview()]
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        return slider
    
    def create_image_displays(self, parent):
        """Képek megjelenítése"""
        # Felső sor: Original, Preprocessing, Result
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Original
        left_frame = ttk.LabelFrame(top_frame, text="Original Frame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.original_canvas = tk.Canvas(left_frame, bg="black")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Preprocessing
        mid_frame = ttk.LabelFrame(top_frame, text="After Preprocessing")
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.preprocessed_canvas = tk.Canvas(mid_frame, bg="black")
        self.preprocessed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Result
        right_frame = ttk.LabelFrame(top_frame, text="Detection Result")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_canvas = tk.Canvas(right_frame, bg="black")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # === BOTTOM ROW: Iris Frontal View ===
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Frontal view (unwrapped circular iris with pupil measurements)
        frontal_frame = ttk.LabelFrame(bottom_frame, text="�️ Iris Frontal View (Perspective Corrected)")
        frontal_frame.pack(fill=tk.BOTH, expand=True)
        self.frontal_canvas = tk.Canvas(frontal_frame, bg="black")
        self.frontal_canvas.pack(fill=tk.BOTH, expand=True)
    
    def on_frame_change(self, value):
        """Frame slider változás"""
        frame_num = int(float(value))
        self.load_frame(frame_num)
        self.update_preview()
    
    def _detect_dark_mode(self):
        """Detect if system is in dark mode"""
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(
                    ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                    capture_output=True,
                    text=True
                )
                return 'Dark' in result.stdout
            # Linux/Windows - default to light mode detection
            return False
        except:
            return False
    
    def load_frame(self, frame_num):
        """Frame betöltése thread-safe módon"""
        with self.video_lock:
            self.current_frame_num = frame_num
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if ret:
                self.original_frame = frame.copy()
            else:
                print(f"Nem sikerült betölteni a frame-t: {frame_num}")
    
    def preprocess_frame(self, frame, return_timing=False):
        """Frame előfeldolgozása a beállítások alapján
        
        Args:
            frame: Input frame
            return_timing: If True, return (processed_frame, timing_dict)
        
        Returns:
            processed_frame or (processed_frame, timing_dict)
        """
        import time
        
        timing = {}
        
        # Measure frame.copy() time
        t0 = time.perf_counter()
        processed = frame.copy()
        timing['frame_copy'] = time.perf_counter() - t0
        
        # 0. Camera undistortion (FIRST!)
        if self.undistort_enabled.get():
            t0_undistort = time.perf_counter()
            processed = self.undistort_frame(processed)
            timing['undistort'] = time.perf_counter() - t0_undistort
        else:
            timing['undistort'] = 0.0
        
        # 1. Glint removal
        if self.glint_enabled.get():
            t0_glint = time.perf_counter()
            processed = self.remove_glints(processed)
            timing['glint_removal'] = time.perf_counter() - t0_glint
        else:
            timing['glint_removal'] = 0.0
        
        # 2. Noise reduction
        noise_is_enabled = self.noise_enabled.get()
        if noise_is_enabled:
            t0_noise = time.perf_counter()
            processed = self.reduce_noise(processed)
            timing['noise_reduction'] = time.perf_counter() - t0_noise
        else:
            timing['noise_reduction'] = 0.0
        
        # 3. CLAHE
        if self.clahe_enabled.get():
            t0_clahe = time.perf_counter()
            processed = self.apply_clahe(processed)
            timing['clahe'] = time.perf_counter() - t0_clahe
        else:
            timing['clahe'] = 0.0
        
        if return_timing:
            return processed, timing
        return processed
    
    def remove_glints(self, frame):
        """Glint eltávolítása"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        threshold = self.glint_threshold.get()
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Blob filtering
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        glint_mask = np.zeros_like(bright_mask)
        min_area = self.glint_min_area.get()
        max_area = self.glint_max_area.get()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                cv2.drawContours(glint_mask, [contour], -1, 255, -1)
        
        # Morphological operations
        iterations = self.glint_iterations.get()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glint_mask = cv2.dilate(glint_mask, kernel, iterations=iterations)
        
        # Inpainting
        result = cv2.inpaint(frame, glint_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def reduce_noise(self, frame):
        """Zajszűrés"""
        method = self.noise_method.get()
        strength = self.noise_strength.get()
        
        if method == "bilateral":
            return cv2.bilateralFilter(frame, strength*2+1, strength*10, strength*10)
        elif method == "gaussian":
            ksize = strength*2+1
            return cv2.GaussianBlur(frame, (ksize, ksize), 0)
        elif method == "median":
            ksize = strength*2+1
            return cv2.medianBlur(frame, ksize)
        
        return frame
    
    def apply_clahe(self, frame):
        """CLAHE alkalmazása"""
        clip_limit = self.clahe_clip_limit.get()
        tile_size = self.clahe_tile_size.get()
        
        # BGR to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return result
    
    def detect_pupil_traditional(self, frame):
        """Hagyományos CV pupilla ÉS iris detektálás"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        annotated = frame.copy()
        pupil_data = None
        iris_data = None
        
        # === PUPIL DETECTION ===
        threshold = self.pupil_threshold.get()
        _, binary_pupil = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphology
        kernel_size = self.pupil_morph_size.get()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_pupil = cv2.morphologyEx(binary_pupil, cv2.MORPH_OPEN, kernel)
        binary_pupil = cv2.morphologyEx(binary_pupil, cv2.MORPH_CLOSE, kernel)
        
        # Contours
        contours_pupil, _ = cv2.findContours(binary_pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_area = self.pupil_min_area.get()
        valid_contours = [c for c in contours_pupil if cv2.contourArea(c) >= min_area]
        
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(annotated, ellipse, (0, 0, 255), 2)  # Red for pupil
                
                center = tuple(map(int, ellipse[0]))
                cv2.circle(annotated, center, 3, (0, 0, 255), -1)
                
                pupil_data = {
                    'center': center,
                    'axes': ellipse[1],
                    'angle': ellipse[2],
                    'area': cv2.contourArea(largest)
                }
                
                cv2.putText(annotated, "Pupil", (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # === IRIS DETECTION (NEW!) ===
        if self.iris_enabled.get():
            iris_threshold = self.iris_threshold.get()
            _, binary_iris = cv2.threshold(gray, iris_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Larger morphology for iris
            kernel_iris = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary_iris = cv2.morphologyEx(binary_iris, cv2.MORPH_OPEN, kernel_iris)
            binary_iris = cv2.morphologyEx(binary_iris, cv2.MORPH_CLOSE, kernel_iris)
            
            # Find contours
            contours_iris, _ = cv2.findContours(binary_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area (iris is bigger than pupil)
            min_iris_area = self.iris_min_area.get()
            max_iris_area = self.iris_max_area.get()
            valid_iris = [c for c in contours_iris 
                         if min_iris_area <= cv2.contourArea(c) <= max_iris_area]
            
            if valid_iris:
                # If we have pupil, find iris that contains it
                if pupil_data is not None:
                    pupil_center = pupil_data['center']
                    best_iris = None
                    best_dist = float('inf')
                    
                    for c in valid_iris:
                        if len(c) >= 5:
                            ellipse_iris = cv2.fitEllipse(c)
                            iris_center = tuple(map(int, ellipse_iris[0]))
                            dist = np.sqrt((iris_center[0] - pupil_center[0])**2 + 
                                         (iris_center[1] - pupil_center[1])**2)
                            if dist < best_dist:
                                best_dist = dist
                                best_iris = (c, ellipse_iris)
                    
                    if best_iris is not None and best_dist < 50:  # Max 50px distance
                        contour, ellipse_iris = best_iris
                        cv2.ellipse(annotated, ellipse_iris, (0, 255, 0), 2)  # Green for iris
                        
                        iris_center = tuple(map(int, ellipse_iris[0]))
                        
                        iris_data = {
                            'center': iris_center,
                            'axes': ellipse_iris[1],
                            'angle': ellipse_iris[2],
                            'area': cv2.contourArea(contour)
                        }
                        
                        cv2.putText(annotated, "Iris", (iris_center[0] + 10, iris_center[1] + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # No pupil, just take largest iris candidate
                    largest_iris = max(valid_iris, key=cv2.contourArea)
                    if len(largest_iris) >= 5:
                        ellipse_iris = cv2.fitEllipse(largest_iris)
                        cv2.ellipse(annotated, ellipse_iris, (0, 255, 0), 2)
                        
                        iris_center = tuple(map(int, ellipse_iris[0]))
                        
                        iris_data = {
                            'center': iris_center,
                            'axes': ellipse_iris[1],
                            'angle': ellipse_iris[2],
                            'area': cv2.contourArea(largest_iris)
                        }
        
        # Status display
        status_y = 30
        if pupil_data:
            cv2.putText(annotated, f"Pupil: {pupil_data['axes'][0]:.1f}x{pupil_data['axes'][1]:.1f}px", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_y += 25
        else:
            cv2.putText(annotated, "No pupil detected", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_y += 25
        
        if iris_data:
            cv2.putText(annotated, f"Iris: {iris_data['axes'][0]:.1f}x{iris_data['axes'][1]:.1f}px", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show ratio
            if pupil_data:
                ratio = pupil_data['axes'][0] / iris_data['axes'][0]
                cv2.putText(annotated, f"P/I Ratio: {ratio:.2f}", 
                           (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif self.iris_enabled.get():
            cv2.putText(annotated, "No iris detected", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated, pupil_data, iris_data

    
    def detect_ellseg(self, frame):
        """
        Detect pupil and iris ellipses using EllSeg CNN model
        Returns frame with annotations and ellipse data
        """
        if not self.ellseg_enabled.get():
            self.ellseg_info_label.set("EllSeg disabled")
            return frame, None, None
        
        if not self.ellseg_available or self.ellseg_detector is None:
            self.ellseg_info_label.set("⚠️  EllSeg model not loaded")
            cv2.putText(frame, "EllSeg not available", 
                       (10, frame.shape[0] - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            return frame, None, None
        
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        
        try:
            # Run EllSeg detection
            results = self.ellseg_detector.detect(gray)
            
            pupil_ellipse = results['pupil_ellipse']
            iris_ellipse = results['iris_ellipse']
            seg_map = results['seg_map']
            confidence = results['confidence']
            
            # Show segmentation overlay if enabled
            if self.ellseg_show_segmentation.get():
                overlay = annotated.copy()
                overlay[seg_map == 1] = [0, 255, 0]    # Iris = green
                overlay[seg_map == 2] = [255, 255, 0]  # Pupil = yellow
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
            
            # Draw pupil ellipse (red)
            if not np.all(pupil_ellipse == -1):
                cx, cy, a, b, angle = pupil_ellipse
                cv2.ellipse(annotated, 
                           (int(cx), int(cy)), 
                           (int(a), int(b)), 
                           np.rad2deg(angle), 
                           0, 360, (0, 0, 255), 2)
                cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                cv2.putText(annotated, "Pupil (EllSeg)", 
                           (int(cx) + 10, int(cy) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw iris ellipse (green)
            if not np.all(iris_ellipse == -1):
                cx, cy, a, b, angle = iris_ellipse
                cv2.ellipse(annotated, 
                           (int(cx), int(cy)), 
                           (int(a), int(b)), 
                           np.rad2deg(angle), 
                           0, 360, (0, 255, 0), 2)
                cv2.putText(annotated, "Iris (EllSeg)", 
                           (int(cx) + 10, int(cy) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display confidence
            info_y = frame.shape[0] - 120
            cv2.putText(annotated, f"EllSeg Confidence: {confidence:.2f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Format pupil/iris data for compatibility with existing code
            pupil_data = None
            iris_data = None
            
            if not np.all(pupil_ellipse == -1):
                cx, cy, a, b, angle = pupil_ellipse
                pupil_data = {
                    'center': (int(cx), int(cy)),
                    'ellipse': ((cx, cy), (2*a, 2*b), np.rad2deg(angle)),
                    'radius': (a + b) / 2,
                    'area': np.pi * a * b,
                    'source': 'ellseg'
                }
            
            if not np.all(iris_ellipse == -1):
                cx, cy, a, b, angle = iris_ellipse
                iris_data = {
                    'center': (int(cx), int(cy)),
                    'ellipse': ((cx, cy), (2*a, 2*b), np.rad2deg(angle)),
                    'radius': (a + b) / 2,
                    'area': np.pi * a * b,
                    'source': 'ellseg'
                }
            
            # Update info label
            pupil_pixels = np.sum(seg_map == 2)
            iris_pixels = np.sum(seg_map == 1)
            self.ellseg_info_label.set(
                f"Conf={confidence:.2f} | P_pix={pupil_pixels} I_pix={iris_pixels}"
            )
            
            # Store ellipses for transformation steps visualization
            self.last_iris_ellipse = iris_ellipse if not np.all(iris_ellipse == -1) else None
            self.last_pupil_ellipse = pupil_ellipse if not np.all(pupil_ellipse == -1) else None
            
            # IRIS PROCESSING (NEW!)
            # Get frontal view and pupil measurements
            if not np.all(iris_ellipse == -1):
                try:
                    results = self.iris_unwrapper.process_iris(
                        gray, 
                        iris_ellipse,
                        pupil_ellipse if not np.all(pupil_ellipse == -1) else iris_ellipse
                    )
                    
                    # Store for display
                    self.unwrapped_frontal = results['frontal_view']
                    self.unwrapped_polar = None  # No longer used
                    
                    # Combine all measurement info
                    self.unwrap_info = {
                        **results['frontal_info'],
                        'iris_ellipse_frontal': results.get('iris_ellipse_frontal'),
                        'pupil_ellipse_frontal': results.get('pupil_ellipse_frontal')
                    }
                    
                    # Add pupil measurements
                    if results['pupil_from_ellipse']:
                        self.unwrap_info['pupil_ellipse'] = results['pupil_from_ellipse']
                    if results['pupil_from_frontal']:
                        self.unwrap_info['pupil_frontal'] = results['pupil_from_frontal']
                    
                    # Add viewing angle and pupil size to display
                    y_info = info_y + 25
                    if 'viewing_angle_deg' in results['frontal_info']:
                        angle_deg = results['frontal_info']['viewing_angle_deg']
                        cv2.putText(annotated, f"View: {angle_deg:.1f}deg", 
                                   (10, y_info), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        y_info += 20
                    
                    # Add pupil measurements
                    if results['pupil_from_ellipse']:
                        pupil_mm = results['pupil_from_ellipse']['pupil_diameter_mm']
                        cv2.putText(annotated, f"Pupil: {pupil_mm:.2f}mm", 
                                   (10, y_info), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                except Exception as e:
                    print(f"Iris processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.unwrapped_frontal = None
                    self.unwrapped_polar = None
            else:
                self.unwrapped_frontal = None
                self.unwrapped_polar = None
            
            return annotated, pupil_data, iris_data
            
        except Exception as e:
            print(f"EllSeg detection error: {e}")
            import traceback
            traceback.print_exc()
            self.ellseg_info_label.set(f"Error: {str(e)[:30]}")
            return frame, None, None
    
    def update_preview(self, *args):
        """Előnézet frissítése"""
        if self.original_frame is None:
            return
        
        self.status_var.set("Processing...")
        self.root.update()
        
        # Preprocessing
        preprocessed = self.preprocess_frame(self.original_frame)
        
        # Store preprocessed frame for transformation steps visualization
        self.preprocessed_frame = preprocessed
        
        # Pupil & Iris Detection (Traditional CV or EllSeg)
        if self.ellseg_enabled.get() and self.ellseg_available:
            # Use EllSeg for robust detection (RECOMMENDED)
            result, pupil_data, iris_data = self.detect_ellseg(preprocessed)
        else:
            # Use traditional CV methods
            result, pupil_data, iris_data = self.detect_pupil_traditional(preprocessed)
        
        # Display
        self.display_image(self.original_frame, self.original_canvas)
        self.display_image(preprocessed, self.preprocessed_canvas)
        self.display_image(result, self.result_canvas)
        
        # Display unwrapped views (if available)
        if self.unwrapped_frontal is not None:
            # Draw ellipses on frontal view if available
            frontal_display = self.unwrapped_frontal.copy()
            
            # Check if we have transformed ellipses
            if hasattr(self, 'unwrap_info') and self.unwrap_info:
                iris_ellipse_frontal = self.unwrap_info.get('iris_ellipse_frontal')
                pupil_ellipse_frontal = self.unwrap_info.get('pupil_ellipse_frontal')
                pupil_frontal = self.unwrap_info.get('pupil_frontal')
                
                # Convert to BGR for color drawing
                if len(frontal_display.shape) == 2:
                    frontal_display = cv2.cvtColor(frontal_display, cv2.COLOR_GRAY2BGR)
                
                # Draw iris ellipse (green)
                if iris_ellipse_frontal is not None:
                    # Format: ((cx, cy), (width, height), angle_deg) from OpenCV
                    if isinstance(iris_ellipse_frontal, tuple) and len(iris_ellipse_frontal) == 3:
                        center, axes, angle = iris_ellipse_frontal
                        cv2.ellipse(frontal_display, 
                                   (int(center[0]), int(center[1])), 
                                   (int(axes[0]/2), int(axes[1]/2)),
                                   angle, 0, 360, (0, 255, 0), 2)
                    elif len(iris_ellipse_frontal) >= 5:
                        cx, cy, a, b, angle = iris_ellipse_frontal
                        cv2.ellipse(frontal_display, 
                                   (int(cx), int(cy)), 
                                   (int(a), int(b)),
                                   np.rad2deg(angle), 0, 360, (0, 255, 0), 2)
                
                # Draw pupil ellipse (blue)
                if pupil_ellipse_frontal is not None:
                    # Format: ((cx, cy), (width, height), angle_deg) from OpenCV
                    if isinstance(pupil_ellipse_frontal, tuple) and len(pupil_ellipse_frontal) == 3:
                        center, axes, angle = pupil_ellipse_frontal
                        cv2.ellipse(frontal_display, 
                                   (int(center[0]), int(center[1])), 
                                   (int(axes[0]/2), int(axes[1]/2)),
                                   angle, 0, 360, (255, 0, 0), 2)
                    elif len(pupil_ellipse_frontal) >= 5:
                        cx, cy, a, b, angle = pupil_ellipse_frontal
                        cv2.ellipse(frontal_display, 
                                   (int(cx), int(cy)), 
                                   (int(a), int(b)),
                                   np.rad2deg(angle), 0, 360, (255, 0, 0), 2)
                
                # Add pupil measurements text
                if pupil_frontal:
                    # Main measurement: diameter from area (pixels only)
                    text = f"Pupil D: {pupil_frontal['pupil_diameter_from_area_px']:.1f}px"
                    cv2.putText(frontal_display, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Additional info: area
                    if 'pupil_area_px' in pupil_frontal:
                        text_area = f"Area: {pupil_frontal['pupil_area_px']:.0f}px^2"
                        cv2.putText(frontal_display, text_area, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # Axes if available
                        if 'pupil_major_axis_px' in pupil_frontal:
                            text_axes = f"Axes: {pupil_frontal['pupil_major_axis_px']:.1f} x {pupil_frontal['pupil_minor_axis_px']:.1f}"
                            cv2.putText(frontal_display, text_axes, (10, 85),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            self.display_image(frontal_display, self.frontal_canvas)
        else:
            # Clear canvas
            self.frontal_canvas.delete("all")
        
        self.status_var.set(f"Frame {self.current_frame_num}/{self.frame_count-1}")
    
    def display_image(self, frame, canvas):
        """Kép megjelenítése canvas-on"""
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 400
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        h, w = frame_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to PIL
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)
        
        # Display
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference
    
    def run_test(self, num_frames):
        """Teszt futtatása N frame-en"""
        self.status_var.set(f"Testing on {num_frames} frames...")
        self.root.update()
        
        # Thread-ben futtatás hogy ne fagyjon a GUI
        thread = Thread(target=self._run_test_thread, args=(num_frames,))
        thread.start()
    
    def _run_test_thread(self, num_frames):
        """Teszt futtatása subprocess-ben - tiszta és gyors megoldás"""
        import subprocess
        import tempfile
        import time
        
        overall_start = time.perf_counter()
        start_frame = self.current_frame_num
        end_frame = min(start_frame + num_frames, self.frame_count)
        
        print(f"\n{'='*60}")
        print(f"🚀 LAUNCHING SUBPROCESS FOR VIDEO CREATION: {num_frames} frames")
        print(f"{'='*60}\n")
        
        # Save current settings to temp config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_config = Path(f.name)
            config_data = {
                'undistort': {'enabled': self.undistort_enabled.get()},
                'glint': {
                    'enabled': self.glint_enabled.get(),
                    'threshold': self.glint_threshold.get(),
                    'min_area': self.glint_min_area.get(),
                    'max_area': self.glint_max_area.get(),
                    'iterations': self.glint_iterations.get()
                },
                'noise': {
                    'enabled': self.noise_enabled.get(),
                    'method': self.noise_method.get(),
                    'strength': self.noise_strength.get()
                },
                'clahe': {
                    'enabled': self.clahe_enabled.get(),
                    'clip_limit': self.clahe_clip_limit.get(),
                    'tile_size': self.clahe_tile_size.get()
                },
                'ellseg': {
                    'enabled': self.ellseg_enabled.get(),
                    'show_segmentation': self.ellseg_show_segmentation.get()
                }
            }
            import yaml
            yaml.dump(config_data, f)
        
        # Output file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"test_frames_{start_frame}_to_{end_frame-1}.mp4"
        
        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            'pipeline_processor.py',
            '--config', str(temp_config),
            '--video', self.video_path,
            '--frames', str(num_frames),
            '--start-frame', str(start_frame),
            '--output', str(output_file)
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}\n")
        
        try:
            # Run subprocess with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output line by line
            detected_count = 0
            frames_processed = 0
            
            for line in process.stdout:
                print(line, end='')  # Print to console
                
                # Parse progress info
                if 'Frame' in line and '/' in line:
                    # Extract frame number and detection count
                    try:
                        parts = line.split('|')
                        frame_part = parts[0].strip()
                        if 'Detected:' in line:
                            detected_part = [p for p in parts if 'Detected:' in p][0]
                            detected_str = detected_part.split('Detected:')[1].split('(')[0].strip()
                            detected_count = int(detected_str.split('/')[0])
                            frames_processed = int(detected_str.split('/')[1])
                            
                            # Update GUI
                            progress = (frames_processed / num_frames) * 100
                            self.root.after(0, lambda p=progress, d=detected_count, f=frames_processed: 
                                          self.status_var.set(f"Subprocess: {p:.0f}% ({d}/{f} detected)"))
                    except:
                        pass  # Ignore parse errors
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            # Clean up temp config
            temp_config.unlink()
            
            overall_time = time.perf_counter() - overall_start
            
            print(f"\n{'='*60}")
            print(f"✅ SUBPROCESS COMPLETED")
            print(f"   Total wallclock time: {overall_time:.2f}s")
            print(f"{'='*60}\n")
            
            # Show results
            # Parse detection rate from output (or assume 100% if we got here)
            detection_rate = (detected_count / frames_processed * 100) if frames_processed > 0 else 100.0
            
            self.root.after(0, lambda: self.show_test_results(detection_rate, frames_processed, output_file))
            self.root.after(0, lambda: self.status_var.set(f"✅ Video ready: {output_file.name}"))
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Subprocess failed with code {e.returncode}")
            self.root.after(0, lambda: self.status_var.set(f"❌ Error: subprocess failed"))
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda e=e: self.status_var.set(f"❌ Error: {e}"))
            
        finally:
            # Clean up temp config if it still exists
            if temp_config.exists():
                temp_config.unlink()
    
    def show_test_results(self, detection_rate, num_frames, video_path):
        """Teszt eredmények megjelenítése és videó megnyitása"""
        # Enable open video button
        self.last_video_path = video_path
        self.open_video_btn.config(state='normal')
        
        result = messagebox.askquestion(
            "Test Results",
            f"Tested on {num_frames} frames\n\n"
            f"Detection Rate: {detection_rate:.1f}%\n"
            f"Detected: {int(detection_rate * num_frames / 100)}/{num_frames} frames\n\n"
            f"Video saved: {video_path}\n\n"
            f"Open video now?",
            icon='info'
        )
        
        if result == 'yes':
            self.open_video(video_path)
    
    def open_last_video(self):
        """Utoljára generált videó megnyitása"""
        if self.last_video_path and Path(self.last_video_path).exists():
            self.open_video(self.last_video_path)
        else:
            messagebox.showwarning("No Video", "No test video available. Run a test first!")
    
    def open_video(self, video_path):
        """Videó megnyitása a rendszer default lejátszóval"""
        import subprocess
        import platform
        
        video_path = str(video_path)
        
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', video_path])
            elif platform.system() == 'Windows':
                subprocess.run(['start', video_path], shell=True)
            else:  # Linux
                subprocess.run(['xdg-open', video_path])
            
            self.status_var.set(f"Video opened: {Path(video_path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video: {e}")
    
    def process_full_video(self):
        """Teljes videó feldolgozása subprocess-ben"""
        # Confirmation dialog
        result = messagebox.askquestion(
            "Process Full Video",
            f"This will process all {self.frame_count} frames.\n\n"
            f"This may take a while depending on video length.\n\n"
            f"Continue?",
            icon='warning'
        )
        
        if result != 'yes':
            return
        
        self.status_var.set(f"Processing full video ({self.frame_count} frames)...")
        self.root.update()
        
        # Run in thread (calls subprocess)
        thread = Thread(target=self._run_test_thread, args=(self.frame_count,))
        thread.start()
    
    def save_settings(self):
        """Beállítások mentése"""
        settings = {
            'camera': {
                'undistort_enabled': self.undistort_enabled.get(),
                'calibration_file': 'camera_calibration.yaml' if self.calibration_loaded else None
            },
            'glint': {
                'enabled': self.glint_enabled.get(),
                'threshold': self.glint_threshold.get(),
                'min_area': self.glint_min_area.get(),
                'max_area': self.glint_max_area.get(),
                'morph_iterations': self.glint_iterations.get()
            },
            'noise': {
                'enabled': self.noise_enabled.get(),
                'method': self.noise_method.get(),
                'strength': self.noise_strength.get()
            },
            'clahe': {
                'enabled': self.clahe_enabled.get(),
                'clip_limit': self.clahe_clip_limit.get(),
                'tile_size': self.clahe_tile_size.get()
            },
            'pupil': {
                'threshold': self.pupil_threshold.get(),
                'min_area': self.pupil_min_area.get(),
                'morph_kernel': self.pupil_morph_size.get()
            },
            'ellseg': {
                'enabled': self.ellseg_enabled.get(),
                'show_segmentation': self.ellseg_show_segmentation.get()
            }
        }
        
        output_file = "pipeline_settings.yaml"
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(settings, f, default_flow_style=False)
        
        messagebox.showinfo("Saved", f"Settings saved to {output_file}")
        self.status_var.set(f"Settings saved to {output_file}")
    
    def load_settings(self):
        """Beállítások betöltése"""
        try:
            with open("pipeline_settings.yaml", 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            # Camera
            if 'camera' in settings:
                self.undistort_enabled.set(settings['camera'].get('undistort_enabled', False))
            
            # Glint
            self.glint_enabled.set(settings['glint']['enabled'])
            self.glint_threshold.set(settings['glint']['threshold'])
            self.glint_min_area.set(settings['glint']['min_area'])
            self.glint_max_area.set(settings['glint']['max_area'])
            self.glint_iterations.set(settings['glint']['morph_iterations'])
            
            # Noise
            self.noise_enabled.set(settings['noise']['enabled'])
            self.noise_method.set(settings['noise']['method'])
            self.noise_strength.set(settings['noise']['strength'])
            
            # CLAHE
            self.clahe_enabled.set(settings['clahe']['enabled'])
            self.clahe_clip_limit.set(settings['clahe']['clip_limit'])
            self.clahe_tile_size.set(settings['clahe']['tile_size'])
            
            # Pupil
            if 'pupil' in settings:
                self.pupil_threshold.set(settings['pupil']['threshold'])
                self.pupil_min_area.set(settings['pupil']['min_area'])
                self.pupil_morph_size.set(settings['pupil']['morph_kernel'])
            
            # EllSeg
            if 'ellseg' in settings:
                self.ellseg_enabled.set(settings['ellseg']['enabled'])
                self.ellseg_show_segmentation.set(settings['ellseg']['show_segmentation'])
            
            self.update_preview()
            messagebox.showinfo("Loaded", "Settings loaded successfully")
            self.status_var.set("Settings loaded")
            
        except FileNotFoundError:
            messagebox.showerror("Error", "pipeline_settings.yaml not found")
    
    def load_settings_at_startup(self):
        """Load config file silently at startup (no popups)"""
        try:
            config_file = self.config_path if hasattr(self, 'config_path') else "pipeline_settings.yaml"
            
            # Try the specified config_path first, then fall back to pipeline_settings.yaml
            if not Path(config_file).exists():
                config_file = "pipeline_settings.yaml"
            
            if not Path(config_file).exists():
                print(f"⚠️  No config file found at startup (tried: {self.config_path}, pipeline_settings.yaml)")
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            # Camera
            if 'camera' in settings:
                self.undistort_enabled.set(settings['camera'].get('undistort_enabled', False))
            
            # Glint
            if 'glint' in settings:
                self.glint_enabled.set(settings['glint'].get('enabled', True))
                self.glint_threshold.set(settings['glint'].get('threshold', 240))
                self.glint_min_area.set(settings['glint'].get('min_area', 5))
                self.glint_max_area.set(settings['glint'].get('max_area', 200))
                self.glint_iterations.set(settings['glint'].get('morph_iterations', 3))
            
            # Noise
            if 'noise' in settings:
                self.noise_enabled.set(settings['noise'].get('enabled', True))
                self.noise_method.set(settings['noise'].get('method', 'bilateral'))
                self.noise_strength.set(settings['noise'].get('strength', 5))
            
            # CLAHE
            if 'clahe' in settings:
                self.clahe_enabled.set(settings['clahe'].get('enabled', False))
                self.clahe_clip_limit.set(settings['clahe'].get('clip_limit', 2.0))
                self.clahe_tile_size.set(settings['clahe'].get('tile_size', 8))
            
            # Pupil
            if 'pupil' in settings:
                self.pupil_threshold.set(settings['pupil'].get('threshold', 50))
                self.pupil_min_area.set(settings['pupil'].get('min_area', 100))
                self.pupil_morph_size.set(settings['pupil'].get('morph_kernel', 5))
            
            # Iris
            if 'iris' in settings:
                self.iris_enabled.set(settings['iris'].get('enabled', True))
                self.iris_threshold.set(settings['iris'].get('threshold', 80))
                self.iris_min_area.set(settings['iris'].get('min_area', 5000))
                self.iris_max_area.set(settings['iris'].get('max_area', 50000))
            
            # EllSeg
            if 'ellseg' in settings:
                self.ellseg_enabled.set(settings['ellseg'].get('enabled', True))
                self.ellseg_show_segmentation.set(settings['ellseg'].get('show_segmentation', True))
            
            print(f"✅ Config loaded from: {config_file}")
            self.status_var.set(f"Config loaded: {Path(config_file).name}")
            
        except Exception as e:
            print(f"⚠️  Could not load config at startup: {e}")
            # Don't show error popup at startup, just continue with defaults

    
    def load_camera_calibration(self, filename="camera_calibration.yaml"):
        """
        Kamera kalibráció betöltése
        """
        try:
            self.camera_matrix, self.dist_coeffs = CameraCalibrator.load_calibration(filename)
            self.calibration_loaded = True
            print(f"✅ Camera calibration loaded from {filename}")
        except FileNotFoundError:
            print(f"⚠️  No calibration file found: {filename}")
            self.calibration_loaded = False
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            self.calibration_loaded = False
    
    def run_calibration_dialog(self):
        """
        Kalibráció futtatása - dialog
        """
        # File selection dialog
        video_file = filedialog.askopenfilename(
            title="Select Calibration Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov"),
                ("All files", "*.*")
            ],
            initialfile="eye_cam.mkv"
        )
        
        if not video_file:
            return
        
        # Chessboard size dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Calibration Settings")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Chessboard Configuration", 
                 font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Chessboard size
        size_frame = ttk.Frame(dialog)
        size_frame.pack(pady=10)
        
        ttk.Label(size_frame, text="Columns (inner corners):").grid(row=0, column=0, sticky=tk.W, padx=5)
        cols_var = tk.IntVar(value=9)
        ttk.Spinbox(size_frame, from_=3, to=20, textvariable=cols_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(size_frame, text="Rows (inner corners):").grid(row=1, column=0, sticky=tk.W, padx=5)
        rows_var = tk.IntVar(value=6)
        ttk.Spinbox(size_frame, from_=3, to=20, textvariable=rows_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(size_frame, text="Square size (mm):").grid(row=2, column=0, sticky=tk.W, padx=5)
        square_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(size_frame, from_=0.1, to=100, textvariable=square_var, 
                   width=10, increment=0.1).grid(row=2, column=1, padx=5)
        
        # Max frames
        ttk.Label(size_frame, text="Max frames to use:").grid(row=3, column=0, sticky=tk.W, padx=5)
        frames_var = tk.IntVar(value=30)
        ttk.Spinbox(size_frame, from_=10, to=100, textvariable=frames_var, width=10).grid(row=3, column=1, padx=5)
        
        result = {"run": False}
        
        def on_ok():
            result["run"] = True
            result["cols"] = cols_var.get()
            result["rows"] = rows_var.get()
            result["square"] = square_var.get()
            result["frames"] = frames_var.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Run Calibration", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        
        if not result["run"]:
            return
        
        # Run calibration in thread
        def calibration_thread():
            self.status_var.set("Running calibration...")
            self.root.update()
            
            calibrator = CameraCalibrator(
                chessboard_size=(result["cols"], result["rows"]),
                square_size_mm=result["square"]
            )
            
            success, info = calibrator.calibrate_from_video(
                video_file,
                max_frames=result["frames"],
                show_detection=False
            )
            
            if success:
                # Save calibration
                output_file = "camera_calibration.yaml"
                calibrator.save_calibration(output_file)
                
                # Load into GUI
                self.camera_matrix = calibrator.camera_matrix
                self.dist_coeffs = calibrator.dist_coeffs
                self.calibration_loaded = True
                self.undistort_enabled.set(True)
                
                self.status_var.set(f"Calibration complete! Error: {info['reprojection_error_px']:.3f}px")
                messagebox.showinfo("Success", 
                                  f"Calibration successful!\n\n"
                                  f"Reprojection error: {info['reprojection_error_px']:.3f} px\n"
                                  f"Frames used: {info['frames_used']}\n"
                                  f"Saved to: {output_file}")
                
                # Update preview
                self.update_preview()
            else:
                self.status_var.set("Calibration failed!")
                messagebox.showerror("Error", 
                                   f"Calibration failed!\n\n{info.get('error', 'Unknown error')}")
        
        Thread(target=calibration_thread, daemon=True).start()
    
    def load_calibration_dialog(self):
        """
        Kalibráció betöltése file-ból
        """
        filename = filedialog.askopenfilename(
            title="Load Calibration File",
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*")
            ],
            initialfile="camera_calibration.yaml"
        )
        
        if not filename:
            return
        
        try:
            self.camera_matrix, self.dist_coeffs = CameraCalibrator.load_calibration(filename)
            self.calibration_loaded = True
            self.undistort_enabled.set(True)
            
            messagebox.showinfo("Success", f"Calibration loaded from:\n{filename}")
            self.status_var.set(f"Calibration loaded from {Path(filename).name}")
            
            # Update preview
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration:\n{str(e)}")
    
    def undistort_frame(self, frame):
        """
        Frame undistort ha enabled és van kalibráció
        """
        if not self.undistort_enabled.get():
            return frame
        
        if not self.calibration_loaded or self.camera_matrix is None:
            return frame
        
        try:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        except Exception as e:
            print(f"Undistort error: {e}")
            return frame
    
    def run(self):
        """GUI futtatása"""
        self.root.mainloop()
    
    def toggle_play(self):
        """Toggle auto-play of frames"""
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()
    
    def start_play(self):
        """Start auto-playing frames"""
        self.is_playing = True
        self.play_button.config(text="⏸ Pause")
        self.advance_frame()
    
    def stop_play(self):
        """Stop auto-playing frames"""
        self.is_playing = False
        self.play_button.config(text="▶ Play")
        if self.play_timer is not None:
            self.root.after_cancel(self.play_timer)
            self.play_timer = None
    
    def advance_frame(self):
        """Advance to next frame during auto-play"""
        if not self.is_playing:
            return
        
        # Move to next frame
        current = self.frame_slider.get()
        if current < self.frame_count - 1:
            self.frame_slider.set(current + 1)
            # Schedule next frame
            self.play_timer = self.root.after(self.play_interval, self.advance_frame)
        else:
            # Reached end, stop playing
            self.stop_play()
    
    def cleanup(self):
        """Cleanup"""
        # Stop auto-play if running
        if self.is_playing:
            self.stop_play()
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def show_transformation_steps(self):
        """Show all 6 transformation steps in a new window"""
        # Check if we have a frame loaded
        if not hasattr(self, 'original_frame') or self.original_frame is None:
            messagebox.showwarning("No Frame", "Please load a video and navigate to a frame first!")
            return
        
        # Check if we have preprocessed frame from last preview
        if not hasattr(self, 'preprocessed_frame') or self.preprocessed_frame is None:
            messagebox.showwarning("No Preview", "Please click 'Update Preview' first to detect ellipses!")
            return
        
        # Check if we have detection results
        if not hasattr(self, 'last_iris_ellipse') or not hasattr(self, 'last_pupil_ellipse'):
            messagebox.showwarning("No Detection", "Please click 'Update Preview' first to detect ellipses!")
            return
        
        if self.last_iris_ellipse is None or self.last_pupil_ellipse is None:
            messagebox.showwarning("No Detection", "No iris/pupil detected in current frame!")
            return
        
        # Use the preprocessed frame and ellipses from last detection
        preprocessed = self.preprocessed_frame
        
        # Convert to grayscale if needed
        if len(preprocessed.shape) == 3:
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        else:
            gray = preprocessed
        
        iris_ellipse = self.last_iris_ellipse
        pupil_ellipse = self.last_pupil_ellipse
        
        try:
            # Get transformation steps
            steps = self.iris_unwrapper.visualize_transformation_steps(
                gray, iris_ellipse, pupil_ellipse
            )
            
            # Create new window
            steps_window = tk.Toplevel(self.root)
            steps_window.title("🔍 Transformation Steps (6 Steps)")
            steps_window.geometry("1200x800")
            
            # Create scrollable frame
            canvas = tk.Canvas(steps_window)
            scrollbar = ttk.Scrollbar(steps_window, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Grid layout: 3 columns
            for idx, step in enumerate(steps):
                row = idx // 3
                col = idx % 3
                
                frame = ttk.LabelFrame(scrollable_frame, text=step['title'])
                frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Resize image to fit
                img = step['image']
                img_resized = cv2.resize(img, (350, 350))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                label = ttk.Label(frame, image=img_tk)
                label.image = img_tk  # Keep reference
                label.pack(padx=5, pady=5)
                
                # Add ellipse info
                if step['iris_ellipse']:
                    center, axes, angle = step['iris_ellipse']
                    info_text = f"Iris: {axes[0]/2:.0f}x{axes[1]/2:.0f}px"
                    ttk.Label(frame, text=info_text, foreground='green').pack()
                if step['pupil_ellipse']:
                    center, axes, angle = step['pupil_ellipse']
                    info_text = f"Pupil: {axes[0]/2:.0f}x{axes[1]/2:.0f}px"
                    ttk.Label(frame, text=info_text, foreground='blue').pack()
            
            # Configure grid weights
            for i in range(3):
                scrollable_frame.columnconfigure(i, weight=1)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate steps:\n{e}")
            import traceback
            traceback.print_exc()


def main():
    """Fő függvény"""
    app = PipelineTunerGUI("eye1.mp4", "config.yaml")
    
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
