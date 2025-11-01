#!/usr/bin/env python3
"""
Interactive Pipeline Tuner GUI
Interaktív felület a pupilla detektálási pipeline beállításához
"""

import sys
sys.path.append('./RITnet')

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
import torch
from models import model_dict
from camera_calibration import CameraCalibrator
from iris_model_3d import IrisPupilModel3D
from iris_model_3d_v2 import EyeballModel3D
from ellipse_iris_model import EllipseIrisPupilModel


class PipelineTunerGUI:
    """Interaktív pipeline beállító GUI"""
    
    def __init__(self, video_path="eye1.mp4", config_path="config.yaml"):
        """GUI inicializálás"""
        self.video_path = video_path
        self.config_path = config_path
        
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
        
        # Thread lock for video capture
        from threading import Lock
        self.video_lock = Lock()
        
        # Load RITnet model for eyelid detection
        print("Loading RITnet model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ritnet_model = model_dict['densenet']
        try:
            self.ritnet_model.load_state_dict(torch.load('./RITnet/best_model.pkl', map_location=self.device))
            self.ritnet_model.to(self.device)
            self.ritnet_model.eval()
            self.ritnet_available = True
            print(f"RITnet loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load RITnet model: {e}")
            self.ritnet_available = False
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_loaded = False
        self.load_camera_calibration()
        
        # 3D Iris Models - initialize all three models
        self.iris_model_original = IrisPupilModel3D(
            self.width, self.height, 
            camera_matrix=self.camera_matrix if self.calibration_loaded else None
        )
        self.iris_model_sphere = EyeballModel3D(
            self.width, self.height,
            camera_matrix=self.camera_matrix if self.calibration_loaded else None
        )
        self.iris_model_ellipse = EllipseIrisPupilModel(
            self.width, self.height
        )
        
        # Current active model (will be set based on dropdown selection)
        self.iris_model_3d = self.iris_model_original
        self.current_iris_params = None
        self.unwrapped_iris = None
        self.ritnet_mask = None
        
        # GUI létrehozása
        self.create_gui()
        
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
        
        self.frame_slider = tk.Scale(
            section1, from_=0, to=self.frame_count-1, 
            orient=tk.HORIZONTAL, length=350,
            label="Frame Number",
            command=self.on_frame_change
        )
        self.frame_slider.pack(pady=5)
        
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
        section5 = ttk.LabelFrame(scrollable_frame, text="5. Pupil Detection (Traditional CV)", padding=10)
        section5.pack(fill=tk.X, pady=5)
        
        self.pupil_threshold = tk.IntVar(value=50)
        self.create_slider(section5, "Threshold:", self.pupil_threshold, 20, 100)
        
        self.pupil_min_area = tk.IntVar(value=100)
        self.create_slider(section5, "Min Area:", self.pupil_min_area, 50, 1000)
        
        self.pupil_morph_size = tk.IntVar(value=5)
        self.create_slider(section5, "Morph Kernel:", self.pupil_morph_size, 3, 15)
        
        # === 6. EYELID DETECTION (RITnet) ===
        section6 = ttk.LabelFrame(scrollable_frame, text="6. Eyelid Detection (RITnet AI)", padding=10)
        section6.pack(fill=tk.X, pady=5)
        
        if self.ritnet_available:
            ttk.Label(section6, text="✅ RITnet Model Loaded", foreground="green").pack()
        else:
            ttk.Label(section6, text="⚠️ RITnet Not Available", foreground="red").pack()
        
        self.eyelid_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(section6, text="Enable Eyelid Detection", 
                       variable=self.eyelid_enabled,
                       command=self.update_preview).pack()
        
        self.show_segmentation = tk.BooleanVar(value=True)
        ttk.Checkbutton(section6, text="Show Segmentation Overlay", 
                       variable=self.show_segmentation,
                       command=self.update_preview).pack()
        
        self.show_eyelid_boundaries = tk.BooleanVar(value=True)
        ttk.Checkbutton(section6, text="Show Eyelid Boundaries", 
                       variable=self.show_eyelid_boundaries,
                       command=self.update_preview).pack()
        
        self.show_vertical_axis = tk.BooleanVar(value=True)
        ttk.Checkbutton(section6, text="Show Vertical Axis", 
                       variable=self.show_vertical_axis,
                       command=self.update_preview).pack()
        
        # === 7. 3D IRIS MODEL (NEW!) ===
        section7 = ttk.LabelFrame(scrollable_frame, text="⭐ 7. 3D Iris Model (NEW!)", padding=10)
        section7.pack(fill=tk.X, pady=5)
        
        ttk.Label(section7, text="Fit 3D model to RITnet masks", 
                 foreground="blue", font=("Arial", 9)).pack()
        
        # Model Selection Dropdown
        model_frame = ttk.Frame(section7)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Model Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.iris_model_type = tk.StringVar(value="Ellipse-based (Best)")
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.iris_model_type, 
                                      state="readonly", width=25)
        model_dropdown['values'] = (
            "Ellipse-based (Best)",
            "Original (Simple 3D)",
            "Sphere-based (Physical)"
        )
        model_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_type_changed)
        
        # Info label for selected model
        self.model_info_label = tk.StringVar(value="IoU ~1.0, Fast (<0.5s)")
        ttk.Label(section7, textvariable=self.model_info_label, 
                 font=("Arial", 8), foreground="darkblue", style="Info.TLabel").pack()
        
        self.iris_model_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(section7, text="Enable 3D Iris Model", 
                       variable=self.iris_model_enabled,
                       command=self.update_preview).pack()
        
        self.show_iris_overlay = tk.BooleanVar(value=True)
        ttk.Checkbutton(section7, text="Show Model Overlay", 
                       variable=self.show_iris_overlay,
                       command=self.update_preview).pack()
        
        self.show_unwrapped = tk.BooleanVar(value=True)
        ttk.Checkbutton(section7, text="Show Unwrapped Iris", 
                       variable=self.show_unwrapped,
                       command=self.update_preview).pack()
        
        # Model parameters display
        self.iris_params_label = tk.StringVar(value="No model fitted yet")
        ttk.Label(section7, textvariable=self.iris_params_label, 
                 font=("Courier", 8), foreground="darkgreen").pack()
        
        self.optimization_method = tk.StringVar(value="de")
        opt_frame = ttk.Frame(section7)
        opt_frame.pack(fill=tk.X, pady=5)
        ttk.Label(opt_frame, text="Optimization:").pack(side=tk.LEFT)
        ttk.Radiobutton(opt_frame, text="Fast (Nelder-Mead)", 
                       variable=self.optimization_method, value="nelder-mead").pack(side=tk.LEFT)
        ttk.Radiobutton(opt_frame, text="Accurate (DE)", 
                       variable=self.optimization_method, value="de").pack(side=tk.LEFT)
        
        # === ACTION BUTTONS ===
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(action_frame, text="🔄 Update Preview", 
                  command=self.update_preview).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🧪 Test on 50 Frames", 
                  command=lambda: self.run_test(50)).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="🧪 Test on 100 Frames", 
                  command=lambda: self.run_test(100)).pack(fill=tk.X, pady=2)
        
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
                 foreground="blue", font=("Arial", 10, "bold")).pack(pady=10)
    
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
        
        # Alsó sor: Unwrapped Iris (NEW!)
        bottom_frame = ttk.LabelFrame(parent, text="⭐ Unwrapped Iris (Frontal View)")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        self.unwrapped_canvas = tk.Canvas(bottom_frame, bg="black")
        self.unwrapped_canvas.pack(fill=tk.BOTH, expand=True)
    
    def on_frame_change(self, value):
        """Frame slider változás"""
        frame_num = int(float(value))
        self.load_frame(frame_num)
        self.update_preview()
    
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
    
    def preprocess_frame(self, frame):
        """Frame előfeldolgozása a beállítások alapján"""
        processed = frame.copy()
        
        # 0. Camera undistortion (FIRST!)
        processed = self.undistort_frame(processed)
        
        # 1. Glint removal
        if self.glint_enabled.get():
            processed = self.remove_glints(processed)
        
        # 2. Noise reduction
        if self.noise_enabled.get():
            processed = self.reduce_noise(processed)
        
        # 3. CLAHE
        if self.clahe_enabled.get():
            processed = self.apply_clahe(processed)
        
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
        """Hagyományos CV pupilla detektálás - JAVÍTOTT"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        threshold = self.pupil_threshold.get()
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphology
        kernel_size = self.pupil_morph_size.get()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated = frame.copy()
        pupil_data = None
        
        # Filter by area
        min_area = self.pupil_min_area.get()
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(annotated, ellipse, (0, 255, 0), 2)
                
                center = tuple(map(int, ellipse[0]))
                cv2.circle(annotated, center, 5, (0, 0, 255), -1)
                
                pupil_data = {
                    'center': center,
                    'axes': ellipse[1],
                    'angle': ellipse[2]
                }
                
                cv2.putText(annotated, "Pupil Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"D: {ellipse[1][0]:.1f}px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "No pupil detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated, pupil_data
    
    def preprocess_for_ritnet(self, frame):
        """
        Preprocess frame for RITnet inference
        1. Gamma correction (0.8)
        2. CLAHE (clipLimit=1.5, tileGridSize=(8,8))
        3. Normalization (mean=0.5, std=0.5)
        4. Resize to 640x400
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Gamma correction
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Resize to 640x400
        resized = cv2.resize(gray, (640, 400))
        
        # Normalize
        normalized = (resized / 255.0 - 0.5) / 0.5
        
        # Convert to tensor: [1, 1, 400, 640]
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def detect_eyelids_ritnet(self, frame, pupil_data=None):
        """
        Detect eyelids using RITnet semantic segmentation
        Returns frame with annotations, eyelid data, and segmentation mask
        """
        if not self.eyelid_enabled.get() or not self.ritnet_available:
            return frame, None, None
        
        annotated = frame.copy()
        original_size = frame.shape[:2]
        
        try:
            # Preprocess for RITnet
            input_tensor = self.preprocess_for_ritnet(frame)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.ritnet_model(input_tensor)
            
            # Get segmentation mask
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Resize back to original size
            mask = cv2.resize(pred.astype(np.uint8), 
                            (original_size[1], original_size[0]),
                            interpolation=cv2.INTER_NEAREST)
            
            # Show segmentation overlay
            if self.show_segmentation.get():
                colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                colored_mask[mask == 1] = [255, 0, 0]    # Sclera: Red
                colored_mask[mask == 2] = [0, 255, 0]    # Iris: Green
                colored_mask[mask == 3] = [0, 0, 255]    # Pupil: Blue
                annotated = cv2.addWeighted(annotated, 0.7, colored_mask, 0.3, 0)
            
            # Extract eyelid boundaries
            eyelid_data = {}
            
            # Find eye region (non-background)
            eye_region = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(eye_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (main eye region)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Find topmost and bottommost points (eyelid boundaries)
                topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
                rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
                
                eyelid_data = {
                    'upper': topmost,
                    'lower': bottommost,
                    'left': leftmost,
                    'right': rightmost
                }
                
                # Draw eyelid boundaries
                if self.show_eyelid_boundaries.get():
                    cv2.circle(annotated, topmost, 5, (0, 255, 255), -1)
                    cv2.circle(annotated, bottommost, 5, (0, 255, 255), -1)
                    cv2.putText(annotated, "Upper", (topmost[0]+10, topmost[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(annotated, "Lower", (bottommost[0]+10, bottommost[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Draw vertical axis
                if self.show_vertical_axis.get():
                    cv2.line(annotated, topmost, bottommost, (255, 255, 0), 2)
                    
                    # Eye center point
                    eye_center_y = (topmost[1] + bottommost[1]) // 2
                    eye_center_x = (leftmost[0] + rightmost[0]) // 2
                    cv2.circle(annotated, (eye_center_x, eye_center_y), 5, (255, 0, 255), -1)
                    
                    # Eye height
                    eye_height = bottommost[1] - topmost[1]
                    cv2.putText(annotated, f"Eye height: {eye_height}px", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # If pupil is detected, show relative position
                if pupil_data and pupil_data.get('center'):
                    pupil_y = pupil_data['center'][1]
                    relative_pos = (pupil_y - topmost[1]) / (bottommost[1] - topmost[1])
                    cv2.putText(annotated, f"Pupil Y pos: {relative_pos:.2f}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Store mask for 3D model fitting
            self.ritnet_mask = mask
            
            return annotated, eyelid_data, mask
            
        except Exception as e:
            print(f"Error in RITnet detection: {e}")
            cv2.putText(annotated, f"RITnet Error: {str(e)[:30]}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return annotated, None, None
    
    def fit_3d_iris_model(self, frame, preprocessed, ritnet_mask):
        """
        Fit 3D iris-pupil model to RITnet segmentation mask
        Returns frame with model overlay and unwrapped iris display
        """
        if not self.iris_model_enabled.get() or ritnet_mask is None:
            self.current_iris_params = None
            self.unwrapped_iris = None
            self.iris_params_label.set("3D model disabled")
            return frame
        
        annotated = frame.copy()
        
        try:
            # Fit 3D model to mask
            params = self.iris_model_3d.fit_to_mask(
                ritnet_mask, 
                method=self.optimization_method.get()
            )
            
            self.current_iris_params = params
            
            # Update parameter display
            param_text = (f"θ={params['theta_deg']:.1f}° φ={params['phi_deg']:.1f}° | "
                         f"r_pupil={params['r_pupil']:.1f} r_iris={params['r_iris']:.1f} | "
                         f"IoU: P={params['iou_pupil']:.2f} I={params['iou_iris']:.2f}")
            self.iris_params_label.set(param_text)
            
            # Draw model overlay
            if self.show_iris_overlay.get():
                # Draw iris boundary (green)
                iris_3d = self.iris_model_3d.generate_circle_points(
                    params['r_iris'], num_points=100
                )
                iris_2d = self.iris_model_3d.project_to_image(
                    iris_3d, params['theta'], params['phi'],
                    params['cx'], params['cy'], params['distance']
                )
                iris_2d = iris_2d[~np.isnan(iris_2d).any(axis=1)]
                if len(iris_2d) > 2:
                    iris_contour = iris_2d.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [iris_contour], True, (0, 255, 0), 2)
                
                # Draw pupil boundary (blue)
                pupil_3d = self.iris_model_3d.generate_circle_points(
                    params['r_pupil'], num_points=100
                )
                pupil_2d = self.iris_model_3d.project_to_image(
                    pupil_3d, params['theta'], params['phi'],
                    params['cx'], params['cy'], params['distance']
                )
                pupil_2d = pupil_2d[~np.isnan(pupil_2d).any(axis=1)]
                if len(pupil_2d) > 2:
                    pupil_contour = pupil_2d.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pupil_contour], True, (255, 0, 0), 2)
                
                # Draw center
                cv2.circle(annotated, (int(params['cx']), int(params['cy'])), 
                          5, (0, 255, 255), -1)
                
                # Draw rotation angles
                cv2.putText(annotated, f"Pitch: {params['theta_deg']:.1f}deg", 
                           (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(annotated, f"Yaw: {params['phi_deg']:.1f}deg", 
                           (10, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Generate unwrapped iris
            if self.show_unwrapped.get():
                self.unwrapped_iris = self.iris_model_3d.unwrap_iris(
                    preprocessed, params, output_size=(256, 64)
                )
            else:
                self.unwrapped_iris = None
            
            return annotated
            
        except Exception as e:
            print(f"Error in 3D iris model fitting: {e}")
            import traceback
            traceback.print_exc()
            cv2.putText(annotated, f"3D Model Error: {str(e)[:40]}", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self.current_iris_params = None
            self.unwrapped_iris = None
            self.iris_params_label.set(f"Error: {str(e)[:50]}")
            return annotated
    
    def update_preview(self, *args):
        """Előnézet frissítése"""
        if self.original_frame is None:
            return
        
        self.status_var.set("Processing...")
        self.root.update()
        
        # Preprocessing
        preprocessed = self.preprocess_frame(self.original_frame)
        
        # Pupil Detection
        result, pupil_data = self.detect_pupil_traditional(preprocessed)
        
        # Eyelid Detection (RITnet) - returns mask as well
        result, eyelid_data, ritnet_mask = self.detect_eyelids_ritnet(result, pupil_data)
        
        # 3D Iris Model (NEW!)
        result = self.fit_3d_iris_model(result, preprocessed, ritnet_mask)
        
        # Display
        self.display_image(self.original_frame, self.original_canvas)
        self.display_image(preprocessed, self.preprocessed_canvas)
        self.display_image(result, self.result_canvas)
        
        # Display unwrapped iris if available
        if self.unwrapped_iris is not None and self.show_unwrapped.get():
            self.display_image(self.unwrapped_iris, self.unwrapped_canvas)
        else:
            # Clear unwrapped canvas
            self.unwrapped_canvas.delete("all")
            self.unwrapped_canvas.create_text(
                self.unwrapped_canvas.winfo_width()//2, 
                self.unwrapped_canvas.winfo_height()//2,
                text="Unwrapped iris disabled or not available",
                fill="gray", font=("Arial", 12)
            )
        
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
        """Teszt futtatása thread-ben és videó generálása"""
        start_frame = self.current_frame_num
        end_frame = min(start_frame + num_frames, self.frame_count)
        
        results = []
        
        # Videó kimenet előkészítése
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"test_frames_{start_frame}_to_{end_frame-1}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Side-by-side: original + result (2x width)
        out = cv2.VideoWriter(str(output_file), fourcc, self.fps, (self.width * 2, self.height))
        
        detected_count = 0
        
        # Külön videó capture a thread számára
        test_cap = cv2.VideoCapture(self.video_path)
        
        try:
            for frame_num in range(start_frame, end_frame):
                test_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = test_cap.read()
                
                if not ret:
                    print(f"Warning: Failed to read frame {frame_num}")
                    continue
                
                # Preprocess
                preprocessed = self.preprocess_frame(frame)
                
                # Detect pupil
                result_frame, pupil_data = self.detect_pupil_traditional(preprocessed)
                
                # Detect eyelids with RITnet
                result_frame, eyelid_data = self.detect_eyelids_ritnet(result_frame, pupil_data)
                
                # Check detection
                detected = pupil_data is not None
                
                results.append(detected)
                if detected:
                    detected_count += 1
                
                # Frame info hozzáadása az originalhoz
                info_frame = frame.copy()
                cv2.putText(info_frame, f"Frame: {frame_num}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_frame, f"Original", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Detection status a result frame-re
                status_color = (0, 255, 0) if detected else (0, 0, 255)
                status_text = "DETECTED" if detected else "NOT DETECTED"
                cv2.putText(result_frame, status_text, (10, self.height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                current_rate = (detected_count / (frame_num - start_frame + 1)) * 100
                cv2.putText(result_frame, f"Rate: {current_rate:.1f}%", (10, self.height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Side-by-side: original + detection result
                combined = np.hstack([info_frame, result_frame])
                out.write(combined)
                
                # Update status
                progress = (frame_num - start_frame + 1) / num_frames * 100
                self.root.after(0, lambda p=progress, d=detected_count, f=frame_num-start_frame+1: 
                              self.status_var.set(f"Creating video: {p:.0f}% ({d}/{f} detected)"))
        
        except Exception as e:
            print(f"Error during test: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
        
        finally:
            out.release()
            test_cap.release()
        
        # Calculate statistics
        detection_rate = sum(results) / len(results) * 100 if results else 0
        
        # Show result and open video
        self.root.after(0, lambda: self.show_test_results(detection_rate, len(results), output_file))
        self.root.after(0, lambda: self.status_var.set(f"Video ready: {output_file.name}"))
    
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
            'eyelid': {
                'enabled': self.eyelid_enabled.get(),
                'show_segmentation': self.show_segmentation.get(),
                'show_boundaries': self.show_eyelid_boundaries.get(),
                'show_vertical_axis': self.show_vertical_axis.get()
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
            
            # Eyelid
            if 'eyelid' in settings:
                self.eyelid_enabled.set(settings['eyelid']['enabled'])
                self.show_segmentation.set(settings['eyelid']['show_segmentation'])
                self.show_eyelid_boundaries.set(settings['eyelid']['show_boundaries'])
                self.show_vertical_axis.set(settings['eyelid']['show_vertical_axis'])
            
            self.update_preview()
            messagebox.showinfo("Loaded", "Settings loaded successfully")
            self.status_var.set("Settings loaded")
            
        except FileNotFoundError:
            messagebox.showerror("Error", "pipeline_settings.yaml not found")
    
    def on_model_type_changed(self, event=None):
        """
        Handle model type dropdown change
        """
        model_type = self.iris_model_type.get()
        
        if model_type == "Ellipse-based (Best)":
            self.iris_model_3d = self.iris_model_ellipse
            self.model_info_label.set("IoU ~1.0, Fast (<0.5s), Direct ellipse fitting")
        elif model_type == "Original (Simple 3D)":
            self.iris_model_3d = self.iris_model_original
            self.model_info_label.set("IoU ~0.97, Medium (2-3s), Simplified 3D projection")
        elif model_type == "Sphere-based (Physical)":
            self.iris_model_3d = self.iris_model_sphere
            self.model_info_label.set("IoU ~0.4-0.7, Slow (3-4s), Physical eyeball sphere")
        
        # Clear cached results
        self.current_iris_params = None
        self.unwrapped_iris = None
        
        # Update preview with new model
        if self.iris_model_enabled.get():
            self.update_preview()
    
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
    
    def cleanup(self):
        """Cleanup"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Fő függvény"""
    app = PipelineTunerGUI("eye1.mp4", "config.yaml")
    
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
