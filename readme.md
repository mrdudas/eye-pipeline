# Eye Pipeline - Pupilla Detection with RITnet

## Project Overview

Accurate pupil detection and tracking pipeline for Near-IR eye tracking video (eye1.mp4: 400×400px, 111.84fps, 45,649 frames).

## Features

### ✅ Completed

1. **Interactive GUI** (`pipeline_tuner_gui.py`)
   - 6-step pipeline with real-time preview
   - Parameter tuning for all stages
   - Video generation for testing
   - Settings save/load (YAML)

2. **Pipeline Steps**
   - **Step 0**: Camera Undistortion (geometric correction)
   - **Step 1**: Image Selection (frame slider)
   - **Step 2**: Glint Removal (threshold, area, morphology)
   - **Step 3**: Noise Reduction (bilateral/gaussian/median)
   - **Step 4**: CLAHE (contrast enhancement)
   - **Step 5**: Pupil & Iris Detection (traditional CV)
   - **Step 5.5**: **EllSeg Robust Detection (CNN-based)** ⭐ NEW!
   - **Step 6**: Eyelid Detection (RITnet AI)
   - **Step 7**: 3D Iris Model (orientation & unwrapping)

3. **EllSeg Integration** ⭐ NEW!
   - **CNN-based robust ellipse detection** for pupil & iris
   - **Handles eyelid occlusions** (vertical axis problem solved!)
   - DenseNet2D (RITnet_v3) architecture
   - Pre-trained on OpenEDS, RITEyes, LPW, NVGaze
   - Outputs: Segmentation map + ellipse parameters
   - 10%+ pupil center, 24%+ iris center improvement
   - See [ELLSEG_INTEGRATION.md](ELLSEG_INTEGRATION.md) for details

4. **RITnet Integration**
   - Semantic segmentation (sclera, iris, pupil)
   - Eyelid boundary detection
   - Vertical axis determination
   - 95%+ accuracy, 100+ fps
   - Near-IR optimized

4. **Camera Calibration**
   - OpenCV calibrateCamera with chessboard
   - Geometric distortion correction
   - YAML persistence (fx, fy, cx, cy, distortion coeffs)
   - 0.18px reprojection error
   - Real-time undistortion

5. **3D Iris Model** ⭐ NEW!
   - **Three model implementations** with GUI selection:
     - **Ellipse-based (Recommended)**: Direct ellipse fitting, IoU ~1.0, <0.5s ⭐
     - **Original (Simple 3D)**: Simplified 3D projection, IoU ~0.97, 2-3s
     - **Sphere-based (Physical)**: Full eyeball model, IoU ~0.4-0.7, 3-4s
   - Fit 3D concentric circles (pupil + iris) to RITnet masks
   - Estimate 3D orientation: pitch (θ) and yaw (φ) angles
   - Perspective projection with camera matrix
   - Iris unwrapping to frontal view (256×64 polar coordinates)
   - Applications: gaze estimation, iris recognition, quality assessment
   - **Model Comparison**: See [MODEL_COMPARISON.md](MODEL_COMPARISON.md)

6. **Video Generation**
   - Side-by-side original|detection output
   - 50/100 frame test videos
   - One-click open functionality
   - Thread-safe processing

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo>
cd eye_pipeline

# Install dependencies
pip install opencv-python numpy matplotlib scikit-image scipy pyyaml tqdm Pillow
pip install torch torchvision torchaudio

# Clone RITnet
git clone https://github.com/AayushKrChaudhary/RITnet.git
```

### Run GUI

```bash
python pipeline_tuner_gui.py
```

## Documentation

- **[ELLSEG_INTEGRATION.md](ELLSEG_INTEGRATION.md)** - EllSeg robust detection (handles occlusions) ⭐ NEW!
- **[MODEL_COMPARISON.md](MODEL_COMPARISON.md)** - Comparison of 3 iris model implementations
- **[IRIS_3D_MODEL.md](IRIS_3D_MODEL.md)** - Original 3D iris-pupil model documentation
- **[CAMERA_CALIBRATION.md](CAMERA_CALIBRATION.md)** - Camera calibration and undistortion
- **[RITNET_INTEGRATION.md](RITNET_INTEGRATION.md)** - RITnet eyelid detection setup and usage
- **[EYELID_DETECTION_RESEARCH.md](EYELID_DETECTION_RESEARCH.md)** - Model comparison and research
- **[VIDEO_GENERATION_GUIDE.md](VIDEO_GENERATION_GUIDE.md)** - Video testing feature
- **[GUI_USAGE_GUIDE.md](GUI_USAGE_GUIDE.md)** - GUI usage instructions
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Installation and setup instructions

## Architecture

```
Eye1.mp4 (Near-IR, 400×400)
    ↓
Step 0: Camera Undistortion
    ↓
Step 1: Frame Selection
    ↓
Step 2: Glint Removal
    ↓
Step 3: Noise Reduction
    ↓
Step 4: CLAHE Enhancement
    ↓
Step 5: Pupil & Iris Detection (Traditional CV)
    ↓
Step 5.5: EllSeg Robust Detection (CNN) ⭐ NEW!
    ├── Segmentation (iris/pupil masks)
    ├── Ellipse regression
    └── Handles eyelid occlusions
    ↓
Step 6: Eyelid Detection (RITnet AI)
    ↓
Step 7: 3D Iris Model (NEW!)
    ├── Fit 3D concentric circles
    ├── Estimate orientation (θ, φ)
    └── Unwrap to frontal view
    ↓
Output: Pupil + Eyelid + 3D Orientation + Unwrapped Iris
```

## Technical Details

### Video Specifications
- **File**: eye1.mp4
- **Resolution**: 400×400 pixels
- **FPS**: 111.84
- **Frames**: 45,649
- **Type**: Near-IR (infrared)

### RITnet Model
- **Architecture**: DenseNet U-Net
- **Input**: 640×400 grayscale
- **Output**: 4-class segmentation (background, sclera, iris, pupil)
- **Accuracy**: 95.78% validation
- **Speed**: 100+ fps on CPU

### Output Data

```python
{
    'pupil': {
        'center': (x, y),
        'axes': (major, minor),
        'angle': theta
    },
    'eyelid': {
        'upper': (x, y),
        'lower': (x, y),
        'left': (x, y),
        'right': (x, y)
    }
}
```

## Next Steps

### Advanced Features (TODO)

Camera Calibration - ✅ DONE!
- ✅ OpenCV calibrateCamera with 9×6 chessboard
- ✅ YAML persistence (fx, fy, cx, cy, distCoeffs)
- ✅ Real-time undistortion in pipeline
- ✅ 0.18px reprojection error

Pupil diameter mm-ben
- Pixel átmérő = 2 × ellipszis kisebbik fél-tengely.
- Mm/pixel skála: kalibrációs target a szem síkjában, vagy stabil, ismert munkatávolság + intrinzik.
- Idősoros simítás: Savitzky–Golay vagy One Euro Filter úgy, hogy a PLR komponensek megmaradjanak.

Temporal Smoothing
- Kalman filter on pupil/eyelid positions
- 87% fluctuation reduction achieved

Blink Detection
- Eye height thresholding
- Frame-by-frame analysis

Gaze Estimation (opcionális)
- DeepVOG 3D eyeball modell + intrinzik → yaw/pitch vagy 3D vektor.
- PCCR út: több IR LED ismert 3D pozícióban, Purkinje reflexek centroidja + pupilla-közép → kis MLP-vel képernyő-kalibráció (9–25 pont) 1–2° célhibára.

Validáció és QA
- Bland–Altman referencia pupillométerrel.
- Gage R&R (ismételhetőség/reprodukálhatóság).
- Drift- és hőmérséklet teszt.

## References

- **RITnet Paper**: Chaudhary et al., ICCVW 2019
- **GitHub**: https://github.com/AayushKrChaudhary/RITnet
- **OpenCV**: https://opencv.org/

## Status

✅ **GUI**: Production ready  
✅ **Camera Calibration**: Integrated (0.18px error)  
✅ **RITnet**: Integrated and tested  
✅ **EllSeg**: Integrated (robust occlusion handling) ⭐ NEW!  
✅ **3D Iris Model**: 3 implementations with GUI selection  
✅ **Video Generation**: Working  
⏳ **Full Video Processing**: In progress  
⏳ **mm Accuracy Conversion**: Planned  

---

**Last Updated**: 2025-11-01  
**Version**: 1.4 (EllSeg Integration for Occlusion Handling)
