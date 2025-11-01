# Eye Pipeline - Setup Guide

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd eye_pipeline
```

### 2. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python packages
pip install opencv-python numpy matplotlib scikit-image scipy pyyaml tqdm Pillow
pip install torch torchvision torchaudio
```

### 3. Download RITnet Model

```bash
# Clone RITnet repository (includes pre-trained model)
git clone https://github.com/AayushKrChaudhary/RITnet.git
```

### 4. Prepare Your Video

Place your Near-IR eye tracking video in the project directory:
```bash
# Your video file should be named: eye1.mp4
# Or modify the path in pipeline_tuner_gui.py
```

### 5. Run the GUI

```bash
python pipeline_tuner_gui.py
```

You should see:
```
Loading RITnet model...
RITnet loaded successfully on cpu
```

## Project Structure

```
eye_pipeline/
├── pipeline_tuner_gui.py       # Main GUI application
├── test_ritnet.py              # Standalone RITnet test
├── pupil_pipeline.py           # Traditional CV pipeline
├── apply_temporal_smoothing.py # Kalman/Savitzky-Golay filters
├── readme.md                   # Project overview
├── .gitignore                  # Git ignore rules
│
├── Documentation/
│   ├── RITNET_INTEGRATION.md
│   ├── RITNET_INTEGRATION_SUCCESS.md
│   ├── EYELID_DETECTION_RESEARCH.md
│   ├── VIDEO_GENERATION_GUIDE.md
│   ├── GUI_USAGE_GUIDE.md
│   └── EYE_CORNERS_DETECTION.md
│
├── RITnet/                     # Cloned from external repo
│   ├── best_model.pkl          # Pre-trained weights
│   ├── models.py
│   ├── densenet.py
│   └── ...
│
└── output/                     # Generated test videos
```

## System Requirements

### Minimum
- **OS**: Windows 10+, macOS 10.14+, Linux
- **Python**: 3.8+
- **RAM**: 4GB
- **CPU**: Intel i5 or equivalent

### Recommended
- **Python**: 3.10+
- **RAM**: 8GB+
- **GPU**: CUDA-capable (for faster inference)
- **CPU**: Intel i7/Apple M1 or better

## Dependencies

### Core
```
opencv-python >= 4.8.0
numpy >= 1.24.0
matplotlib >= 3.7.0
scikit-image >= 0.21.0
scipy >= 1.11.0
PyYAML >= 6.0
tqdm >= 4.66.0
Pillow >= 10.0.0
```

### Deep Learning
```
torch >= 2.0.0
torchvision >= 0.15.0
```

### GUI
```
tkinter (usually included with Python)
```

## Troubleshooting

### RITnet Model Not Found

**Error**: `Could not load RITnet model`

**Solution**:
1. Check `./RITnet/best_model.pkl` exists
2. Run: `git clone https://github.com/AayushKrChaudhary/RITnet.git`

### PyTorch Installation Issues

**Error**: `No module named 'torch'`

**Solution**:
```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Video File Not Found

**Error**: `Nem sikerült megnyitni: eye1.mp4`

**Solution**:
1. Place your video in the project root
2. Or modify line in `pipeline_tuner_gui.py`:
   ```python
   def __init__(self, video_path="your_video.mp4", ...):
   ```

### GUI Not Starting (macOS)

**Error**: `ApplePersistenceIgnoreState` warning

**Solution**: This is normal on macOS, GUI should still work.

### Slow Performance

**Problem**: Inference takes >100ms per frame

**Solution**:
1. **Use GPU**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Reduce resolution** (modify preprocessing):
   ```python
   resized = cv2.resize(gray, (320, 200))  # Instead of 640x400
   ```

3. **Disable segmentation overlay**:
   Uncheck "Show Segmentation Overlay" in GUI

## Usage Examples

### 1. Interactive Parameter Tuning

```bash
python pipeline_tuner_gui.py
```

- Use sliders to adjust preprocessing parameters
- Navigate frames with slider
- Real-time preview updates
- Save/load settings to YAML

### 2. Standalone RITnet Test

```bash
python test_ritnet.py
```

Outputs:
- `ritnet_test_result.png` (6-panel visualization)
- `ritnet_frame_result.png` (single annotated frame)

### 3. Video Generation

In GUI:
1. Click "🧪 Test on 50 Frames"
2. Wait for processing
3. Output: `output/test_frames_0_to_50.mp4`

### 4. Traditional CV Pipeline

```bash
python pupil_pipeline.py
```

### 5. Temporal Smoothing

```bash
python apply_temporal_smoothing.py
```

## Configuration

### Settings File

Settings are saved to `pipeline_settings.yaml`:

```yaml
glint:
  enabled: true
  threshold: 240
  min_area: 10
  max_area: 100
  morph_iterations: 2

noise:
  enabled: true
  method: bilateral
  strength: 3

clahe:
  enabled: true
  clip_limit: 2.0
  tile_size: 8

pupil:
  threshold: 50
  min_area: 100
  morph_kernel: 5

eyelid:
  enabled: true
  show_segmentation: true
  show_boundaries: true
  show_vertical_axis: true
```

### Modify Video Path

Edit `pipeline_tuner_gui.py`:

```python
if __name__ == "__main__":
    gui = PipelineTunerGUI(
        video_path="path/to/your/video.mp4",  # Change this
        config_path="config.yaml"
    )
    gui.run()
```

## GPU Support

### Check GPU Availability

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Force GPU Usage

Edit `pipeline_tuner_gui.py`:

```python
# In __init__ method
self.device = torch.device('cuda')  # Force GPU
```

## Performance Optimization

### Batch Processing

For processing entire video:

```python
# Collect 10 frames
batch_frames = [frames[i:i+10] for i in range(0, len(frames), 10)]

# Process batch
for batch in batch_frames:
    tensors = torch.stack([preprocess(f) for f in batch])
    with torch.no_grad():
        outputs = model(tensors)
```

### Multi-threading

Already implemented in GUI for video generation (`_run_test_thread`).

### Model Optimization

```python
# TorchScript compilation
model_scripted = torch.jit.script(model)
model_scripted.save('ritnet_scripted.pt')

# ONNX export
torch.onnx.export(model, dummy_input, "ritnet.onnx")
```

## Development

### Adding New Features

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push: `git push origin feature/new-feature`
5. Create Pull Request

### Testing

```bash
# Test RITnet integration
python test_ritnet.py

# Test GUI (visual inspection)
python pipeline_tuner_gui.py

# Test traditional pipeline
python pupil_pipeline.py
```

## Documentation

- **[README.md](readme.md)** - Project overview
- **[RITNET_INTEGRATION.md](RITNET_INTEGRATION.md)** - RITnet setup details
- **[RITNET_INTEGRATION_SUCCESS.md](RITNET_INTEGRATION_SUCCESS.md)** - Implementation summary
- **[EYELID_DETECTION_RESEARCH.md](EYELID_DETECTION_RESEARCH.md)** - Model research
- **[VIDEO_GENERATION_GUIDE.md](VIDEO_GENERATION_GUIDE.md)** - Video testing
- **[GUI_USAGE_GUIDE.md](GUI_USAGE_GUIDE.md)** - GUI instructions

## Support

### Issues

Report bugs or feature requests on GitHub Issues.

### Questions

Check existing documentation first:
1. README.md
2. RITNET_INTEGRATION.md
3. Troubleshooting section above

## License

This project uses RITnet model which is subject to its own license.
Check: https://github.com/AayushKrChaudhary/RITnet

## Credits

- **RITnet**: Aayush K Chaudhary et al. (ICCVW 2019)
- **OpenCV**: https://opencv.org/
- **PyTorch**: https://pytorch.org/

## Changelog

### v1.0 (2025-11-01)
- ✅ Initial release
- ✅ RITnet integration
- ✅ Interactive GUI
- ✅ Video generation
- ✅ Settings save/load
- ✅ Thread-safe processing

---

**Last Updated**: 2025-11-01
