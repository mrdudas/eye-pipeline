# EllSeg Optimization Analysis

## Current Performance (Baseline)
- **Average Detection Time**: 104.35ms (~9.6 FPS)
- **Model Forward Pass**: 86.41ms (82.8% of total time)
- **Preprocessing**: 2.58ms (2.5%)
- **Postprocessing**: 5.65ms (5.4%)
- **Overhead**: 9.71ms (9.3%)

## Timing Breakdown
```
┌────────────────────────────────────────┐
│ Total: 104.35ms                        │
├────────────────────────────────────────┤
│ Model Forward: 86.41ms (82.8%) ████████│ ← BOTTLENECK
│ Overhead: 9.71ms (9.3%) █              │
│ Postprocessing: 5.65ms (5.4%) █        │
│ Preprocessing: 2.58ms (2.5%)           │
└────────────────────────────────────────┘
```

## Optimization Strategies

### 1. ⚡ **Model Optimization** (Highest Impact)
**Target**: Reduce 86.41ms model forward time

#### A. Reduce Input Resolution ❌ **NOT POSSIBLE**
- **Problem**: Model architecture has **fixed input size** (320x240)
- The `elReg` regression head uses Linear layers that expect fixed dimensions
- After conv layers: 320x240 → 3x5 feature map → flatten to 480 features
- `self.l1 = nn.Linear(480, 256)` expects exactly 480 input features
- **Conclusion**: Changing input size would require retraining the model

**Why it failed**:
```python
# Error with 256x192 input:
# RuntimeError: linear(): input and weight.T shapes cannot be multiplied (1x96 and 480x256)
# The flattened feature map is 96 (3x32) but l1 expects 480
```

**Alternative**: Would need to retrain model with different input size, not feasible for quick optimization

#### B. Model Quantization (Medium effort)
- Convert model from FP32 → FP16 (half precision)
- Expected speedup: **1.5-2x** → **~40-50ms savings**
- MPS (Apple Silicon) supports FP16 well

**Implementation**:
```python
# After loading model
self.model = self.model.half()  # Convert to FP16
input_tensor = input_tensor.half()  # Also convert inputs
```

#### C. Model Pruning/Distillation (High effort, requires retraining)
- Remove redundant network layers
- Create lighter "student" model
- Expected speedup: **2-3x** → **~60-80ms savings**
- Requires: EllSeg training code, GPU, dataset

**Not recommended** unless you need real-time performance (30+ FPS)

---

### 2. 🔧 **Postprocessing Optimization** (Medium Impact)
**Target**: Reduce 5.65ms postprocessing time

#### A. Skip Segmentation Map Rescaling
- Only rescale if needed for visualization
- **Savings**: ~2-3ms

```python
# In detect() method, only rescale seg_map if show_segmentation=True
if self.show_segmentation:
    seg_map = self._rescale_segmap(seg_map, transform_info)
else:
    seg_map = None  # Don't rescale
```

#### B. Use Network Ellipse Regression Directly
- Current: Fit ellipse to segmentation mask (cv2.fitEllipse)
- Alternative: Use direct ellipse regression output from model
- **Savings**: ~3-4ms (skip contour finding + ellipse fitting)

**Implementation**:
```python
def _extract_ellipses_from_regression(self, elOut, transform_info):
    """Extract ellipses directly from network regression output"""
    # elOut format: [batch, 10] (5 params for pupil + 5 for iris)
    pupil_params = elOut[0, :5].cpu().numpy()
    iris_params = elOut[0, 5:].cpu().numpy()
    
    # Rescale from normalized coords [0,1] to image coords
    h, w = transform_info['orig_shape']
    pupil_ellipse = self._denormalize_ellipse(pupil_params, w, h)
    iris_ellipse = self._denormalize_ellipse(iris_params, w, h)
    
    return pupil_ellipse, iris_ellipse
```

---

### 3. 📉 **Reduce Overhead** (Small Impact)
**Target**: Reduce 9.71ms overhead

#### A. Batch Processing
- Process multiple frames in single forward pass
- Only helps for offline video processing (not real-time)
- **Savings**: ~5ms per frame (when batching 4-8 frames)

```python
def detect_batch(self, frames):
    """Detect on batch of frames"""
    batch_tensors = []
    for frame in frames:
        tensor, _ = self.preprocess_frame(frame)
        batch_tensors.append(tensor)
    
    batch = torch.cat(batch_tensors, dim=0)  # [N, 1, H, W]
    
    with torch.no_grad():
        x4, x3, x2, x1, x = self.model.enc(batch)
        # ... process batch
```

#### B. Optimize PyTorch Operations
- Use `torch.inference_mode()` instead of `torch.no_grad()`
- Pre-allocate output tensors
- **Savings**: ~2-3ms

```python
with torch.inference_mode():  # Slightly faster than no_grad()
    x4, x3, x2, x1, x = self.model.enc(input_tensor)
```

---

## Recommended Quick Wins

### 🥇 **Priority 1: FP16/Mixed Precision** (15 min effort, 40-50ms speedup)
```python
# In EllSegDetector.__init__(), after loading model:
if self.device.type in ['cuda', 'mps']:
    self.model = self.model.half()  # Convert to FP16
```
**Expected Result**: 104ms → **50-60ms** (~17-20 FPS)
**Note**: Test on MPS backend for numerical stability

### 🥈 **Priority 2: Skip Unnecessary Rescaling** (10 min effort, 2-3ms speedup)
```python
# Only rescale segmentation map if needed for visualization
seg_map_rescaled = self._rescale_segmap(seg_map, transform_info) if return_viz else None
```
**Expected Result**: 70ms → **67ms** (~15 FPS)

### 🥉 **Priority 3: Use Direct Regression** (30 min effort, 3-4ms speedup)
```python
# Use network's ellipse regression instead of cv2.fitEllipse
pupil_ellipse, iris_ellipse = self._extract_ellipses_from_regression(elOut, transform_info)
```
**Expected Result**: 67ms → **63ms** (~16 FPS)

---

## Performance Targets

| Optimization Level | Time per Frame | FPS | Speedup |
|-------------------|----------------|-----|---------|
| **Current** | 104ms | 9.6 | 1.0x |
| + Skip Rescaling | 101ms | 9.9 | 1.0x |
| + Direct Regression | 97ms | 10.3 | 1.1x |
| + FP16 Quantization | 50-60ms | 16-20 | 1.7-2.1x |
| + torch.compile (Py3.11+) | 40-50ms | 20-25 | 2.1-2.6x |
| + Batch Processing | 35-40ms | 25-28 | 2.6-3.0x |

---

## Trade-offs

### Resolution Reduction
- ✅ Easy to implement (1 line change)
- ✅ No additional dependencies
- ✅ Still accurate for ellipse detection
- ⚠️ Slightly less precise for small pupils
- ⚠️ Test on your data first

### FP16 Quantization
- ✅ Good speedup (1.5-2x)
- ✅ MPS supports FP16 natively
- ⚠️ Minimal accuracy loss (~0.1% typically)
- ⚠️ Requires testing for numerical stability

### Batch Processing
- ✅ Best for offline video processing
- ❌ Not useful for real-time (adds latency)
- ❌ Increases memory usage

---

## Next Steps

1. **Test reduced resolution** (224x168) on sample video
2. **Measure detection accuracy** (compare ellipse params)
3. If acceptable, **implement Priority 1-3** optimizations
4. If need more speed, **implement FP16 quantization**
5. **Re-benchmark** and compare

---

## Expected Final Performance

**Optimistic**: 60-70ms per frame (~15 FPS) with simple changes
**With FP16**: 40-50ms per frame (~20-25 FPS)
**With all optimizations**: 30-40ms per frame (~25-30 FPS)

**Current pipeline bottleneck**: Detection is 92.8% of total time
After optimization, detection would be ~50-60% of total time.
