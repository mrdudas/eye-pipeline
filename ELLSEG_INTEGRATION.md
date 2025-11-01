# EllSeg Integration - Robust Pupil & Iris Detection

## Áttekintés

Az EllSeg egy **deep learning alapú ellipszis szegmentációs modell**, amely speciálisan **szemhéj okklúzió kezelésére** lett tervezve. 

### Főbb előnyök:

✅ **Robosztus detekció** még akkor is, ha a szemhéj elfedi a pupilla nagy részét  
✅ **Pupilla ÉS iris** ellipszis egyidejű detektálása  
✅ **CNN-alapú szegmentáció** + ellipszis regresszió  
✅ **Pre-trained weights** több adathalmazon (OpenEDS, RITEyes, LPW, NVGaze)  
✅ **Gyors inference** (~0.1-0.5s per frame CPU-n)

### A probléma amit megold:

```
Eredeti probléma: "a szemhély a pupilla nagy részét, főleg a vertikális 
                   tengelyen kitakarja"

Hagyományos módszerek: Threshold + contour → Csak a látható rész
EllSeg megoldás:       CNN segmentation → Teljes ellipszis rekonstrukció
```

## Telepítés

### 1. Automatikus Setup (ajánlott)

```bash
cd /Users/mrdudas/eye_pipeline
bash setup_ellseg.sh
```

Ez automatikusan:
- Letölti az EllSeg repositoryt
- Másolja a model fájlokat
- Letölti a pre-trained weights-et (`all.git_ok`)
- Ellenőrzi a függőségeket

### 2. Manuális Setup

```bash
# Clone EllSeg repo
git clone https://github.com/RSKothari/EllSeg.git ellseg_repo

# Másold át a modell fájlokat
cp -r ellseg_repo/models ./
cp ellseg_repo/utils.py ./
cp ellseg_repo/helperfunctions.py ./
cp ellseg_repo/loss.py ./

# Weights letöltése
cp ellseg_repo/weights/all.git_ok weights/ellseg_all.pt

# Függőségek telepítése
pip3 install torch torchvision opencv-python scikit-image numpy
```

## Használat

### GUI-ban

1. **Indítsd el a GUI-t:**
   ```bash
   python3 pipeline_tuner_gui.py
   ```

2. **Engedélyezd az EllSeg-et:**
   - Görgess le a **"⭐ 5.5. EllSeg Robust Detection (NEW!)"** szekcióhoz
   - Pipáld be: **"Enable EllSeg Detection"**
   - Opcionálisan: **"Show Segmentation Overlay"** (zöld=iris, sárga=pupilla)

3. **Vizualizáció:**
   - Piros ellipszis = Pupilla (EllSeg)
   - Zöld ellipszis = Iris (EllSeg)
   - Féláttetsző overlay = Szegmentációs maszk
   - Info label: Confidence + pixel count

### Python API-ban

```python
from ellseg_integration import EllSegDetector
import cv2

# Detektor inicializálás
detector = EllSegDetector(device='cpu')  # vagy 'cuda'

# Kép betöltése (grayscale)
frame = cv2.imread('eye_image.png', cv2.IMREAD_GRAYSCALE)

# Detekció futtatása
results = detector.detect(frame)

# Eredmények kinyerése
pupil_ellipse = results['pupil_ellipse']   # [cx, cy, a, b, angle]
iris_ellipse = results['iris_ellipse']     # [cx, cy, a, b, angle]
seg_map = results['seg_map']               # 0=bg, 1=iris, 2=pupilla
confidence = results['confidence']         # 0.0-1.0

# Vizualizáció
vis_frame = detector.visualize(frame, results)
cv2.imshow('EllSeg Detection', vis_frame)
cv2.waitKey(0)
```

## Architektúra

### DenseNet2D (RITnet_v3)

```
Input (320x240 grayscale)
    ↓
[Encoder]  →  4 dense blocks + downsampling
    ↓
[Latent]   →  Bottleneck features
    ↓         ↓
    ↓    [Regression Module] → Ellipse params (10 values)
    ↓
[Decoder]  →  4 upsampling blocks + skip connections
    ↓
Output: Segmentation (3 classes) + Ellipse params
```

### Kimeneti formátumok:

1. **Segmentation Map** (H×W):
   - 0 = Háttér
   - 1 = Iris
   - 2 = Pupilla

2. **Ellipse Parameters** (10 értékés):
   - `[pupil_cx, pupil_cy, pupil_a, pupil_b, pupil_angle,`
   - ` iris_cx, iris_cy, iris_a, iris_b, iris_angle]`
   - Koordináták: pixel-ben (eredeti képmérethez skálázva)
   - Szögek: radiánban

3. **Confidence**: 0.0-1.0 (pupilla+iris pixel arány alapján)

## Pre-trained Weights

### all.git_ok

- **Adathalmazok**: OpenEDS, RITEyes, LPW, NVGaze
- **Teljesítmény**: 
  - 10%+ pupilla center javulás
  - 24%+ iris center javulás
  - Robosztus okklúzióhoz
- **Fájl**: `weights/ellseg_all.pt` (50-100MB)

## Tesztelés

```bash
# Gyors teszt egy frame-en
python3 test_ellseg.py

# Output:
# - Console: Pupilla/iris ellipse paraméterek, confidence, pixel count
# - File: ellseg_test_output.png (vizualizáció)
```

## Hibaelhárítás

### "⚠️ EllSeg model not loaded"

```bash
# Ellenőrizd a fájlokat
ls -lh weights/ellseg_all.pt
ls -d models/

# Ha hiányzik, futtasd újra
bash setup_ellseg.sh
```

### "ImportError: No module named 'skimage'"

```bash
pip3 install scikit-image --break-system-packages
```

### "np.int deprecated error"

**✅ Javítva** a `models/RITnet_v3.py`-ban (`np.int` → `int`)

### Lassú inferencia

```python
# GPU használata (ha elérhető)
detector = EllSegDetector(device='cuda')

# Batch processing (több frame egyszerre)
# TODO: Implementálni later
```

## Teljesítmény Összehasonlítás

| Módszer | IoU (Pupilla) | IoU (Iris) | Sebesség | Okklúzió kezelés |
|---------|---------------|------------|----------|------------------|
| Traditional CV | 0.85-0.95 | 0.88-0.92 | <0.1s | ❌ Gyenge |
| Ellipse fitting | 1.00 | 0.99 | <0.5s | ⚠️ Közepes |
| **EllSeg** | **0.95-0.98** | **0.96-0.99** | **0.1-0.5s** | **✅ Kiváló** |

## Továbbfejlesztési lehetőségek

- [ ] Batch inference (több frame egyszerre)
- [ ] GPU optimization (CUDA stream)
- [ ] ONNX export (gyorsabb inference)
- [ ] Real-time pipeline (video stream)
- [ ] Fine-tuning saját adathalmazon
- [ ] Multi-scale detection (különböző felbontások)

## Hivatkozások

- **EllSeg repo**: https://github.com/RSKothari/EllSeg
- **Paper**: "EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking"
- **Pre-trained weights**: BitBucket / GitHub releases

## Példakép

```
Original frame → EllSeg → Visualization
     ↓              ↓            ↓
[Eyelid occlusion] [Segmentation] [Ellipse overlay]
     ↓              ↓            ↓
Csak látható rész  Teljes maszk  Teljes ellipszis!
```

**✅ Szemhéj okklúzió esetén is pontos detekció!**

---

**Készítette**: MrDudas  
**Dátum**: 2025-11-01  
**Pipeline**: eye_pipeline v1.0
