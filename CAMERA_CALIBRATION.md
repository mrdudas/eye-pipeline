# Camera Calibration - Step 0: Undistortion

## Áttekintés

A pipeline **Step 0**-ban került integrálásra a **kamera geometriai korrekció** (undistortion), amely a kamera lencséjéből származó torzításokat korrigálja.

## Miért fontos?

A kamera lencséje (főleg wide-angle vagy Near-IR kamerák) **geometriai torzításokat** okoznak:
- **Barrel distortion** (hordó-torzítás): A kép széle felé hajlik
- **Pincushion distortion** (párnás torzítás): A kép széle felé befelé görbül
- **Tangential distortion**: Aszimmetrikus torzítás

Ezek a torzítások **hibás pupilla detektálást** okozhatnak:
- ❌ Téves ellipszis fitting
- ❌ Pontatlan pupilla méret
- ❌ Elcsúszott pupilla pozíció
- ❌ mm pontosság lehetetlenné válik

## Kalibráció Folyamata

### 1. Kalibráció Videó Elkészítése

Szükséges:
- **Sakktábla minta** (checkerboard pattern)
- **Ismert méretű négyzetek** (pl. 1mm × 1mm raszter)
- **Többszörös felvétel** különböző szögekben

Példa videó: `eye_cam.mkv`
- 9×6 belső sarkok
- 1mm négyzet méret
- 1805 frame, 60 fps
- 400×400 felbontás

### 2. Automatikus Kalibráció

#### GUI Használat

1. **Indítsd el a GUI-t**:
   ```bash
   python pipeline_tuner_gui.py
   ```

2. **Navigálj a Step 0-hoz**: "Camera Undistortion"

3. **Kattints**: `📹 Run Calibration`

4. **Válaszd ki a kalibrációs videót**: `eye_cam.mkv`

5. **Állítsd be a paramétereket**:
   ```
   Columns (inner corners): 9
   Rows (inner corners): 6
   Square size (mm): 1.0
   Max frames to use: 30
   ```

6. **Run Calibration** → Várd meg a feldolgozást

7. **Eredmény**:
   ```
   ✅ Calibration successful!
   📊 Reprojection error: 0.1756 pixels
   
   Camera Matrix:
     fx = 512.88 px
     fy = 524.10 px
     cx = 264.78 px
     cy = 215.58 px
   
   Distortion Coefficients:
     k1 = 0.053682
     k2 = -0.776959
     p1 = 0.005171
     p2 = 0.018493
     k3 = 1.107195
   
   💾 Calibration saved to: camera_calibration.yaml
   ```

#### Command Line Használat

```bash
python camera_calibration.py \
    --video eye_cam.mkv \
    --chessboard 9x6 \
    --square-size 1.0 \
    --max-frames 30 \
    --output camera_calibration.yaml
```

### 3. Kalibráció Betöltése

A GUI automatikusan betölti a `camera_calibration.yaml` fájlt induláskor.

**Manuális betöltés**:
1. Kattints: `📂 Load Calibration`
2. Válaszd ki a YAML fájlt

## Kamera Mátrix (Intrinsics)

```
K = | fx  0  cx |
    | 0  fy  cy |
    | 0  0   1  |
```

- **fx, fy**: Fókusztávolság (pixel-ben)
- **cx, cy**: Principal point (kép középpontja)

**Eredményünk**:
```
fx = 512.88 px  (horizontal focal length)
fy = 524.10 px  (vertical focal length)
cx = 264.78 px  (center x, ~200 ideal for 400px width)
cy = 215.58 px  (center y, ~200 ideal for 400px height)
```

## Torzítási Együtthatók (Distortion Coefficients)

```
D = [k1, k2, p1, p2, k3]
```

- **k1, k2, k3**: Radiális torzítás (radial distortion)
- **p1, p2**: Tangenciális torzítás (tangential distortion)

**Eredményünk**:
```
k1 =  0.053682  (positive → pincushion)
k2 = -0.776959  (negative → barrel)
p1 =  0.005171  (small tangential)
p2 =  0.018493  (small tangential)
k3 =  1.107195  (correction term)
```

## Undistortion Folyamat

### Matematikai Modell

OpenCV `cv2.undistort()` használja az alábbi transzformációt:

```python
x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y

ahol r² = x² + y²
```

### Pipeline Integráció

```python
# Step 0: Undistortion (FIRST!)
undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

# Step 1-6: További feldolgozás
# ... glint removal, noise reduction, CLAHE, stb.
```

**FONTOS**: Az undistortion **mindig először** fut le, még a glint removal előtt!

## Kalibráció Minőség

### Reprojection Error

A kalibráció pontosságát a **reprojection error** mutatja:

```
Mean reprojection error = 0.1756 pixels
```

**Értékelés**:
- ✅ **< 0.5 px**: Kiváló kalibráció
- ⚠️ **0.5-1.0 px**: Elfogadható
- ❌ **> 1.0 px**: Gyenge, újra kell kalibrálni

A mi eredményünk: **0.1756 px** → **Kiváló!** ✅

### Ellenőrzési Módszerek

1. **Vizuális inspekció**:
   ```python
   python camera_calibration.py --show
   ```
   - Látni kell a detektált sarkokat
   - Zöld overlay a sakktáblán

2. **Before/After összehasonlítás**:
   ```python
   visualizer = CameraCalibrator.load_calibration('camera_calibration.yaml')
   comparison = visualizer.visualize_distortion(frame)
   cv2.imshow('Original vs Undistorted', comparison)
   ```

3. **Grid overlay**:
   - Eredeti képen: görbült vonalak
   - Undistorted képen: egyenes vonalak

## Fájl Struktúra

### camera_calibration.yaml

```yaml
calibration_date: '2025-11-01'
chessboard_size: [9, 6]
square_size_mm: 1.0

camera_matrix:
  rows: 3
  cols: 3
  data: [512.88, 0.0, 264.78, 0.0, 524.10, 215.58, 0.0, 0.0, 1.0]

distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.053682, -0.776959, 0.005171, 0.018493, 1.107195]

intrinsics:
  fx: 512.88
  fy: 524.10
  cx: 264.78
  cy: 215.58

distortion:
  k1: 0.053682
  k2: -0.776959
  p1: 0.005171
  p2: 0.018493
  k3: 1.107195

reprojection_error: 0.1756
```

## Használat a Kódban

### Önálló Használat

```python
from camera_calibration import CameraCalibrator

# Load calibration
camera_matrix, dist_coeffs = CameraCalibrator.load_calibration('camera_calibration.yaml')

# Undistort frame
frame = cv2.imread('eye_frame.png')
undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
```

### GUI Integráció

```python
# Automatically loaded on startup
self.load_camera_calibration('camera_calibration.yaml')

# Applied in preprocessing
def preprocess_frame(self, frame):
    # Step 0: Undistortion
    processed = self.undistort_frame(frame)
    # ... további lépések
```

## Troubleshooting

### Probléma: "No chessboard corners detected"

**Okok**:
- Rossz sakktábla méret (cols × rows)
- Elmosódott vagy sötét kép
- Sakktábla nem látható

**Megoldás**:
1. Használd a debug scriptet:
   ```bash
   python debug_chessboard.py
   ```
2. Különböző méretekkel próbálkozz (9×6, 10×7, 8×5)
3. Javítsd a megvilágítást

### Probléma: Magas reprojection error (>1.0 px)

**Okok**:
- Kevés frame (< 20)
- Rossz sakktábla detektálás
- Moving target

**Megoldás**:
1. Növeld a frame számot: `--max-frames 50`
2. Több különböző szögből készíts felvételt
3. Stabilizáld a sakktáblát

### Probléma: Undistortion nem javít

**Okok**:
- Kalibráció rossz kamerához tartozik
- Rossz méretű képre alkalmazzuk

**Megoldás**:
1. Ellenőrizd a kamera mátrixot (fx, fy ≈ image width)
2. Ugyanazzal a kamerával készítsd a kalibrációt
3. Ugyanazzal a felbontással dolgozz

## Performance

### Sebesség

- **Kalibráció**: ~2-3 másodperc / 30 frame
- **Undistortion**: ~1-2 ms / frame (400×400)
- **Total overhead**: Elhanyagolható (<2%)

### Memory

- **Calibration YAML**: ~2 KB
- **Camera matrix**: 9 × 8 bytes = 72 bytes
- **Dist coeffs**: 5 × 8 bytes = 40 bytes

## Best Practices

### ✅ DO

1. **Új kamerához újra kalibrálj**
2. **Mentsd a YAML-t verziókezelésbe**
3. **Ellenőrizd a reprojection error-t**
4. **Használj legalább 20-30 frame-et**
5. **Undistort-ot mindig először futtasd**

### ❌ DON'T

1. **Ne használj más kamera kalibrációját**
2. **Ne kalibráld különböző felbontással**
3. **Ne hagyd ki az undistort lépést** (ha mm pontosság kell)
4. **Ne commit-old a nagy videó fájlt** (eye_cam.mkv)

## További Funkciók

### 1. Rectification (Opcionális)

Ha sztereó kamerád van, használd a `cv2.stereoCalibrate()` függvényt.

### 2. Fisheye Models

Ha fisheye lencsét használsz:
```python
cv2.fisheye.calibrate()
cv2.fisheye.undistortImage()
```

### 3. Online Calibration

Real-time kalibráció közvetlenül a live stream-ből.

## Referenciák

- **OpenCV Camera Calibration**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- **Zhang's Method**: Z. Zhang, "A Flexible New Technique for Camera Calibration", PAMI 2000
- **Chessboard Pattern Generator**: https://calib.io/pages/camera-calibration-pattern-generator

## Összefoglalás

✅ **Step 0: Camera Undistortion** sikeresen integrálva!

**Mit kaptunk**:
- 📹 Automatikus kamera kalibráció GUI-ból vagy CLI-ból
- 🔧 OpenCV `calibrateCamera` használat 9×6 sakktáblával
- 💾 YAML perzisztencia (fx, fy, cx, cy, k1-k3, p1-p2)
- 🎯 0.1756 px reprojection error (kiváló!)
- ⚡ Real-time undistortion minden frame-en
- 🎛️ GUI toggle be/ki kapcsoláshoz

**Mit nyertünk**:
- ✅ Geometriailag korrekt képek
- ✅ Pontos pupilla ellipszis fitting
- ✅ mm pontosság lehetséges
- ✅ Stabil detektálás a kép szélein is

---

**Date**: 2025-11-01  
**Status**: ✅ Production Ready  
**Reprojection Error**: 0.1756 pixels (Excellent!)
