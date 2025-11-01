# Glint Detektálás és Eltávolítás - Analízis Jelentés

## Összefoglaló

Az `eye1.mp4` videó glint (IR csillanás) kezelésének elemzése és optimalizálása.

## 🔍 Vizsgált Módszerek

### 1. Threshold-alapú Detektálás
- **Tesztelt értékek:** 200, 220, 240, 250
- **Eredmények:**
  - Threshold 200: 10 blob, sok false positive
  - Threshold 220: 6 blob
  - **Threshold 240: 3 blob** ✅ (optimális)
  - Threshold 250: 3 blob, túl agresszív

**Javaslat:** Threshold = 240 optimális a fényes spotok detektálásához.

### 2. Blob Detektálás Módszerek

#### Connected Components (jelenlegi)
✅ **Előnyök:**
- Gyors
- Egyszerű implementáció
- Jól működik threshold után

❌ **Hátrányok:**
- Nem veszi figyelembe a blob formáját
- Zajra érzékeny

#### LoG (Laplacian of Gaussian)
- Tesztelve: 0 blob detektálva az alapértelmezett paraméterekkel
- Finomhangolást igényel
- Jobb lenne körszerű glint-ekhez

**Javaslat:** Maradjon a connected components, de add hozzá szűrési kritériumokat:
- Minimum/maximum terület
- Circularity (körszerűség)
- Aspect ratio

### 3. Maszk Finomítás (Morfológiai Műveletek)

**Optimális beállítások:**
- Kernel: 3x3 ellipszis
- Dilatáció: 1 iteráció

**Miért fontos:**
- A glint élek simítása
- Kis rések bezárása
- Pupilla perem védelme inpainting során

### 4. Glint Eltávolítás Módszerek

#### a) **Telea Inpainting** ✅ (AJÁNLOTT)
- **Előnyök:**
  - Sima átmenetek
  - Jól megőrzi a pupilla perem információit
  - Gyors
- **Használat:** `cv2.inpaint(..., cv2.INPAINT_TELEA)`

#### b) Navier-Stokes Inpainting
- **Előnyök:**
  - Precízebb nagyobb területeken
- **Hátrányok:**
  - Lassabb
  - Túlzott simítás kis területeken

#### c) Median Helyettesítés
- **Előnyök:**
  - Nagyon gyors
- **Hátrányok:**
  - Flat, nem természetes
  - Látható artifact-ok

#### d) Gaussian Blur Helyettesítés
- **Előnyök:**
  - Gyors
- **Hátrányok:**
  - Nem veszi figyelembe a környezetet

## 📊 Statisztikák (eye1.mp4, első képkockák)

### Pixel Intenzitások:
- Min: 0-10
- Max: 255
- Átlag: ~80-100
- **Glint threshold:** 240 (top 5% fényesség)

### Glint Jellemzők:
- Típusos blob méret: 20-40 pixel
- Detektált blobs/frame: 2-4
- Glint arány: ~0.5-2% (frame-függő)

## 🎯 Optimalizált Pipeline Paraméterek

```yaml
glint_removal:
  enabled: true
  
  # Detektálás
  threshold: 240
  
  # Blob szűrés (ÚJ!)
  min_area: 5        # pixel
  max_area: 200      # pixel
  min_circularity: 0.3
  
  # Maszk finomítás
  morph_kernel_size: 3
  morph_iterations: 1
  morph_shape: "ellipse"
  
  # Eltávolítás
  inpainting: true
  inpainting_method: "telea"  # vagy "ns"
  inpainting_radius: 3
```

## 🔬 Továbbfejlesztési Javaslatok

### 1. Adaptív Threshold
Jelenleg fix 240, de frame-enként változhat a megvilágítás:
```python
# Otsu automatic thresholding
threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# majd finomhangolás: threshold = threshold * 0.95
```

### 2. Blob Szűrés Javítása
```python
def filter_glint_blobs(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Szűrési kritériumok
        if 5 < area < 200:  # Területszűrés
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio < 3:  # Nem túl nyújtott
                filtered_mask[labels == i] = 255
    
    return filtered_mask
```

### 3. Temporal Smoothing
Ha egy glint több frame-en keresztül ugyanott van:
- Tracking a blob pozíciók között
- Kalman filter a glint pozíció predikciójához
- Konzisztensebb eltávolítás

### 4. Multi-scale Detection
LoG blob detection finomhangolása:
```python
from skimage.feature import blob_log

blobs = blob_log(
    image, 
    min_sigma=1,     # Kis glint-ek
    max_sigma=10,    # Nagy glint-ek  
    num_sigma=20,    # Finomság
    threshold=0.1    # Érzékenység (csökkentve!)
)
```

## 📈 Következő Lépések

### Azonnal implementálandó:
1. ✅ Blob területszűrés hozzáadása
2. ✅ Circularity szűrés
3. ✅ Konfiguráció update

### Későbbi optimalizálás:
4. ⏳ Adaptív threshold tesztelése
5. ⏳ LoG blob detection finomhangolása
6. ⏳ Temporal consistency

## 🔧 Implementáció

A javított glint removal modul elkészítve a `pupil_pipeline.py` frissítésével.

### Használat:
```python
python pupil_pipeline.py  # Frissített glint removal-lal
```

## 📸 Generált Vizualizációk

Az `output/` mappában:
- `glint_analysis.png` - Threshold értékek összehasonlítása
- `glint_removal_methods.png` - Eltávolítási módszerek
- `blob_log_detection.png` - LoG blob detection
- `glint_multiple_frames.png` - Több frame összehasonlítása
- `glint_optimized_comparison.png` - Optimalizált pipeline eredmények

## ✅ Minőségi Kritériumok

**Sikeres glint eltávolítás:**
- ✅ Fényes spotok teljesen eltávolítva
- ✅ Pupilla perem megőrizve
- ✅ Sima, természetes átmenetek
- ✅ Nincs látható artifact
- ✅ Gyors (<10ms/frame)

---

**Készítette:** mrdudas  
**Dátum:** 2025. október 31.  
**Verzió:** 0.2 - Glint Optimization
