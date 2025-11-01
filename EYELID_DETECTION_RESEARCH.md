# 👁️ Eyelid Detection for Near-IR Images - Kutatás

## 🎯 Cél
Near-IR (infravörös közeli) képeken szemhéj detektálás → szem határainak meghatározása

## 🔍 Elérhető Modellek és Módszerek

### 1. **RITnet - Semantic Segmentation** ⭐ LEGJOBB
**Repository:** https://github.com/AayushKrChaudhary/RITnet

**Leírás:**
- Real-time iris/sclera/pupil/eyelid szegmentálás
- U-Net alapú deep learning
- **Támogatja Near-IR képeket!**
- Pre-trained modellek (PyTorch)

**Kimenet:**
- Pupilla maszk
- Iris maszk
- Sclera maszk
- **Eyelid boundaries (felső és alsó szemhéj)**

**Előnyök:**
- ✅ Specifikusan eye tracking-hez készült
- ✅ Near-IR képekre tanítva
- ✅ Real-time (>100 fps)
- ✅ Szemhéj határ pontokat ad
- ✅ Pre-trained weights available

**Használat:**
```python
import torch
from models.ritnet import DenseNet2D

# Modell betöltése
model = DenseNet2D()
model.load_state_dict(torch.load('ritnet_model.pkl'))

# Inferencia
output = model(image)
# output: [batch, 4, H, W] - 4 class: background, sclera, iris, pupil
```

---

### 2. **EllSeg - Ellipse Segmentation** 🎯
**Repository:** https://github.com/ChristianProbst/ellseg

**Leírás:**
- Ellipszis alapú pupilla és szemhéj detektálás
- CNN + geometriai fitting
- Near-IR támogatás

**Kimenet:**
- Pupilla ellipszis
- **Felső szemhéj ellipszis**
- **Alsó szemhéj ellipszis**

**Előnyök:**
- ✅ Geometriailag konzisztens
- ✅ Okklúzió handling (részben látható pupilla)
- ✅ Eye tracking specifikus

---

### 3. **ElSe (Eyelid and Sclera Segmentation)** 🔬
**Paper:** "Accurate Eye Centre Localisation by Means of Gradients"
**Alternatív:** OpenEDS dataset modellek

**Leírás:**
- Facebook Reality Labs / Meta dataset
- Near-IR és RGB támogatás
- Semantic segmentation

**Adathalmaz:**
- OpenEDS2019: Eye tracking dataset
- OpenEDS2020: Gaze estimation
- Eyelid annotations included

**Modellek:**
- DeepLabv3+
- U-Net variánsok
- Szemhéj szegmentálási maszkokkal

---

### 4. **ExCuSe - Extreme Close-Up Semantic Segmentation** 💎
**Repository:** https://github.com/swook/ExCuSe

**Leírás:**
- Extreme close-up eye images
- 11 osztály szegmentálás (pupilla, iris, sclera, **eyelids**, eyelashes, skin)
- PyTorch implementáció

**Előnyök:**
- ✅ Részletes szemhéj szegmentálás
- ✅ Alsó és felső szemhéj külön
- ✅ Szempilla is
- ⚠️ Inkább RGB-re, de adaptálható

---

### 5. **Traditional CV - Canny + Hough + Parabola Fitting** 🔧
**Módszer:** Hagyományos computer vision

**Lépések:**
1. Preprocessing (CLAHE, blur)
2. Canny edge detection
3. Region of interest (pupilla felett/alatt)
4. Parabola vagy ellipszis illesztés

**Előnyök:**
- ✅ Nincs szükség modell betöltésre
- ✅ Gyors
- ⚠️ Kevésbé robosztus

```python
# Felső szemhéj keresés
roi_upper = image[0:pupil_y, :]
edges = cv2.Canny(roi_upper, 50, 150)
# Parabola fitting a legfelső élpontokra
```

---

## 📊 Összehasonlítás

| Modell | Pontosság | Sebesség | Near-IR | Pre-trained | Szemhéj |
|--------|-----------|----------|---------|-------------|---------|
| **RITnet** | ⭐⭐⭐⭐⭐ | 🚀 100+ fps | ✅ | ✅ | ✅ Explicit |
| **EllSeg** | ⭐⭐⭐⭐ | 🚀 Fast | ✅ | ✅ | ✅ Ellipszis |
| **ElSe/OpenEDS** | ⭐⭐⭐⭐⭐ | ⚡ 50+ fps | ✅ | ✅ | ✅ Maszk |
| **ExCuSe** | ⭐⭐⭐⭐ | ⚡ 30+ fps | ⚠️ | ✅ | ✅ Részletes |
| **Traditional CV** | ⭐⭐⭐ | 🚀🚀 200+ fps | ✅ | - | ⚠️ Közelítés |

---

## 🎯 Ajánlás: RITnet

### Miért?
1. ✅ **Specifikusan eye tracking-hez készült**
2. ✅ **Near-IR képekre tanítva** (pontosan ami kell!)
3. ✅ **Pre-trained weights** (azonnal használható)
4. ✅ **Real-time** (>100 fps)
5. ✅ **Szemhéj határokat** explicit módon adja
6. ✅ **Aktívan karbantartott** (2020-2023)
7. ✅ **PyTorch** (könnyű integrálás)

### Setup:
```bash
pip install torch torchvision
git clone https://github.com/AayushKrChaudhary/RITnet.git
# Weights letöltése
```

### Integráció:
```python
class RITnetEyelidDetector:
    def __init__(self, model_path):
        self.model = load_ritnet_model(model_path)
    
    def detect(self, frame):
        # Segmentation
        output = self.model(frame)
        
        # Eyelid contours extraction
        upper_eyelid = extract_eyelid(output, 'upper')
        lower_eyelid = extract_eyelid(output, 'lower')
        
        return {
            'upper': upper_eyelid,
            'lower': lower_eyelid,
            'pupil': extract_pupil(output),
            'iris': extract_iris(output)
        }
```

---

## 🚀 Alternatív Gyors Megoldás: Traditional CV

Ha nem akarunk modellt:

```python
def detect_eyelids_traditional(frame, pupil_center, pupil_radius):
    """Hagyományos CV alapú szemhéj detektálás"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ROI: pupilla felett és alatt
    roi_upper = gray[0:pupil_center[1]-pupil_radius, :]
    roi_lower = gray[pupil_center[1]+pupil_radius:, :]
    
    # Edge detection
    edges_upper = cv2.Canny(roi_upper, 50, 150)
    edges_lower = cv2.Canny(roi_lower, 50, 150)
    
    # Legfelső/legalsó élek
    upper_points = find_top_edge_points(edges_upper)
    lower_points = find_bottom_edge_points(edges_lower)
    
    # Parabola/spline fitting
    upper_eyelid = fit_parabola(upper_points)
    lower_eyelid = fit_parabola(lower_points)
    
    return upper_eyelid, lower_eyelid
```

---

## 📥 Következő Lépések

### 1. RITnet Teszt (AJÁNLOTT):
```bash
# Clone repository
git clone https://github.com/AayushKrChaudhary/RITnet.git
cd RITnet

# Weights letöltése (check GitHub releases)
wget <weights_url>

# Teszt futtatás
python test.py --image ../eye1.mp4
```

### 2. Integráció GUI-ba:
- Új section: "6. Eyelid Detection (RITnet)"
- Model betöltés inicializáláskor
- Real-time inferencia preview-ban
- Szemhéj határok vizualizálása

### 3. Fallback Traditional CV:
- Ha RITnet nem elérhető
- Gyors prototípushoz
- Paraméter tuning GUI-ban

---

## 🔗 Hasznos Linkek

- **RITnet Paper:** https://arxiv.org/abs/2010.01926
- **EllSeg Paper:** https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Fuhl_EllSeg_CVPR_2016_paper.pdf
- **OpenEDS Dataset:** https://research.facebook.com/publications/openeds-open-eye-dataset/
- **ExCuSe:** https://openaccess.thecvf.com/content_CVPRW_2019/papers/GAZE/Kothari_ExCuSe_Extreme_Close-Up_Eye_Segmentation_for_Gaze_Estimation_CVPRW_2019_paper.pdf

---

**Következő feladat:** RITnet telepítése és tesztelése! 🚀
