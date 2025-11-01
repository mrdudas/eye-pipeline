# AI-alapú Pupilla Detektálás - Kutatás és Implementációs Terv

## 🎯 Probléma

A jelenlegi hagyományos CV módszer (Canny edge + ellipszis illesztés) problémái:
- ❌ Nem robusztus különböző megvilágításhoz
- ❌ Sok false positive edge
- ❌ Nem kezeli jól az occlusionokat (szemhéj, szempilla)
- ❌ Paraméter-függő
- ❌ Nem tanul a data-ból

## 🚀 Modern AI Megoldások

### 1. **DeepVOG** ⭐ (AJÁNLOTT kezdéshez)

**Mi ez?**
- Deep Learning alapú pupilla és iris szegmentálás
- 2019, kifejezetten eye tracking-hez fejlesztve
- U-Net alapú architektúra

**Előnyök:**
- ✅ Kifejezetten eye tracking-hez készült
- ✅ Pre-trained modellek elérhetők
- ✅ Gyors inferencia (~100-200 fps)
- ✅ Python/PyTorch implementáció
- ✅ 3D eyeball model support (gaze estimation-höz)

**Hátrányok:**
- ⚠️ Lehet nem ideális IR (infravörös) képekhez
- ⚠️ Fine-tuning szükséges lehet az eye1.mp4-hez

**Repository:**
```
https://github.com/pydsgz/DeepVOG
```

**Használat:**
```python
from deepvog import DeepVOG
model = DeepVOG()
pupil_ellipse = model.fit_pupil(frame)
```

---

### 2. **EllSeg** (U-Net + Ellipse Fitting)

**Mi ez?**
- CNN szegmentálás + geometriai ellipszis illesztés
- Kifejezetten pupilla/iris szegmentáláshoz
- Robusztus occlusionokhoz

**Előnyök:**
- ✅ Szegmentálás maszk → tiszta pupilla régió
- ✅ Jól működik szemhéj/szempilla mellett
- ✅ Pre-trained modellek

**Repository:**
```
https://github.com/ChristianProbst/ellseg
```

---

### 3. **PupilNet** (Lightweight CNN)

**Mi ez?**
- Könnyű CNN architektúra
- Real-time mobil eszközökön is
- Direct ellipse parameter regression

**Előnyök:**
- ✅ Nagyon gyors (real-time)
- ✅ Kis modell méret
- ✅ Direct regression → nincs post-processing

---

### 4. **RITnet** (Real-time Iris Segmentation)

**Mi ez?**
- Dense U-Net architektúra
- Multi-class szegmentálás: pupilla, iris, sclera
- State-of-the-art pontosság

**Előnyök:**
- ✅ Legpontosabb szegmentálás
- ✅ Multi-class → több info
- ✅ Robusztus

**Hátrányok:**
- ⚠️ Lassabb mint a többiek
- ⚠️ Nagyobb modell

**Repository:**
```
https://github.com/AayushKrChaudhary/RITnet
```

---

### 5. **Saját U-Net Fine-tuning**

**Stratégia:**
- Pre-trained U-Net (ImageNet/Medical imaging)
- Fine-tune az eye1.mp4 annotált mintáin
- Binary szegmentálás: pupilla vs háttér

**Előnyök:**
- ✅ Teljesen customizálható
- ✅ Optimális az eye1.mp4-hez
- ✅ Transfer learning → kevesebb adat kell

**Hátrányok:**
- ⚠️ Annotáció szükséges (~100-500 kép)
- ⚠️ Training idő
- ⚠️ GPU szükséges

---

## 📊 Összehasonlítás

| Módszer | Pontosság | Sebesség | Setup | Pre-trained | Ajánlás |
|---------|-----------|----------|-------|-------------|---------|
| **DeepVOG** | ⭐⭐⭐⭐ | 🚀🚀🚀 (fast) | ✅ Könnyű | ✅ Igen | 🥇 **LEGJOBB kezdéshez** |
| **EllSeg** | ⭐⭐⭐⭐⭐ | 🚀🚀 (medium) | ✅ Könnyű | ✅ Igen | 🥈 Második választás |
| **RITnet** | ⭐⭐⭐⭐⭐ | 🚀 (slower) | ⚠️ Közepes | ✅ Igen | 🥉 Ha pontosság kritikus |
| **PupilNet** | ⭐⭐⭐ | 🚀🚀🚀🚀 (fastest) | ✅ Könnyű | ✅ Igen | ⚡ Real-time-hoz |
| **Saját U-Net** | ⭐⭐⭐⭐⭐ | 🚀🚀 (medium) | ❌ Nehéz | ❌ Annotálás kell | 🎓 Research projekthez |

---

## 🎯 AJÁNLÁS: DeepVOG

### Miért?
1. **Eye tracking-specifikus** - pontosan erre tervezték
2. **Pre-trained** - nincs szükség annotálásra
3. **Gyors** - 100+ fps
4. **PyTorch** - könnyű integrálni
5. **3D eyeball model** - később gaze estimation-höz is használható

### Implementációs Terv

#### 1. fázis: DeepVOG Setup (30 perc)
```bash
pip install torch torchvision
git clone https://github.com/pydsgz/DeepVOG
cd DeepVOG
pip install -e .
```

#### 2. fázis: Integráció (1 óra)
```python
# pupil_pipeline.py módosítása
from deepvog import DeepVOG

class EyeTrackingPipeline:
    def __init__(self, ...):
        # DeepVOG model betöltése
        self.deepvog = DeepVOG()
        self.deepvog.load_model('pretrained_weights.pth')
    
    def detect_pupil_ai(self, frame):
        # AI-alapú detektálás
        result = self.deepvog.process_frame(frame)
        pupil_ellipse = result['pupil_ellipse']
        confidence = result['confidence']
        
        return pupil_ellipse, confidence
```

#### 3. fázis: Tesztelés (30 perc)
- Első 100 frame tesztelése
- Összehasonlítás hagyományos módszerrel
- Pontosság és sebesség mérése

#### 4. fázis: Teljes videó (1 óra)
- Teljes eye1.mp4 feldolgozása
- Eredmények mentése
- Vizualizáció

---

## 🔧 Alternatív Gyors Megoldás: MediaPipe Iris

**Google MediaPipe:**
- Production-ready
- Real-time
- Pre-trained iris landmark detection

```bash
pip install mediapipe
```

```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Iris landmarks!
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

results = face_mesh.process(frame)
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0]
    # Iris landmarks: 468-477
    iris_landmarks = landmarks.landmark[468:477]
```

**Előnyök:**
- ✅ Google production-ready
- ✅ Nagyon gyors
- ✅ Egy pip install
- ✅ Iris + pupilla landmarks

**Hátrányok:**
- ⚠️ Teljes arc kell (lehet probléma close-up eye videónál)
- ⚠️ Nem specifikusan pupillometria-hoz

---

## 📋 Action Plan

### Option A: DeepVOG (Ajánlott) 🥇

**Időigény:** 2-3 óra
**Előkészület:**
1. DeepVOG telepítés
2. Pre-trained weights letöltése
3. Integráció a pipeline-ba
4. Tesztelés

**Előny:** Kifejezetten eye tracking-hez, pre-trained, gyors

---

### Option B: MediaPipe Iris (Leggyorsabb) ⚡

**Időigény:** 30 perc - 1 óra
**Előkészület:**
1. `pip install mediapipe`
2. Egyszerű integráció
3. Tesztelés

**Előny:** Production-ready, egy parancs setup, nagyon gyors

**Kockázat:** Lehet nem működik close-up eye videóhoz

---

### Option C: RITnet (Legpontosabb) 🎯

**Időigény:** 3-4 óra
**Előkészület:**
1. RITnet repo clone
2. Dependencies telepítés
3. Pre-trained model letöltés
4. Integráció
5. Tesztelés

**Előny:** State-of-the-art pontosság, multi-class szegmentálás

---

### Option D: Saját U-Net (Research) 🎓

**Időigény:** 2-3 nap
**Előkészület:**
1. 100-500 kép manuális annotálása
2. U-Net architektúra
3. Training (GPU!)
4. Evaluáció
5. Fine-tuning

**Előny:** Teljesen customizált, optimális az adathoz

**Csak akkor, ha:** Research projekt, van idő és GPU

---

## 🚦 Mi Legyen a Következő Lépés?

### Ajánlásom: **Próbáljuk ki a MediaPipe-ot ELŐSZÖR**

**Miért?**
1. **5 perc setup** - egy pip install
2. **Production-ready** - Google által támogatott
3. **Gyors teszt** - azonnal látjuk működik-e close-up eye videóhoz
4. **Ha működik** → kész, ha nem → DeepVOG

### Majd ha MediaPipe nem ideális:
→ **DeepVOG** (eye tracking specifikus, pre-trained)

### Ha extrém pontosság kell:
→ **RITnet** (state-of-the-art)

---

## 💡 Döntés Fa

```
eye1.mp4 pupilla detektálás
    │
    ├─ Gyors prototípus? (5 perc)
    │   └─→ MediaPipe Iris
    │       ├─ Működik? ✅ → KÉSZ
    │       └─ Nem működik? ❌ → DeepVOG
    │
    ├─ Eye tracking specifikus? (2 óra)
    │   └─→ DeepVOG
    │
    ├─ Maximum pontosság? (3 óra)
    │   └─→ RITnet
    │
    └─ Research projekt? (2-3 nap)
        └─→ Saját U-Net + Annotálás
```

---

## 🎬 Kezdjük?

Szerinted melyik opciót próbáljuk?

**Gyors javaslat:**
1. **5 perc:** MediaPipe próba
2. **Ha nem jó:** DeepVOG telepítés
3. **Tesztelés:** 100 frame összehasonlítás
4. **Döntés:** Melyik módszer megy tovább

Mit szólsz? 🚀
