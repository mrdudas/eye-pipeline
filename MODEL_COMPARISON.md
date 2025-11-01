# Iris Model Comparison

Ez a dokumentum összehasonlítja a három különböző iris/pupilla modell implementációt.

## Modellek áttekintése

### 1. Ellipse-based Model (AJÁNLOTT) ⭐

**Fájl:** `ellipse_iris_model.py`
**Osztály:** `EllipseIrisPupilModel`

**Leírás:**
- Közvetlen ellipszis illesztés OpenCV `fitEllipse` használatával
- Két koncentrikus ellipszis: pupilla és iris
- Ugyanaz a középpont és elforgatási szög mindkettőnél
- Tekintet becslés az ellipszis arányból (minor/major ≈ cos(viewing_angle))

**Előnyök:**
- ✅ **Kiváló pontosság**: IoU 1.000 (pupilla), 0.988 (iris)
- ✅ **Nagyon gyors**: < 0.5 másodperc/frame
- ✅ **Egyszerű**: 434 sor kód, nincs komplex 3D matematika
- ✅ **Fizikailag helyes**: 3D körök → 2D ellipszisek vetítése
- ✅ **Robusztus**: OpenCV-vel tesztelt algoritmus

**Hátrányok:**
- ❌ Nincs kamera kalibráció támogatás (nincs szükség rá)
- ❌ Nincs explicit 3D térbeli pozíció becslés

**Használat:**
```python
model = EllipseIrisPupilModel(width, height)
params = model.fit_to_mask(ritnet_mask)
unwrapped = model.unwrap_iris(frame, params)
```

**Teljesítmény szintetikus adaton:**
```
IoU: Pupil = 1.000, Iris = 0.988
Speed: < 0.5 sec
Gaze estimation: θ=0.0° φ=-0.0°
```

---

### 2. Original Model (Simple 3D)

**Fájl:** `iris_model_3d.py`
**Osztály:** `IrisPupilModel3D`

**Leírás:**
- Egyszerűsített 3D vetítés
- Körök rotációja és transzlációja
- Perspektív vetítés egyszerűsített képletekkel
- Optimalizáció-alapú illesztés (Nelder-Mead vagy Differential Evolution)

**Előnyök:**
- ✅ Jó pontosság: IoU ~0.97
- ✅ Támogatja kamera kalibrációt
- ✅ 3D térbeli paraméterek (távolság, forgatás)
- ✅ Dokumentált és tesztelt

**Hátrányok:**
- ⚠️ Lassabb: 2-3 másodperc/frame
- ⚠️ Kissé bonyolultabb: 572 sor kód
- ⚠️ Nem pontosan fizikailag helyes (egyszerűsített vetítés)

**Használat:**
```python
model = IrisPupilModel3D(width, height, camera_matrix)
params = model.fit_to_mask(ritnet_mask, method='nelder-mead')
unwrapped = model.unwrap_iris(frame, params)
```

**Teljesítmény:**
```
IoU: ~0.968
Speed: 2-3 sec (Nelder-Mead), 5-8 sec (DE)
```

---

### 3. Sphere-based Model (Physical)

**Fájl:** `iris_model_3d_v2.py`
**Osztály:** `EyeballModel3D`

**Leírás:**
- Fizikai szemgolyó modell (12mm sugár)
- Iris és pupilla a gömb felszínén
- Teljes 3D rotáció (pitch, yaw, roll)
- Rodrigues rotációs formula
- Perspektív vetítés

**Előnyök:**
- ✅ Fizikailag pontos szemgolyó geometria
- ✅ Teljes 3D rotációs szabadság
- ✅ Kamera kalibráció támogatás

**Hátrányok:**
- ❌ **Rossz pontosság**: IoU 0.724 (pupilla), 0.374 (iris)
- ❌ **Lassú**: 3-4 másodperc/frame
- ❌ **Komplex**: 580 sor kód
- ❌ Optimalizáció nehezen konvergál (túl sok paraméter)

**Használat:**
```python
model = EyeballModel3D(width, height, camera_matrix)
params = model.fit_to_mask(ritnet_mask, method='de')
```

**Teljesítmény szintetikus adaton:**
```
IoU: Pupil = 0.724, Iris = 0.374 ⚠️
Speed: 3-4 sec
Gaze: θ=43.4° φ=28.2°
```

**Probléma:** A túl sok szabad paraméter miatt az optimalizáció gyakran lokális minimumba ragad.

---

## Összehasonlító táblázat

| Modell | IoU Pupilla | IoU Iris | Sebesség | Komplexitás | Fizikai pontosság | Ajánlott? |
|--------|-------------|----------|----------|-------------|-------------------|-----------|
| **Ellipse-based** | **1.000** | **0.988** | **<0.5s** | Alacsony (434 sor) | ✅ Teljes | ✅ **IGEN** |
| Original | 0.968 | 0.968 | 2-3s | Közepes (572 sor) | ⚠️ Részleges | 🟨 Alternatíva |
| Sphere-based | 0.724 | 0.374 | 3-4s | Magas (580 sor) | ✅ Teljes | ❌ NEM |

---

## Matematikai háttér

### Probléma megfogalmazása

A szemben az iris és a pupilla **3D térben körök**. Azonban a 2D kamera képén ezek **ellipszisként** jelennek meg, ha a szem el van fordítva a kamerától.

**Fizikai valóság:**
- 3D térben: körök (egyenlő sugarak minden irányban)
- 2D kamera képen: ellipszisek (különböző major/minor tengelyek)

**Geometriai transzformáció:**
```
3D kör (r) + Forgatás (θ, φ) + Perspektív vetítés 
    → 2D ellipszis (a, b, angle)
```

### Megoldási stratégiák

#### 1. Ellipse-based: Közvetlen illesztés
- **Módszer:** Közvetlenül ellipszist illeszt a 2D kontúrra
- **Előny:** Nincs szükség komplex 3D → 2D transzformációra
- **Eredmény:** Tökéletes illeszkedés (IoU ~1.0)

#### 2. Original: Egyszerűsített 3D vetítés
- **Módszer:** 3D körök → egyszerűsített perspektív vetítés
- **Előny:** Gyors, közepes pontosság
- **Eredmény:** Jó illeszkedés (IoU ~0.97)

#### 3. Sphere-based: Teljes fizikai modell
- **Módszer:** Szemgolyó gömb + teljes 3D rotáció
- **Hátrány:** Túl sok paraméter, optimalizáció nehéz
- **Eredmény:** Rossz illeszkedés (IoU ~0.4-0.7)

---

## Tekintet becslés

Mind a három modell képes tekintet (gaze) becslésre:

### Ellipse-based modell:
```python
# Ellipszis arányból
cos(viewing_angle) ≈ minor_axis / major_axis

# Példa: 
# Ha minor/major = 0.7 → viewing_angle ≈ 45°
```

### Original/Sphere modell:
```python
# Forgatási szögekből
theta (pitch): függőleges tekintet
phi (yaw): vízszintes tekintet
```

---

## Ajánlások

### Általános használatra:
✅ **Ellipse-based Model** (`ellipse_iris_model.py`)
- Legjobb pontosság
- Leggyorsabb
- Legegyszerűbb

### Ha szükséges kamera kalibráció:
🟨 **Original Model** (`iris_model_3d.py`)
- Támogatja camera_matrix-ot
- Közepes pontosság és sebesség

### Ha szükséges teljes 3D modell:
⚠️ Vagy fejleszd tovább a Sphere-based modellt optimalizációs paraméter csökkentéssel,
vagy használd az Ellipse-based modellt + külön gaze estimation modult.

---

## GUI Használat

A `pipeline_tuner_gui.py`-ban kiválaszthatod melyik modellt használod:

1. Nyisd meg a GUI-t: `python3 pipeline_tuner_gui.py`
2. A **"7. 3D Iris Model"** szekcióban válaszd ki a modellt a dropdown menüből:
   - **"Ellipse-based (Best)"** - Ajánlott
   - **"Original (Simple 3D)"** - Alternatíva
   - **"Sphere-based (Physical)"** - Kísérleti (rossz IoU)

3. A modell infó mutatja:
   - IoU értékek
   - Sebesség
   - Rövid leírás

---

## Fejlesztési javaslatok

### Ellipse-based modell továbbfejlesztése:
1. **Kamera kalibráció támogatás** hozzáadása
2. **3D pozíció becslés** az ellipszis paraméterekből
3. **Pontosabb gaze estimation** több frame alapján (temporal filtering)

### Sphere-based modell javítása:
1. **Paraméter szám csökkentése** (fix eyeball radius, stb.)
2. **Jobb initial guess** az optimalizációhoz
3. **Hierarchikus optimalizáció** (először pozíció, aztán rotáció)

---

## Tesztelés

Mind a három modell tesztelhető szintetikus adaton:

```bash
# Ellipse-based
python3 ellipse_iris_model.py

# Original
python3 iris_model_3d.py

# Sphere-based
python3 iris_model_3d_v2.py
```

Minden teszt generál egy vizualizációt és kiírja az IoU értékeket.

---

## Konklúzió

Az **Ellipse-based Model** (`ellipse_iris_model.py`) jelenleg a legjobb választás:
- ✅ Tökéletes pontosság (IoU ~1.0)
- ✅ Gyors (<0.5s)
- ✅ Egyszerű és karbantartható
- ✅ Fizikailag helyes (3D körök → 2D ellipszisek)

Az eredeti és sphere-based modellek archiválhatók vagy tovább fejleszthetők specifikus use case-ekhez.

