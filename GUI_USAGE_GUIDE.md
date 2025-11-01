# 🎨 Pipeline Tuner GUI - Használati Útmutató

## 🚀 Indítás

```bash
python pipeline_tuner_gui.py
```

## 📋 Felület Áttekintés

### Bal Oldal: Kontrollok (5 Lépés)

#### **1. Image Selection** 📷
- **Frame Slider**: Csúsztasd a slidert, válassz ki egy frame-t a videóból (0-45648)
- Látod a total frames-t és FPS-t
- Aktuális frame azonnal betöltődik

#### **2. Glint Removal** ✨
- **Enable Glint Removal**: Ki/be kapcsolás
- **Threshold** (200-255): Fényesség küszöb - magasabb = csak nagyon fényes pontok
- **Min Area** (1-50): Minimum glint méret pixelben
- **Max Area** (50-500): Maximum glint méret pixelben
- **Morph Iterations** (1-10): Morfológiai műveletek száma - magasabb = agresszívebb

💡 **Tipp**: Kezdd 240 threshold-dal, 5-200 area-val, 3 iterációval

#### **3. Noise Reduction** 🔇
- **Enable Noise Reduction**: Ki/be kapcsolás
- **Módszer választás**:
  - **Bilateral Filter**: Élek megtartása + zajcsökkentés (AJÁNLOTT)
  - **Gaussian Blur**: Egyszerű simítás
  - **Median Blur**: Impulzuszaj ellen
- **Strength** (1-15): Szűrés erőssége - magasabb = simább

💡 **Tipp**: Bilateral filter, strength 5

#### **4. CLAHE / Histogram** 📊
- **Enable CLAHE**: Ki/be kapcsolás
- **Clip Limit** (0.5-10.0): Kontraszt erősítés - magasabb = drámaibb
- **Tile Size** (4-32): Grid méret - kisebb = lokálisabb

⚠️ **FIGYELEM**: CLAHE-val óvatosan! MediaPipe-nál gyakran nem kell!

💡 **Tipp**: Először próbáld CLAHE NÉLKÜL!

#### **5. Pupil Detection (AI)** 🤖
- **Módszer választás**:
  - **MediaPipe Iris**: AI-alapú iris detektálás (gyors, stabil)
  - **Traditional CV**: Hagyományos CV ellipse fitting
- **Show Landmarks**: Landmark pontok megjelenítése

### Jobb Oldal: Képek (3 Panel)

1. **Original Frame**: Eredeti frame a videóból
2. **After Preprocessing**: Előfeldolgozás után (glint + noise + CLAHE)
3. **Pupil Detection Result**: Végső detektálás eredménye

## 🎯 Munkafolyamat

### 1. Frame Kiválasztás
```
1. Csúsztasd a Frame Slider-t
2. Találj egy reprezentatív frame-t (középen lévő pupilla)
3. Jegyezd meg a frame számot
```

### 2. Glint Optimalizálás
```
1. Nézd meg az Original Frame-ben a fényes pontokat
2. Állítsd be a Threshold-ot hogy csak a glint-et kapja el
3. Min/Max Area-val szűrd a méret alapján
4. Iterations-szal finomítsd
5. Ellenőrizd az "After Preprocessing" panelen
```

### 3. Zajszűrés Finomítás
```
1. Kapcsold be a Noise Reduction-t
2. Válassz módszert (Bilateral ajánlott)
3. Strength-tel állítsd a simítás mértékét
4. Ne simítsd túl! Elvesztheted a részleteket
```

### 4. CLAHE Tesztelés
```
1. Először próbáld CLAHE NÉLKÜL!
2. Ha MediaPipe nem detektál, próbáld meg bekapcsolni
3. Ha hagyományos CV-t használsz, CLAHE segíthet
4. Clip Limit 2.0, Tile Size 8 jó kezdőérték
```

### 5. Detektálás Választás
```
1. Próbáld MediaPipe-ot először
2. Ha nem jó, váltsd Traditional CV-re
3. Show Landmarks-szal nézd meg a pontokat
4. Compare the results!
```

## 🧪 Teszt Funkciók

### 🔄 Update Preview
- Manuális előnézet frissítés
- Használd paraméter változtatás után

### 🧪 Test on 50 Frames
- Tesztel 50 frame-et az aktuális frame-től kezdve
- **Videót generál** side-by-side (Original | Detection)
- Megmutatja a detektálási rátát (%)
- Automatikusan felajánlja a videó megnyitását
- Fájl: `output/test_frames_START_to_END.mp4`

### 🧪 Test on 100 Frames
- Tesztel 100 frame-et
- **Videót generál** side-by-side formátumban
- Részletesebb statisztika
- Lassabb, de alaposabb
- Automatikusan megnyitható

### 🎬 Open Last Video
- Megnyitja az utoljára generált teszt videót
- Egy kattintással újranézhető
- Csak teszt futtatása után aktív
- macOS: QuickTime, Windows: Media Player, Linux: default player

### 💾 Save Settings
- Elmenti az összes beállítást `pipeline_settings.yaml`-ba
- Használható későbbi futtatásokhoz

### 📂 Load Settings
- Betölti a mentett beállításokat
- Visszaállítja a paramétereket

## 💡 Tippek & Trükkök

### MediaPipe nem detektál?
1. ❌ NE használj CLAHE-t!
2. ✅ Csökkentsd a noise reduction-t
3. ✅ Válassz másik frame-t (világosabb)
4. ✅ Ellenőrizd a glint removal-t (túl agresszív?)

### Traditional CV nem pontos?
1. ✅ Próbáld CLAHE-val (2.0, 8x8)
2. ✅ Növeld a noise reduction-t
3. ✅ Finomítsd a glint removal-t
4. ✅ Válassz frame-t ahol tisztán látszik a pupilla

### Fluktuál az eredmény?
1. Tesztelj 100 frame-en!
2. Nézd meg a detektálási rátát
3. Ha < 90% → finomíts paramétereken
4. Ha > 95% → jó beállítás! 🎉

### Túl lassú a preview?
1. Válassz kisebb frame számot teszteléshez
2. Kapcsold ki a Show Landmarks-t
3. Használd a Traditional CV-t (gyorsabb)

## 📊 Optimális Beállítások (Kiindulópont)

### MediaPipe Setup (AJÁNLOTT):
```yaml
glint:
  enabled: true
  threshold: 240
  min_area: 5
  max_area: 200
  morph_iterations: 3

noise:
  enabled: true
  method: bilateral
  strength: 5

clahe:
  enabled: false  # ❌ NE használd MediaPipe-hoz!

detection:
  method: mediapipe
  show_landmarks: true
```

### Traditional CV Setup:
```yaml
glint:
  enabled: true
  threshold: 240
  min_area: 5
  max_area: 200
  morph_iterations: 3

noise:
  enabled: true
  method: bilateral
  strength: 5

clahe:
  enabled: true  # ✅ Traditional CV-hez hasznos
  clip_limit: 2.0
  tile_size: 8

detection:
  method: traditional
  show_landmarks: false
```

## 🐛 Troubleshooting

### GUI nem indul?
```bash
pip install Pillow opencv-python mediapipe
```

### "No detection" message?
- Válassz másik frame-t
- Csökkentsd az előfeldolgozást
- Próbáld ki mindkét detektálási módszert

### Képek nem látszanak?
- Várj pár másodpercet (első betöltés lassú)
- Méretezd át az ablakot
- Kattints "Update Preview"-ra

### Test lefagy?
- Ez normális! 100 frame ~10-20 másodperc
- Türelem, a háttérben dolgozik
- Ne zárd be az ablakot!

## 🎓 Best Practices

1. **Kezdd egyszerűen**: Glint + Noise, CLAHE nélkül
2. **Tesztelj gyakran**: 50 frame teszt minden változtatás után
3. **Mentsd a beállításokat**: Ha jó eredmény, mentsd el!
4. **Dokumentáld**: Jegyezd meg melyik frame számmal dolgoztál
5. **Iterálj**: Próbáld ki különböző frame-eken is!

## 📁 Output Files

- `pipeline_settings.yaml`: Mentett beállítások
- `output/test_frames_START_to_END.mp4`: Teszt videók
  - Side-by-side: Original (bal) | Detection Result (jobb)
  - Frame számmal és detektálási státusszal
  - Real-time detection rate megjelenítés
  - Azonnal megnyitható egy kattintással
- GUI screenshot-ok: Készíts képernyőképet a jó beállításokról!

## 🎬 Videó Formátum

A generált teszt videók:
- **Felbontás**: 800x400 (2x 400x400 side-by-side)
- **FPS**: 111.84 (eredeti videó FPS)
- **Codec**: MP4V
- **Bal oldal**: Original frame + frame szám
- **Jobb oldal**: Detection result + státusz + detection rate
- **Státusz színek**: 
  - 🟢 Zöld "DETECTED" = sikeres detektálás
  - 🔴 Piros "NOT DETECTED" = sikertelen

---

**Készítette**: mrdudas  
**Verzió**: 1.0  
**Dátum**: 2025. november 1.

**Következő lépés**: Optimális beállítások után → full videó feldolgozás! 🚀
