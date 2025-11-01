# 👁️ Eye Corners Detection - Új Funkció

## ✅ Mi változott?

### 🔴 Eltávolítva: MediaPipe Iris
- MediaPipe **NEM** jó pupilla detektáláshoz
- Csak iris-t detektál, nem pupilla középpontot
- Túl nagy fluktuáció

### ✅ Hozzáadva: Traditional CV Pupilla + Eye Corners

## 📋 Új Pipeline Felépítés (6 Lépés)

### 1. Image Selection 📷
- Frame választás (változatlan)

### 2. Glint Removal ✨
- Fényes pontok eltávolítása (változatlan)

### 3. Noise Reduction 🔇
- Zajszűrés (változatlan)

### 4. CLAHE / Histogram 📊
- Kontraszt erősítés (változatlan)

### 5. Pupil Detection (Traditional CV) 👁️
**ÚJ PARAMÉTEREK:**
- **Threshold** (20-100): Binarizálási küszöb
  - Kisebb = világosabb pupillák is
  - Nagyobb = csak nagyon sötét
  - Ajánlott: 50
  
- **Min Area** (50-1000): Minimum pupilla terület
  - Kiszűri a zajt és kis objektumokat
  - Ajánlott: 100-200
  
- **Morph Kernel** (3-15): Morfológiai kernel méret
  - Simítja a pupilla kontúrt
  - Ajánlott: 5

### 6. Eye Corners Detection 🎯 **ÚJ!**
**Funkció:** Szem bal és jobb sarkának megtalálása

**Paraméterek:**
- **Enable Eye Corners Detection**: Ki/be kapcsolás
  
- **Módszerek:**
  - 🔵 **Harris Corner Detector**: Sarok pontok keresése
  - 🟢 **Good Features to Track** (Shi-Tomasi): AJÁNLOTT! Robosztus sarok detektálás
  - 🟡 **Template Matching**: Pupilla alapú sarok keresés
  
- **Quality Level** (0.001-0.1): Sarok minőség küszöb
  - Kisebb = több sarok
  - Nagyobb = csak erős sarkok
  - Ajánlott: 0.01
  
- **Min Distance** (10-200): Minimum távolság sarkok között
  - Megakadályozza hogy túl közel legyenek
  - Ajánlott: 50
  
- **Show Horizontal Axis**: Horizontális tengely megjelenítése
  - Vonal a bal és jobb sarok között
  - Középpont jelölése
  - Szög és távolság kiírása

## 🎯 Mit Detektál?

### Pupilla (5. lépés):
- ⚫ Pupilla kontúr (zöld ellipszis)
- 🔴 Pupilla centrum (piros pont)
- 📏 Átmérő (pixel)

### Eye Corners (6. lépés):
- 🔵 **L**: Bal szem sarok (kék pont)
- 🔵 **R**: Jobb szem sarok (kék pont)
- 🟡 **Horizontális tengely**: Sárga vonal
- 🔵 **Középpont**: Cián pont
- 📐 **Axis angle**: Tengely szöge (fok)
- 📏 **Eye width**: Szem szélesség (pixel)

## 🎨 Vizuális Eredmény

```
┌─────────────────────────────────────┐
│  L ←──────────────→ R               │
│  🔵        ⚫        🔵              │
│          (pupilla)                   │
│                                      │
│  Axis angle: 2.3°                   │
│  Eye width: 285.4px                 │
│  D: 64.5px                          │
└─────────────────────────────────────┘
```

## 🔧 Használat

### Gyors Start:
1. Válassz frame-t ahol tisztán látszik a szem
2. **Pupil Detection** beállítások:
   - Threshold: 50
   - Min Area: 150
   - Morph Kernel: 5
3. **Eye Corners** beállítások:
   - Enable: ✅
   - Method: Good Features to Track
   - Quality: 0.01
   - Min Distance: 50
   - Show Axis: ✅
4. Update Preview!

### Optimalizálás:

#### Ha nem látszik a pupilla:
- ✅ Csökkentsd a Threshold-ot (40-45)
- ✅ Csökkentsd a Min Area-t (100)
- ✅ Kapcsold be a CLAHE-t

#### Ha nem találja a sarkokat:
- ✅ Csökkentsd a Quality Level-t (0.005)
- ✅ Csökkentsd a Min Distance-t (30-40)
- ✅ Próbáld ki a másik módszert

#### Ha rossz sarkokat talál:
- ✅ Növeld a Quality Level-t (0.02-0.05)
- ✅ Növeld a Min Distance-t (70-100)
- ✅ Javíts a preprocessing-en (glint, noise)

## 📊 Módszerek Összehasonlítás

### Harris Corner Detector:
- ✅ Gyors
- ✅ Jól működik éles sarkokra
- ⚠️ Sok false positive
- ⚠️ Érzékeny a zajra

### Good Features to Track (Shi-Tomasi):
- ✅ **LEGJOBB általános használatra**
- ✅ Robosztus
- ✅ Kevés false positive
- ✅ Quality-based filtering
- ⚠️ Kicsit lassabb

### Template Matching:
- ✅ Pupilla-alapú (intelligens keresés)
- ⚠️ Csak ha pupilla detektálva
- ⚠️ Experimentális
- ⚠️ Lehet pontatlan

## 💡 Miért Fontos?

### Horizontális Tengely Használata:
1. **Normalizálás**: Pupilla pozíció relatív a szem szélességhez
2. **Rotáció korrekció**: Szög alapján el lehet forgatni
3. **Koordináta rendszer**: Saját referencia frame
4. **Összehasonlíthatóság**: Frame-ek között konzisztens mérés

### Példa Adatok:
```yaml
pupil:
  center: [200, 180]
  diameter: 64.5
  
eye_corners:
  left: [50, 175]
  right: [350, 182]
  axis_angle: 2.3°
  eye_width: 285.4px
  
normalized:
  pupil_x_relative: 0.526  # (200-50)/(350-50)
  pupil_y_offset: 1.75     # 180 - (175+182)/2
```

## 🧪 Teszt Eredmények

A teszt videó most tartalmazza:
- ✅ Pupilla detektálás (zöld ellipszis)
- ✅ Bal és jobb szem sarok (kék pontok)
- ✅ Horizontális tengely (sárga vonal)
- ✅ Szög és távolság információk

## 🚀 Következő Lépések

1. ✅ Optimalizáld a paramétereket
2. ✅ Futtass 50-100 frame tesztet
3. ✅ Ellenőrizd a detection rate-et
4. ✅ Mentsd el a beállításokat (Save Settings)
5. ➡️ **Következő**: Szem normalizálás és koordináta transzformáció

---

**Verzió**: 1.2 - Eye Corners Detection  
**Dátum**: 2025. november 1.  
**Státusz**: ✅ Ready for Testing!
