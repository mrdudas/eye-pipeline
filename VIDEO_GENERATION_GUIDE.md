# 🎬 Pipeline Tuner - Videó Generálás

## ✅ Új Funkciók

### 1. Automatikus Videó Generálás 🎥
- **Test on 50 Frames** → automatikusan videót készít
- **Test on 100 Frames** → automatikusan videót készít
- Side-by-side megjelenítés: Original | Detection Result

### 2. Videó Információk 📊
**Bal oldal (Original)**:
- Frame szám
- "Original" felirat
- Nyers frame (előfeldolgozás előtt)

**Jobb oldal (Detection)**:
- Detektálási eredmény (MediaPipe vagy Traditional CV)
- "DETECTED" (zöld) vagy "NOT DETECTED" (piros)
- Real-time detection rate (%)
- Landmark pontok (ha be van kapcsolva)

### 3. Egy-Kattintásos Megnyitás 🖱️
- Teszt után: "Open video now?" → Yes/No
- **🎬 Open Last Video** gomb → bármikor újra megnyitható
- Automatikusan a rendszer default lejátszóját használja:
  - macOS: QuickTime Player
  - Windows: Windows Media Player
  - Linux: Default video player

### 4. Fájlnév Konvenció 📝
```
output/test_frames_0_to_49.mp4      (50 frames, 0-tól kezdve)
output/test_frames_100_to_199.mp4   (100 frames, 100-tól kezdve)
output/test_frames_500_to_549.mp4   (50 frames, 500-tól kezdve)
```

## 🎯 Használat

### Gyors Teszt:
1. Válassz ki egy frame-t (pl. 100)
2. Állítsd be a paramétereket
3. Kattints: **🧪 Test on 50 Frames**
4. Várj ~5-10 másodpercet
5. Kérdés: "Open video now?" → **Yes**
6. ✅ Videó megnyílik a lejátszódban!

### Videó Újranézése:
1. Kattints: **🎬 Open Last Video**
2. ✅ Utolsó teszt videó újra megnyílik!

### Több Teszt Készítése:
1. Tesztelj különböző paraméterekkel
2. Tesztelj különböző frame tartományokon
3. Minden teszt új videót készít
4. Össze tudod hasonlítani őket!

## 📊 Videó Előnyei

### Real-time Monitoring:
- Látod hogy minden frame-en hogyan működik
- Észreveszed a problémákat (hol nem detektál)
- Látod a fluktuációt

### Paraméter Optimalizálás:
- Készíts videót paraméter A-val
- Készíts videót paraméter B-vel
- Nézd meg side-by-side melyik jobb!

### Dokumentáció:
- Megmutatható eredmények
- Validálható működés
- Reprodukálható tesztek

## 🎨 Példa Workflow

```
1. Frame Selection: 500
2. Beállítások:
   - Glint: ON (threshold=240)
   - Noise: Bilateral (strength=5)
   - CLAHE: OFF
   - Detection: MediaPipe
3. Test on 50 Frames
4. Videó: output/test_frames_500_to_549.mp4
5. Nézd meg → "Hmm, frame 520-nál nem detektál..."
6. Finomíts paramétereken
7. Test on 50 Frames újra
8. Videó: output/test_frames_500_to_549.mp4 (felülírva)
9. Hasonlítsd össze → "Most már jobb!"
10. Save Settings → pipeline_settings.yaml
```

## 💡 Tippek

### Jó Teszt Készítéséhez:
- ✅ Válassz reprezentatív frame tartományt
- ✅ Tesztelj különböző fényerősségű részeken
- ✅ 50 frame = gyors, 100 frame = alaposabb
- ✅ Nézd végig a videót, keress problémákat!

### Videó Elemzéshez:
- 🔍 Állj meg érdekes frame-eknél
- 🔍 Figyeld a detection rate változását
- 🔍 Nézd meg hol "NOT DETECTED"
- 🔍 Ellenőrizd a landmark pontok helyét

### Optimalizáláshoz:
- 📊 Készíts videót minden nagy változtatás után
- 📊 Hasonlítsd össze a detection rate-eket
- 📊 Dokumentáld a jó beállításokat
- 📊 Mentsd el a settings-t!

## 🚀 Következő Lépések

Miután megtaláltad az optimális beállításokat:

1. **Save Settings** → `pipeline_settings.yaml`
2. Készíts egy finális tesztet 100 frame-en
3. Ha detection rate > 95% → SIKER! 🎉
4. Használd ezeket a paramétereket a teljes videó feldolgozásához

---

**Készült**: 2025. november 1.  
**Verzió**: 1.1 - Videó generálás  
**Státusz**: ✅ Production Ready!
