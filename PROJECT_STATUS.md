# Eye Tracking Pipeline - Pupilla Detektálás

## Áttekintés

Ez a projekt egy precíz pupilla detektáló és követő rendszer, amely videófájlokból képkockánként dolgozza fel és felismeri a pupillát.

## Jelenlegi Állapot - v0.1 (Alap Pipeline)

✅ **Elkészült komponensek:**
- Videó betöltés és képkocka-szintű feldolgozás
- Előfeldolgozás (glint eltávolítás, CLAHE kontrasztjavítás)
- Pupilla detektálás (edge detection + ellipszis illesztés)
- Eredmények mentése (JSON + annotált videó)
- Konfigurálható paraméterek (YAML)

## Telepítés

```bash
# Virtuális környezet aktiválása
source .venv/bin/activate

# Csomagok már telepítve vannak:
# opencv-python, numpy, matplotlib, scikit-image, scipy, pyyaml, tqdm
```

## Használat

### Gyors teszt (első 50 kép)
```bash
python test_pipeline.py
```

### Teljes videó feldolgozása
```bash
python pupil_pipeline.py
```

## Konfiguráció

A `config.yaml` fájlban állítható:
- Videó be/kimenet
- Glint eltávolítás paraméterei
- CLAHE kontrasztjavítás
- Edge detection (Canny)
- Ellipszis illesztés

## Kimenetek

A feldolgozás után az `output/` mappában:
- `pupil_data.json` - Detektált pupilla adatok minden képkockára
- `annotated_output.mp4` - Vizualizált videó az ellipszis és adatokkal

## Jelenlegi Teljesítmény

- **Videó:** eye1.mp4 (400x400, 111.84 FPS, 45649 képkocka)
- **Detektálási ráta:** 50/50 képkockában sikeres (teszt)
- **Feldolgozási sebesség:** ~80 FPS

## Következő Lépések

### Pontosság Javítása
1. **RANSAC alapú ellipszis illesztés** - robusztusabb detektálás outlierek ellen
2. **Subpixel pontosság** - edge refinement
3. **Temporal smoothing** - Savitzky-Golay vagy Kalman filter
4. **Deep learning** - DeepVOG vagy U-Net szegmentálás

### Kamera Kalibráció
5. Sakktábla/körrács alapú kalibráció
6. Torzítás korrekció implementálása

### Metrikák
7. Pupilla átmérő mm-ben (kalibrációval)
8. Konfidencia score javítása
9. Validáció referencia adatokkal

## Projekt Struktúra

```
eye_pipeline/
├── eye1.mp4              # Bemeneti videó
├── config.yaml           # Fő konfiguráció
├── pupil_pipeline.py     # Fő pipeline kód
├── test_pipeline.py      # Teszt script
├── output/               # Kimenetek
│   ├── pupil_data.json
│   └── annotated_output.mp4
└── readme.md            # Ez a fájl
```

## Technikai Részletek

### Előfeldolgozás
- Szürkeárnyalatos konverzió
- Glint detektálás (threshold-alapú)
- Inpainting (Telea algoritmus)
- CLAHE adaptive histogram equalization

### Pupilla Detektálás
- Canny edge detection
- Kontúr keresés és szűrés
- Ellipszis illesztés (cv2.fitEllipse)
- Konfidencia: kontúr/ellipszis terület arány

### Adatstruktúra
Minden képkockához:
- Képkocka szám
- Központ (x, y)
- Tengelyek (major, minor)
- Forgatás (angle)
- Konfidencia (0-1)

## Fejlesztő: mrdudas
**Dátum:** 2025. október 31.
