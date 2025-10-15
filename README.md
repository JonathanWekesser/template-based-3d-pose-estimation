# Schätzung der initialen Pose eines Objekts in einer 3D-Punktwolke
**Bachelorarbeit – Jonathan Wekesser (Hochschule Ravensburg-Weingarten, Institut für Künstliche Intelligenz)**

Dieses Repository enthält den Code zur **Schätzung einer initialen 6D-Pose** eines bekannten Objekts in einer RGB-D-Szene anhand einer rekonstruierten Punktwolke.
Die Methode kombiniert klassische, nicht‑neuronale Verfahren der Punktwolkenregistrierung (FPFH, RANSAC, ICP) mit einer optionalen Bildsegmentierung (YOLO‑Segmentation).
Zusätzlich werden **Teilmodell‑Templates** unterstützt, um Robustheit gegenüber Verdeckungen und Selbstähnlichkeiten zu erhöhen.

> Zielkontext: *Initiale Pose* als Startwert für nachgelagerte, ggf. präzisere Optimierer/Tracker in der Roboter‑Greifplanung.

---

## Inhaltsverzeichnis
- [Hauptfunktionen](#hauptfunktionen)
- [Projektstruktur](#projektstruktur)
- [Schnellstart](#schnellstart)
- [Konfiguration](#konfiguration)
- [Datenlayout & Kamera](#datenlayout--kamera)
- [Experimente ausführen](#experimente-ausführen)
- [Ausgabe & Evaluationsmetriken](#ausgabe--evaluationsmetriken)
- [Troubleshooting](#troubleshooting)

---

## Hauptfunktionen
- **Vorverarbeitung & Merkmalsextraktion**: Voxel‑Downsampling, Normalenschätzung, FPFH‑Features.
- **Zweistufige Registrierung**: Grobausrichtung via RANSAC, Verfeinerung via ICP (Point‑to‑Point/Point‑to‑Plane).
- **Teilmodell‑Templates**: Laden und Durchprobieren mehrerer Modellansichten.
- **Segmentierung (optional)**: YOLO‑Segmentierung zur Maskierung der Szene vor der Tiefen‑zu‑Punktwolke‑Rekonstruktion.
- **Evaluation**: Fitness, Inlier‑RMSE, Translations‑/Rotationsfehler, Korrespondenzanzahl, Laufzeit.
- **Batch‑Runner**: Reproduzierbare Experimente über Datensatz‑IDs, CSV‑Export, optionale Visualisierung/Rendering.

---

## Projektstruktur
```text
repo-root/
├─ src/
│  ├─ camera.py                # Camera intrinsics handling (load from YAML)
│  ├─ config.py                # Dataclass für Registrierungs-Parameter + YAML-Loader
│  ├─ dataset_loader.py        # RGB, Depth, T_gt für eine ID laden
│  ├─ evaluation.py            # Kennzahlen & Fehlermaße
│  ├─ features.py              # FPFH-Extraktion
│  ├─ mask.py                  # Maskenanwendung auf Bilder
│  ├─ paths.py                 # Pfadverwaltung via .env (BASE_PATH, DATA_PATH, …)
│  ├─ pose_initializer.py      # Pipeline: Preprocess → FPFH → RANSAC → ICP → Evaluation
│  ├─ preprocessing.py         # Filter, Downsampling, Normalen
│  ├─ registration.py          # RANSAC/ICP Wrapper (Open3D)
│  ├─ runner.py                # ExperimentRunner: Batch-Ausführung, CSV, Render
│  ├─ segmentation.py          # YOLO-Segmentation (optional)
│  └─ template.py              # TemplateLoader (Vollmodell + Teilmodelle)
├─ data/                       # (extern konfiguriert) Kameras, Modelle, Templates
├─ docs/                       # Die Thesis (PDF) und weitere READMEs
├─ models/                     # YOLO-Gewichte (z. B. yolo11n-seg.pt)
├─ notebooks/                  # (optional) Analyse/Visualisierung
├─ configs/                    # (optional) YAML-Konfigurationen
├─ results/                    # (wird erstellt) CSVs & Visualisierungen
├─ .env.example                # Beispiel für Pfad-Variablen
├─ .python-version             # UV - Python-Version pinnen
├─ pyproject.toml              # UV - Projektdetails
├─ uv.lock                     # UV - Generierte Datei für Dependencies
└─ README.md                   # Diese Datei
```

---

## Schnellstart
### 1) Python-Umgebung mit uv
```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2) Pfade konfigurieren
Erstellen Sie eine Datei **`.env`** im Repository‑Wurzelverzeichnis (siehe [.env.example](./.env.example)) und setzen Sie mindestens:
- `BASE_PATH` – Basisordner je Objekt (RGB‑D‑Daten und Ground‑Truth pro ID).
- `DATA_PATH` – Ordner mit **Modellpunktwolken** (`{item}.ply`) und **Template‑.ply**s.
- Optional: `CAMERA_PATH` – Pfad zur `camera.yaml` (falls abweichend).

> **Hinweis:** Kameraparameter werden über eine YAML geladen (vgl. `camera.py`).

### 3) YOLO-Gewichte (optional, für Segmentierung)
Legen Sie die Datei `yolo11n-seg.pt` in `./models/` ab (oder passen Sie den Pfad in `segmentation.py` an).

---

## Konfiguration
Die Registrierungsparameter sind als Dataclass definiert (siehe `src/config.py`). Eine externe YAML kann via `load_config_yaml(path)` geladen werden.

**Beispiel `configs/universal.yaml`:**
```yaml
# RegistrationConfig fields
voxel_size: 0.005
stat_nb_neighbors: 50
stat_std_ratio: 0.7
normal_radius: 0.1
normal_max_nn: 30
fpfh_radius: 0.025
fpfh_max_nn: 100
ransac_corr_dist: 0.008
ransac_n: 4
ransac_edge_length: 0.9
ransac_max_iter: 100000
ransac_confidence: 0.99
icp_variant: "p2p"      # "p2p" | "p2l"
icp_corr_dist: 0.005
icp_max_iter: 100
visualize: false
```

**Programmatisches Laden:**
```python
from src.config import load_config_yaml
cfg = load_config_yaml("configs/universal.yaml")
```

---

## Datenlayout & Kamera

### A) Verzeichnis- und Dateinamenkonvention (RGB‑D + Ground‑Truth)
Der `dataset_loader.py` erwartet standardisierte Namen innerhalb von `BASE_PATH/<item>/rgbddata/`:

```
BASE_PATH/
└─ <item>/
   └─ rgbddata/
      ├─ <item><id>_color.(png|jpg|npy)
      ├─ <item><id>_depth.(png|npy)
      └─ <item><id>.txt               # 4x4 Pose-Matrix T_gt (row-major)
```

- `<item>`: Objektname (z. B. `banana`, `apple`, `bottle`).
- `<id>`: Ganzzahlige ID ohne führende Nullen (z. B. `1`, `23`, `147`).  
- **Farbkanal**: RGB in `uint8` als `*.png`/`*.jpg` **oder** als NumPy‐Array `*.npy` mit Shape `(H, W, 3)`.
- **Tiefenkanal**:  
  - `*.png`: **Millimeter** als `uint16` (z. B. OpenNI/RealSense‑Konvention).  
  - `*.npy`: Float‑Meter (`float32`) mit Shape `(H, W)`.
- **Ground‑Truth**: Datei `<item><id>.txt` enthält eine **4×4 Homogenmatrix** `T_gt` (row‑major), die die Pose des Objekts im **Kamerakoordinatensystem** beschreibt.

**Beispiel (`banana`):**
```
BASE_PATH/banana/rgbddata/
  banana1_color.png
  banana1_depth.png
  banana1.txt
```

> **Tiefe/Skalierung:** Bei `depth.png` in mm wird intern auf **Meter** skaliert (`depth_m = depth_mm / 1000.0`).

### B) Modell- und Template-Pfade
Im `DATA_PATH/<item>/` werden die Modell‑Punktwolken abgelegt:
```
DATA_PATH/
└─ <item>/
   ├─ <item>.ply                     # Vollmodell
   └─ <item>_template_<k>.ply        # Teilmodell-Templates (k = 0,1,2,...)
```

### C) Kamera‑Konfiguration (`camera.yaml`)
Die Datei beschreibt **intrinsische** Parameter. 
Mehrere Kameras können als Schlüssel abgelegt werden. 
Die Intel® RealSense™ Tiefenkamera D435i (kurz D435i) ist unter [`camera.yaml`](./data/camera_intrinsics.yaml) abgelegt.

---

**Wichtige Konventionen:**
- **Koordinatenachsen** (Open3D/Kamera‑Frame, rechtshändig): `+X` nach rechts, `+Y` nach unten, `+Z` nach vorne (weg von der Kamera).  
- **`T_gt`** in `<item><id>.txt` beschreibt die Pose **Objekt im Kameraframe**.  
- **Einheiten**: Tiefenmeter [m], Translation [m], Rotation [°] (für Berichte/CSV entsprechend angegeben).

Eine eigenständige, komprimierte Referenz mit Beispielen ist in [`DATASET_LAYOUT.md`](./docs/DATASET_LAYOUT.md) enthalten.

---

## Experimente ausführen
### Minimalbeispiel (Batch‑Eval mit Templates)
```python
import logging
from src.camera import Camera
from src.config import load_config_yaml
from src.paths import PathManager
from src.template import TemplateLoader
from src.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)

item = "banana"                                # Objektname (entspricht Verzeichnisnamen)
pm = PathManager(item)                         # Nutzt .env
cam = Camera.from_yaml(pm.get_camera_path(), "my_rgbd_camera")

cfg = load_config_yaml("configs/universal.yaml")
templates = TemplateLoader(pm).load_all()     # Vollmodell + Teilmodelle laden

runner = ExperimentRunner(cfg, item, cam, templates, out_dir="./results")
# Bereich der Datensatz-IDs (inklusiv)
rows = runner.run_range(idmin=1, idmax=100, return_only_best=False)
```

### Nur bestes Template pro ID speichern
```python
best = runner.run_range(idmin=1, idmax=100, return_only_best=True)
```

### Visualisierung/Rendering aktivieren
Setzen Sie `visualize: true` in der Konfiguration. Pro Datensatz/Template wird eine PNG unter `./results/visualization/` erzeugt.

---

## Ausgabe & Evaluationsmetriken
- **`results.csv`** mit u. a.:
  - `fitness`, `inlier_rmse`, `translation_error` [m], `rotation_error` [°], `correspondence_set_size`, `duration` [s]
- **Interpretation** (Kurzüberblick):
  - Höhere **Fitness** → bessere Überdeckung.
  - Niedriger **Inlier‑RMSE** → geringere Abweichung der korrespondierenden Punkte.
  - **Translations‑/Rotationsfehler** werden nur berechnet, wenn *Ground‑Truth* vorhanden ist (`T_gt`).

---

## Troubleshooting
- **Skalierungsfehler in der Tiefe**: Prüfen Sie `depth.scale` in `camera.yaml` sowie das Tiefenformat (`png` in mm vs. `npy` in m).
- **Leere Templates**: Stellen Sie sicher, dass `DATA_PATH/<item>/<item>_template_*.ply` existiert und lesbar ist.
- **Open3D ohne Headless‑Rendering**: In Serverumgebungen Visualisierung deaktivieren (`visualize: false`) oder OSMesa/Filament installieren.
- **CUDA/ROCm**: Dieses Repo benötigt keine GPU zwingend. Für YOLO‑Inference mit GPU bitte passende PyTorch‑Pakete installieren.
- **Dateipfade stimmen nicht**: `.env` prüfen, `PathManager` verwendet `BASE_PATH`/`DATA_PATH` unverändert.
