# DATASET_LAYOUT.md

Dieses Dokument beschreibt das erwartete Datenlayout für RGB‑D‑Daten, Ground‑Truth und Modell‑/Template‑Dateien.

## 1) BASE_PATH (RGB‑D + Ground‑Truth)
```
BASE_PATH/
└─ <item>/
   └─ rgbddata/
      ├─ <item><id>_color.npy
      ├─ <item><id>_depth.npy
      └─ <item><id>.txt
```
- `<item>`: Objektname (z. B. `banana`, `apple`, `bottle`)
- `<id>`: Integer ohne führende Nullen (z. B. `1`, `23`, `147`)
- `*_color.npy`: `(H, W, 3)`, `uint8`
- `*_depth.npy`: `(H, W)`, `float32`, **Meter**
- `<item><id>.txt`: 4×4 Homogenmatrix `T_gt` (row‑major), Pose des **Objekts im Kameraframe**

## 2) DATA_PATH (Modelle & Templates)
```
DATA_PATH/
└─ <item>/
   ├─ <item>.ply
   └─ <item>_template_<k>_az<az-value>_el<el-value>.ply         # k = 0,1,2,... ; 
                                                                # az-value = azimuth [°]
                                                                # el-value = elevation [°]
```

## 3) Kamera (camera.yaml)
Mehrere Kameras als Schlüssel möglich (z. B. `my_rgbd_camera`, `realsense_d435`, …).  
Siehe `camera.yaml.example`.

**Konventionen**
- Open3D/Kamera‑Frame (rechtshändig): `+X` rechts, `+Y` unten, `+Z` vorwärts.
- Einheiten: Translation [m], Tiefe [m], Rotation [°] (für Berichte/CSV).
