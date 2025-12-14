# Stereo-calibration

Lightweight utilities for stereo calibration: feature extraction/matching, an 8‑point fundamental matrix estimator, and essential matrix decomposition with pose selection.

Overview
- Purpose: extract and match features, estimate the fundamental matrix from matches, compute the essential matrix from intrinsics, and obtain camera pose candidates.
- Scope: research/prototype code. Includes a neural-feature wrapper (SuperPoint/SuperGlue) plus a deterministic ORB fallback used by the example/demo.

Requirements
- Basic: `numpy`, `opencv-python` (used by ORB example and image I/O).
- Optional (neural features): `torch` and `kornia` to enable `feature.py` SuperPoint/SuperGlue functionality.
- See `requirements.txt` for suggested packages.

Install (example)
```bash
# using pip
pip install -r requirements.txt
# optional: install neural feature deps
pip install torch kornia
```

Running the demo
- The repository contains a small ORB-based demo at `demo.py` which does not require `torch`/`kornia`.
- From the project root, run (paths shown relative to repo root):

```bash
cd stereo-calibration
python demo.py --left examples/left_synth.jpg --right examples/right_synth.jpg
```

Notes on `-m` usage and package name
- The folder is named `stereo-calibration` (contains a hyphen). Python module names cannot contain `-`, so `python -m stereo-calibration...` will fail. Use one of:
	- Run the script directly as above (preferred), or
	- Rename the folder to a valid package name (e.g., `stereo_calibration`) and add `__init__.py` if you want `-m` style imports.

Files
- `calibration.py`: `StereoCalibration` computes the essential matrix from a fundamental matrix and intrinsics and provides pose selection logic.
- `estimator.py`: `estimate_fundamental_matrix(matches)` implements an 8‑point style linear solve and enforces rank‑2; `residuals(F, matches)` computes algebraic error.
- `feature.py`: wrapper for SuperPoint/SuperGlue (lazy-loaded) with `feature_extraction`, `feature_matching`, and `topk_matches`. Raises `ImportError` when neural deps are missing.
- `main.py`: CLI-friendly wiring function `stereo_calibration(...)` (supports SuperPoint/SuperGlue when installed).
- `demo.py`: ORB-based end-to-end example (no neural deps).
- `requirements.txt`: minimal/pinned packages used for examples.

Example
```python
import cv2
import numpy as np
from main import stereo_calibration

imgL = cv2.imread('left.jpg')
imgR = cv2.imread('right.jpg')
k = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])

poses = stereo_calibration(imgL, imgR, k, k)
print(poses)
```
