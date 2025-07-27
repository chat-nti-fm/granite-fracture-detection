# Granite Fracture Detection

This repository provides a Python implementation to correct common artefacts in micro‑CT images of cylindrical rock samples and to identify thin fractures whose thickness can be smaller than a single voxel.  The code is written in pure NumPy and SciPy to maximise performance.

## Motivation

In X‑ray computed tomography (CT) of rock cores, the measured attenuation suffers from imaging artefacts such as beam hardening and ring artefacts.  Beam hardening, also known as **ray hardening**, produces a “cupping” effect where the interior of the sample appears darker than the edges because low‑energy photons are preferentially absorbed【284965845346994†L168-L178】.  This variation scales with the radial distance from the centre of the cylindrical core【87478035248452†L115-L139】.  Ring artefacts originate from defective detector elements and appear as high‑frequency intensity oscillations along circular paths around the rotation axis【893105350074866†L58-L140】.

Fractures in granite are often narrow and complex; their apertures can be at or below the voxel size.  Traditional global thresholding methods cannot reliably segment such features because of partial‑volume effects.  State‑of‑the‑art approaches for fracture detection exploit correlations along unknown, fractal fracture surfaces using multi‑scale Hessian filters or iterative local thresholding【656784612022342†L28-L43】【4781395552577†L14-L33】【739475936590376†L510-L533】.

This project implements three steps:

1. **Beam hardening (cupping) correction** – Estimates and removes the radially varying bias caused by beam hardening.
2. **Ring artefact removal** – Suppresses high‑frequency radial oscillations while preserving low‑frequency components.
3. **Fracture detection** – Applies a multi‑scale Hessian‑based filter to identify dark, sheet‑like structures and reports both the estimated fracture fraction per voxel and its uncertainty.

The package also includes synthetic test generation, automated evaluation of noise removal and fracture detection accuracy, and unit tests.  When using real CT data, the same functions can be applied directly to 3‑D NumPy arrays.

## Installation

The project uses only NumPy and SciPy, which are available by default in most scientific Python distributions.  To install the package and its development dependencies, run:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy pytest
```

## Usage

```python
import numpy as np
from granite_fracture_detection import beam_hardening, ring_correction, fracture_detection

# `volume` is a 3‑D NumPy array (z, y, x) representing reconstructed CT intensities
volume = ...  # Load your volume here

# Step 1: Correct radial beam hardening
volume_corrected = beam_hardening.correct_beam_hardening(volume)

# Step 2: Remove ring artefacts
volume_denoised = ring_correction.remove_radial_artifacts(volume_corrected)

# Step 3: Detect fractures
fracture, fracture_std = fracture_detection.detect_fractures(volume_denoised, sigmas=[1.0, 1.5, 2.0])

# `fracture` contains the estimated fraction of each voxel occupied by a fracture (0–1)
# `fracture_std` contains the estimated uncertainty across scales
```

Synthetic tests and metrics can be executed via:

```bash
pytest
```

## License

This project is released under the MIT license.