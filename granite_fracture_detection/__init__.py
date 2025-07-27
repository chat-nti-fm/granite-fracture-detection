"""Granite fracture detection package.

This package provides high‑level functions to correct beam hardening and ring artefacts
from CT volumes of cylindrical rock cores and to detect thin fractures.  See
``README.md`` in the project root for an introduction.

Modules:

- :mod:`beam_hardening` – Radial intensity correction (cupping/beam hardening).
- :mod:`ring_correction` – Removal of high‑frequency radial artefacts.
- :mod:`fracture_detection` – Multi‑scale Hessian‑based fracture extraction.
- :mod:`synthetic` – Synthetic data generation for testing.
- :mod:`metrics` – Utility functions to evaluate algorithms.

"""

from .beam_hardening import correct_beam_hardening
from .ring_correction import remove_radial_artifacts
from .fracture_detection import detect_fractures
from .synthetic import generate_synthetic_volume
from .metrics import snr, f1_score, mse

__all__ = [
    "correct_beam_hardening",
    "remove_radial_artifacts",
    "detect_fractures",
    "generate_synthetic_volume",
    "snr",
    "f1_score",
    "mse",
]