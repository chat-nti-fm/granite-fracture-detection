"""Tests for the multi‑scale fracture detection algorithm.

The detection is evaluated on synthetic volumes containing a number of
planar fractures.  The algorithm should produce a fracture map that
correlates with the ground truth.  We do not expect perfect segmentation
because the simulated fractures may be partly occluded by noise and
artefacts, but the F1 score should exceed a baseline threshold and the
mean squared error should be reasonable.
"""

from __future__ import annotations

import numpy as np

from granite_fracture_detection import (
    correct_beam_hardening,
    remove_radial_artifacts,
    detect_fractures,
    generate_synthetic_volume,
    f1_score,
    mse,
)


def test_fracture_detection_metrics():
    # Generate a small synthetic volume with a couple of fractures
    shape = (16, 64, 64)
    vol, gt = generate_synthetic_volume(
        shape,
        num_fractures=2,
        thickness=1.0,
        noise_sigma=0.01,
        ray_hardening_strength=0.3,
        radial_fluctuation_strength=0.1,
        fill_fraction=0.0,
    )
    # Apply artefact corrections
    vol_corr = correct_beam_hardening(vol, order=2)
    vol_corr = remove_radial_artifacts(vol_corr, sigma=2.0)
    # Detect fractures
    fracture, frac_std = detect_fractures(vol_corr, sigmas=[1.0, 2.0], alpha=1.0, beta=0.5)
    # Compute evaluation metrics
    f1 = f1_score(fracture, gt, threshold=0.3)
    error = mse(fracture, gt)
    # The F1 score should be better than random guessing (e.g. >0.3)
    assert f1 > 0.3, f"F1 score too low: {f1:.2f}"
    # MSE should be less than 0.1 for meaningful detection
    assert error < 0.1, f"MSE too high: {error:.3f}"
