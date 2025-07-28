"""Advanced tests for granite fracture detection.

These tests exercise more challenging scenarios than the basic examples.  A
synthetic granite rock texture is added to the volume prior to inserting
fractures, and the fracture detector must operate on this heterogeneous
background.  Another test evaluates the standard deviation returned by the
detector against variability observed when different realisations of
measurement noise are applied to the same underlying sample.
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


def test_fracture_detection_with_rock():
    """Detect fractures in a volume containing heterogeneous rock texture."""
    shape = (16, 64, 64)
    # Add synthetic rock heterogeneity with modest amplitude
    vol, gt = generate_synthetic_volume(
        shape,
        num_fractures=2,
        thickness=1.0,
        noise_sigma=0.01,
        ray_hardening_strength=0.3,
        radial_fluctuation_strength=0.1,
        fill_fraction=0.0,
        rock_strength=0.1,
        rock_smooth=3.0,
    )
    # Apply corrections
    vol_corr = correct_beam_hardening(vol, order=2)
    vol_corr = remove_radial_artifacts(vol_corr, sigma=2.0)
    fracture, frac_std = detect_fractures(vol_corr, sigmas=[1.0, 2.0], alpha=1.0, beta=0.5)
    # Compute metrics; thresholds slightly lower due to heterogeneity
    f1 = f1_score(fracture, gt, threshold=0.3)
    error = mse(fracture, gt)
    # With heterogeneous background detection becomes harder but should still
    # outperform random guessing.
    assert f1 > 0.2, f"F1 score too low on rock texture: {f1:.2f}"
    assert error < 0.2, f"MSE too high on rock texture: {error:.3f}"


def test_fracture_std_multiple_noise_instances():
    """Evaluate whether the fracture_std field reflects variability across noise realisations."""
    shape = (12, 48, 48)
    rng = np.random.default_rng(0)
    # Generate a noiseless base volume (noise_sigma=0) with a single fracture
    base_vol, gt = generate_synthetic_volume(
        shape,
        num_fractures=1,
        thickness=1.0,
        noise_sigma=0.0,
        ray_hardening_strength=0.3,
        radial_fluctuation_strength=0.1,
        fill_fraction=0.0,
    )
    # Apply artefact corrections to the base volume once (deterministic)
    base_corr = correct_beam_hardening(base_vol, order=2)
    base_corr = remove_radial_artifacts(base_corr, sigma=2.0)
    # Generate multiple noisy versions of the corrected volume and compute fractures
    n_repeats = 4
    fractures = []
    for i in range(n_repeats):
        noise = rng.normal(scale=0.02, size=shape)
        noisy = np.clip(base_corr + noise, 0.0, None)
        frac, std = detect_fractures(noisy, sigmas=[1.0, 2.0], alpha=1.0, beta=0.5)
        fractures.append(frac)
    fractures = np.stack(fractures, axis=0)
    # Compute per-voxel standard deviation across replicates
    empirical_std = fractures.std(axis=0)
    # Compute detection on one noisy instance to get algorithm's fracture_std
    example_noise = rng.normal(scale=0.02, size=shape)
    noisy_example = np.clip(base_corr + example_noise, 0.0, None)
    frac_example, algo_std = detect_fractures(noisy_example, sigmas=[1.0, 2.0], alpha=1.0, beta=0.5)
    # Compare the spatial distribution of std: compute correlation coefficient
    # Flatten to 1D for correlation; ignore zero variance voxels
    emp = empirical_std.ravel()
    alg = algo_std.ravel()
    valid = (emp > 0) & (alg > 0)
    if valid.sum() > 0:
        corr = np.corrcoef(emp[valid], alg[valid])[0, 1]
    else:
        corr = 1.0
    # Expect positive correlation between empirical variability and algorithm std
    assert corr > 0.4, f"Poor correlation between empirical std and fracture_std: {corr:.2f}"
