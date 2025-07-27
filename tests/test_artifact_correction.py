"""Tests for beam hardening and ring artefact correction.

These tests create synthetic volumes containing only artefacts (no fractures)
and verify that the correction functions reduce the noise level and improve
similarity to the ground‑truth data.  The SNR is used as a quantitative
metric; the corrected volume should have a higher SNR than the raw volume.
"""

from __future__ import annotations

import numpy as np

from granite_fracture_detection import (
    correct_beam_hardening,
    remove_radial_artifacts,
    generate_synthetic_volume,
    snr,
)


def test_artifact_correction_improves_snr():
    # Generate a synthetic volume with only artefacts (no fractures)
    shape = (16, 64, 64)
    vol, gt = generate_synthetic_volume(
        shape,
        num_fractures=0,
        noise_sigma=0.01,
        ray_hardening_strength=0.6,
        radial_fluctuation_strength=0.3,
    )
    # Ground truth is uniform intensity of 1.0 since no fractures
    gt_uniform = np.ones_like(gt)
    # Compute SNR of raw data vs ground truth
    snr_before = snr(gt_uniform, vol)
    # Apply beam hardening correction and ring artefact removal
    vol_corr = correct_beam_hardening(vol, order=2)
    vol_corr = remove_radial_artifacts(vol_corr, sigma=2.0)
    snr_after = snr(gt_uniform, vol_corr)
    # Ensure that SNR improves after correction by at least a few dB
    assert snr_after > snr_before + 3.0, f"SNR improvement too low: {snr_before:.2f} -> {snr_after:.2f} dB"
