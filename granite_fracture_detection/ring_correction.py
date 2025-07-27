"""Removal of high‑frequency radial artefacts (ring artefacts).

In micro‑CT reconstructions, defective detector elements or calibration errors
produce *ring artefacts* – bright or dark stripes arranged in concentric
circles around the rotation axis.  Although sophisticated sinogram based
methods exist, a simple and robust post‑reconstruction technique is to
estimate the radial intensity variation in each slice and subtract its
high‑frequency component.  This module implements such a technique using
one‑dimensional smoothing of the radial profiles【893105350074866†L58-L140】.

The algorithm proceeds as follows for each z‑slice:

1. Compute the mean intensity at each integer radial distance within the
   slice.
2. Apply a one‑dimensional Gaussian filter to smooth the radial profile,
   capturing the slowly varying trend.
3. Take the difference between the original profile and its smoothed version
   (the high‑frequency ring artefact).
4. Subtract the high‑frequency component from every pixel in the slice.

This approach suppresses radial oscillations while leaving low‑frequency
variation (e.g. residual beam hardening) untouched.  It does not require
transforming the image to polar coordinates and therefore avoids associated
interpolation artefacts.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.typing import ArrayLike

__all__ = ["remove_radial_artifacts"]


def _radial_profile_slice(slice2d: np.ndarray, r_int: np.ndarray, r_max: int) -> np.ndarray:
    """Compute the mean intensity at each integer radial bin for a single slice."""
    profile = np.zeros(r_max + 1, dtype=np.float64)
    counts = np.zeros(r_max + 1, dtype=np.int64)
    for r in range(r_max + 1):
        mask = r_int == r
        if mask.any():
            profile[r] = slice2d[mask].mean()
        else:
            profile[r] = np.nan
    # Interpolate missing values
    if np.isnan(profile).any():
        valid = ~np.isnan(profile)
        x = np.nonzero(valid)[0]
        y = profile[valid]
        profile = np.interp(np.arange(r_max + 1), x, y)
    return profile


def remove_radial_artifacts(volume: ArrayLike, *, sigma: float = 3.0, centre: tuple[float, float] | None = None) -> np.ndarray:
    """Suppress high‑frequency radial oscillations (ring artefacts) from a volume.

    Parameters
    ----------
    volume : array‑like of shape (z, y, x)
        The input volume after beam‑hardening correction.  Intensity values
        should be approximately uniform in the background.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used to smooth the radial
        profiles.  Larger values remove wider rings but may oversmooth
        genuine intensity variations.
    centre : tuple of two floats or None
        Pixel coordinates ``(y_c, x_c)`` of the rotation axis.  If ``None``,
        the centre is assumed to be the geometric centre.

    Returns
    -------
    corrected : ndarray of shape (z, y, x)
        Volume with high‑frequency radial artefacts removed.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError("volume must be a 3‑D array (z, y, x)")
    z_dim, y_dim, x_dim = vol.shape

    # Determine centre coordinates
    if centre is None:
        y_c = (y_dim - 1) / 2.0
        x_c = (x_dim - 1) / 2.0
    else:
        y_c, x_c = centre

    # Precompute radial distances for all pixels
    yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
    r = np.sqrt((yy - y_c) ** 2 + (xx - x_c) ** 2)
    r_int = np.floor(r).astype(int)
    r_max = int(r_int.max())

    corrected = np.empty_like(vol)
    # Iterate over slices independently
    for z in range(z_dim):
        sl = vol[z]
        profile = _radial_profile_slice(sl, r_int, r_max)
        # Smooth profile to capture low‑frequency component
        smooth = gaussian_filter1d(profile, sigma=sigma, mode="nearest")
        # High frequency = original - smooth
        high_freq = profile - smooth
        # Subtract high‑frequency component from slice
        correction = high_freq[r_int]
        corrected[z] = sl - correction
    return corrected